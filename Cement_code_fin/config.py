"""
Column mapping, control/monitor split, spec bands, and physical cleaning rules for the
Chronos-2 cement quality net.

This is the cleaned-up ("_fin") rebuild of chronos_quality/. Only the canonical pipeline
survives here -- all the A/B experiment knobs (statistical IQR removal, damper clip vs NaN,
ProdType drop vs mask, prev-quality sparse vs ffill) that accumulated in the old folder are
gone. Add a branch back only when an experiment actually needs it.

Canonical decisions carried over from chronos_quality/EXPERIMENT_LOG.md:
  - SOURCE_XLSX = "운전&품질데이터_실측치.xlsx" (2026-09-05 measured-actual export; a controlled
    comparison showed swapping to it makes every zero-shot/LoRA/full combo beat the naive
    baseline, where the old "운전 & 품질 데이터_2026.xlsx" had non-measurement-hour values
    bleeding in and inflating the baseline).
  - No statistical IQR outlier removal -- Chronos-2 masks NaN natively, so deleting
    extreme-but-real values just discards training signal. Only the fixed physical/equipment
    limits in CLEANING_RULES are applied.
  - prev-quality covariates (blaine_prev / residue_prev) kept SPARSE: present only at the
    scheduled 4-hour measurement rows, NaN on every hourly row between.

STD_COLUMNS_BY_POSITION is positional against the raw 2CM/3CM/4CM sheet headers. CONTROL_COLS /
MONITOR_COLS / TARGET_COLS use the 1st-year ANN's RP_*/mill_*/blaine/residue short codes so this
pipeline lines up with the ANN. A few columns have no confirmed raw<->ANN mapping and keep their
POLYCOM_/MILL_ names (marked inline).
"""

from __future__ import annotations

import os
from pathlib import Path


def default_device_map() -> str:
    """Auto-detect GPU vs CPU so scripts run unmodified on whichever machine executes them."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_chronos_pipeline(checkpoint: str, device_map: str):
    """
    Load a Chronos2Pipeline from either the HF hub base model or a local fine-tuned checkpoint.

    Local checkpoints saved with finetune_mode="lora" are PEFT adapters, and reloading them
    requires passing `import_allowlist=["chronos.chronos2.model"]` (a security allowlist check
    added by peft). That kwarg is NOT accepted when loading the base HF model or a full-finetune
    checkpoint, so we retry with it only if the plain load fails with the allowlist error.
    """
    from chronos import Chronos2Pipeline

    try:
        return Chronos2Pipeline.from_pretrained(checkpoint, device_map=device_map)
    except ValueError as e:
        if "import_allowlist" not in str(e):
            raise
        return Chronos2Pipeline.from_pretrained(
            checkpoint, device_map=device_map, import_allowlist=["chronos.chronos2.model"]
        )


# Raw xlsx source. Only data/build_dataset.py needs it, and only to (re)generate
# data/processed/quality_timeseries.csv. The processed CSV travels with the folder, so on a
# machine without the raw file training/eval still work. Override via env var CHRONOS_SOURCE_XLSX.
#
# "운전&품질데이터_실측치.xlsx" lives one level above this project folder (i.e. in the Cement/
# folder, next to Cement_code_fin/). Sheet layout is identical to the older
# "운전 & 품질 데이터_2026.xlsx": single header row, NO "근무자"(Worker) column, operating data
# on the 2CM/3CM/4CM sheets.
SOURCE_XLSX = os.environ.get(
    "CHRONOS_SOURCE_XLSX",
    str(Path(__file__).resolve().parents[1] / "운전&품질데이터_실측치.xlsx"),
)
OPERATING_SHEETS = ["2CM", "3CM", "4CM"]  # trailing-space-free sheet names = hourly operating data

# 2CM/4CM sheets: headers on row 2, data from row 3. The 3CM sheet has no header row (data from
# row 1), so build_dataset.py detects the real data-start row per sheet by scanning for the
# "근무일자" header cell instead of trusting DATA_START_ROW blindly.
HEADER_ROW = 2
DATA_START_ROW = 3

# Raw (line-broken) xlsx header -> standardized column name, for the unambiguous columns.
# Order matches the physical column order in the 2CM/3CM/4CM operating sheets. The full
# authoritative positional list is STD_COLUMNS_BY_POSITION below (it also covers the columns that
# collide under openpyxl's text collapse, e.g. the duplicated ROLLER NO1/NO2 pair).
RAW_TO_STD = {
    "근무일자": "Date",
    "근무시간": "WorkTime",
    "POLYCOM 운전시간": "POLYCOM_ProcTime",
    "POLYCOM W/F FEED TOTAL": "POLYCOM_WF_FEED_TOTAL",
    "POLYCOM W/F FEED 크링커": "POLYCOM_WF_FEED_Clinker",
    "POLYCOM W/F FEED 석고": "POLYCOM_WF_FEED_Gypsum",
    "POLYCOM W/F FEED SLAG": "POLYCOM_WF_FEED_SLAG",
    "POLYCOM W/F FEED F/A": "POLYCOM_WF_FEED_FA",
    "POLYCOM ROLLER SPAC": "POLYCOM_ROLLER_SPAC",
    "POLYCOM ROLLER SKEW": "POLYCOM_ROLLER_SKEW",
}

# Positional column list for the 2CM/3CM/4CM operating sheets (index -> std name). 45 entries.
# This IS the dataframe's actual column names right after load_raw(), so downstream selection
# lists (CONTROL_COLS / MONITOR_COLS / TARGET_COLS) must match these names, not alias them.
STD_COLUMNS_BY_POSITION = [
    "Date", "WorkTime",
    "RP_proc_time",
    "feed_total", "feed_clinker", "feed_gypsum",
    "feed_slag", "feed_FA",
    "RP_spac", "RP_skew",
    "RP_roller_p1", "RP_roller_p2",              # pressure gauges (1st NO1/NO2 pair)
    "RP_roller_energy1", "RP_roller_energy2",    # position/energy sensors (2nd NO1/NO2 pair)
    "RP_roller_vib",
    "RP_BE_energy1", "RP_BE_energy2",
    "dosing_BE_energy",
    "RP_sep_rpm", "RP_sep_fan_damper",
    "RP_sep_BF_pressure", "RP_sep_BF_damper",
    "mill_energy",                             # raw MILL 166 B/E; ANN called this dosing_BE_energy
    "mill_feed_c", "mill_feed_cir",
    "MILL_CM_MAIN", "mill_in_temp", "mill_out_temp",   # raw MILL C/M MAIN; ANN called this mill_energy
    "mill_BE_energy",
    "mill_out_gas_temp", "mill_out_mater_temp",
    "mill_BF_pressure", "mill_BF_damper",
    "mill_sep_rpm", "mill_sep_fan_damper",
    "mill_sep_BF_pressure", "mill_sep_BF_fan_damper",
    "final_BE_1", "final_BE_2", "final_mater_temp",
    "grind_aid",
    "blaine", "residue",
    "ProdType", "Remark",
]

TARGET_COLS = ["blaine", "residue"]
QUALITY_MEASUREMENT_HOURS = [0, 4, 8, 12, 16, 20]

# blaine_prev / residue_prev: the preceding SCHEDULED measured value (not necessarily the most
# recent actual measurement). Computed by shifting(1) among scheduled measurement rows only,
# so it stays SPARSE (NaN everywhere except at measurement events). Passed as a KNOWN-FUTURE
# covariate for both target models (matches the reference notebook's variables_blaine /
# variables_residue, which both include Blaine_prev AND Residue_prev).
PREV_QUALITY_COLS = [f"{t}_prev" for t in TARGET_COLS]  # ["blaine_prev", "residue_prev"]

# --- Controllable (operator-settable) variables -> Chronos-2 known-future covariates ---
# The 1st-year ANN's CONTROL_COLS_9 (autoControl_b_v2.ipynb). Raw xlsx positions cross-checked
# against the ANN's declaration.
CONTROL_COLS = [
    "RP_roller_p1",             # POLYCOM_ROLLER_NO1 (roller press 1)
    "RP_roller_p2",             # POLYCOM_ROLLER_NO2 (roller press 2)
    "RP_sep_rpm",               # POLYCOM_SEPOL_SEPOL (separator rpm, 1st stage)
    "RP_sep_fan_damper",        # POLYCOM_SEPOL_FANDP (separator fan damper, 1st stage)
    "RP_sep_BF_damper",
    "mill_BF_damper",
    "mill_sep_rpm",             # MILL_SEPOL_SEPOL (separator rpm, 2nd stage)
    "mill_sep_fan_damper",      # MILL_SEPOL_FANDP (separator fan damper, 2nd stage)
    "grind_aid",                # Agent (grinding aid dosing)
]

# --- Monitor (observed, not directly set) variables -> past covariates ---
MONITOR_COLS = [
    "feed_total", "feed_clinker", "feed_gypsum", "feed_slag", "feed_FA",
    "RP_spac", "RP_skew",
    "RP_roller_energy1", "RP_roller_energy2", "RP_roller_vib",
    "RP_BE_energy1", "RP_BE_energy2",
    "RP_sep_BF_pressure",
    "mill_energy", "mill_feed_c", "mill_feed_cir",
    "MILL_CM_MAIN",           # raw MILL C/M MAIN, corresponding to ANN's mill_energy signal
    "mill_in_temp", "mill_out_temp", "mill_BE_energy",
    "mill_out_gas_temp", "mill_out_mater_temp",
    "mill_BF_pressure",
    "mill_sep_BF_pressure", "mill_sep_BF_fan_damper",
    "final_BE_1", "final_BE_2", "final_mater_temp",
    "dosing_BE_energy",
    "RP_proc_time",           # minutes actually operated this hour (0=idle, 60=full hour)
] + PREV_QUALITY_COLS

# Quality spec bands -- same thresholds used throughout the 1st-year project's Confusion-Matrix
# accuracy evaluation (autoControl_b_v2.ipynb, 강원대 성과발표자료.pdf).
SPEC_RANGES = {
    "blaine": (3700.0, 3900.0),
    "residue": (7.0, 9.0),
}

# --- Physical / equipment-limit cleaning rules (NOT statistical) ---
# Each threshold sits in a large gap between the flagged value(s) and the next plausible reading,
# so only unambiguous sensor-fault / sentinel values are removed. Ambiguous mid-range tails are
# left alone pending plant/engineer confirmation.
CLEANING_RULES = {
    "min_date": "2020-01-15",
    "damper_max_pct": 100.0,       # *_FANDP, *_DP columns: opening ratio must be <= 100%
    "damper_min_pct": 0.0,         # opening ratio can't be negative
    "bag_filter_press_max": 0.0,   # *_BF_Press columns: pressure must be <= 0
    "mill_feed_c_max": 10000.0,
    "blaine_min": 1000.0,
    "blaine_max": 10000.0,
    "residue_max": 20.0,
    "require_prodtype": "내수",
    "mill_out_gas_temp_max": 5000.0,    # real max otherwise ~1405; sentinel found at 100000008
    "mill_out_mater_temp_max": 5000.0,  # real max otherwise ~1285; sentinel found at 9100
    "roller_p2_max": 20000.0,           # real max otherwise ~5119; sentinel found at 83147
}

DAMPER_COLS = [
    "RP_sep_fan_damper", "RP_sep_BF_damper",
    "mill_BF_damper", "mill_sep_fan_damper", "mill_sep_BF_fan_damper",
]
BAG_FILTER_PRESS_COLS = ["RP_sep_BF_pressure", "mill_BF_pressure", "mill_sep_BF_pressure"]

# Statistical (quantile-based) outlier fence multiplier -- ONLY used by the
# experiments/iqr/ A/B (build_dataset.build(remove_iqr=True)); the canonical pipeline never
# runs statistical outlier removal (see the config docstring / EXPERIMENT_LOG). Standard Tukey
# 1.5x IQR, applied per item_id per column.
IQR_OUTLIER_K = 1.5
