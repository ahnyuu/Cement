"""
Column mapping, control/monitor split, and cleaning rules for the Chronos-2 quality net.

STD_COLUMNS_BY_POSITION below is derived positionally from the raw POLYCOM/MILL xlsx headers
(originally "운전 & 품질 Data (2025)_20251125.xlsx", now "운전 & 품질 데이터_2026.xlsx" -- see
the HEADER_ROW note). CONTROL_COLS/MONITOR_COLS/TARGET_COLS were renamed 2026-08-12 to the
1st-year ANN's original RP_*/mill_*/blaine/residue short codes (per autoControl_b_v2.ipynb /
Blaine.ipynb / Residue.ipynb), at the user's request, so this pipeline's columns line up with the
ANN's. A handful of columns have no confirmed old<->new mapping (no raw->RP_* mapping script or
reference PDF is available on this machine, and the two datasets share no join key to verify
against) -- those are left with their POLYCOM_/MILL_ names rather than guessed, and are marked
"UNCONFIRMED"/"no 1st-year ANN equivalent" inline below.
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
    requires passing `import_allowlist=["chronos.chronos2.model"]` to
    `AutoPeftModel.from_pretrained` (a security allowlist check added by peft). That kwarg is
    NOT accepted when loading the base HF hub model or a full-finetune checkpoint, so we retry
    with it only if the plain load fails with the allowlist error, rather than guessing the
    checkpoint type upfront.
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


# Only needed by data/build_dataset.py, and only if you want to regenerate
# data/processed/quality_timeseries.csv from the raw xlsx export. The processed CSV itself
# travels with the rest of this folder (git/copy/zip), so on a machine that doesn't have the raw
# xlsx you normally don't need this at all -- training/eval only read the already-built processed
# CSV. Override via env var CHRONOS_SOURCE_XLSX if the raw file lives somewhere else.
#
# Source file: "운전 & 품질 데이터_2026.xlsx", kept one level above the cement_code/ project
# folder (i.e. next to it, not inside it). Data range 2017/2020~2025-11-17; header layout per the
# HEADER_ROW note below (2-row header, no Worker column). Operating sheets: 2CM/3CM/4CM.
SOURCE_XLSX = os.environ.get(
    "CHRONOS_SOURCE_XLSX",
    str(Path(__file__).resolve().parents[2] / "운전 & 품질 데이터_2026.xlsx"),
)
OPERATING_SHEETS = ["2CM", "3CM", "4CM"]  # trailing-space-free sheets = hourly operating data

# NOTE: header layout changed between xlsx exports. The original "운전 & 품질 Data (2025)_
# 20251125.xlsx" (1st-year source) had a merged 7-row header (data from row 8) and included a
# "근무자"(Worker) column. The current "운전 & 품질 데이터_2026.xlsx" (2nd-year source, layout
# verified 2026-08-12 by comparing loaded rows/values against the raw sheet) has a single header
# row and, critically, does NOT include the Worker column -- every column from position 2 onward
# is shifted by one vs. the old layout. HEADER_ROW / DATA_START_ROW / STD_COLUMNS_BY_POSITION
# below are tuned for the CURRENT (no-Worker) file; if a future export brings Worker back,
# re-insert "Worker" after "Date" and bump these back to 7/8.
#
# HEADER_ROW/DATA_START_ROW below describe the common case (2CM/4CM sheets). The 3CM sheet has no
# header row at all -- its data starts at row 1 -- so data/build_dataset.py's _load_sheet()
# detects the real data-start row per sheet (by scanning for the "근무일자" header cell) instead
# of trusting DATA_START_ROW blindly for every sheet.
HEADER_ROW = 2  # 1-indexed row in each sheet containing the column headers (when present)
DATA_START_ROW = 3  # typical data-start row; see per-sheet detection note above

# Raw (line-broken) xlsx header -> standardized column name.
# Order matches the physical column order in the 2CM/3CM/4CM operating sheets.
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
    # NOTE: "ROLLER NO1"/"NO2" appear twice in the raw header (pressure gauge vs. position/energy
    # sensor, i.e. NO1/NO2 vs NO11/NO21). openpyxl collapses both to the same text, so we
    # disambiguate positionally in build_dataset.py using column index, not this dict.
}

# Positional column list for the 2CM/3CM/4CM operating sheets (index -> std name), built from
# RAW_TO_STD plus the two duplicated ROLLER NO1/NO2 occurrences and the remaining tail columns
# that also collide under RAW_TO_STD (Impal, quality, etc). This is the authoritative list used
# by build_dataset.py; RAW_TO_STD above documents the unambiguous ones for readability.
# 45 entries, verified 2026-08-12 against row 2 header text for the 2CM/4CM sheets (3CM has no
# header row -- see the per-sheet detection note above; its columns were cross-checked
# positionally against 2CM/4CM instead). No "Worker" column in this export -- see
# HEADER_ROW note above. Renamed
# 2026-08-12 to the old ANN names wherever CONTROL_COLS/MONITOR_COLS above uses them, so this
# raw-load-time list and the downstream selection lists stay in sync (this IS the dataframe's
# actual column names right after load_raw(), so it must match, not just alias, those lists).
STD_COLUMNS_BY_POSITION = [
    "Date", "WorkTime",
    "RP_proc_time",
    "feed_total", "feed_clinker", "feed_gypsum",
    "feed_slag", "feed_FA",
    "RP_spac", "RP_skew",
    "RP_roller_p1", "RP_roller_p2",     # pressure gauges (1st NO1/NO2 pair)
    "RP_roller_energy1", "RP_roller_energy2",   # position/energy sensors (2nd NO1/NO2 pair)
    "RP_roller_vib",
    "RP_BE_energy1", "RP_BE_energy2",
    "dosing_BE_energy",
    "RP_sep_rpm", "RP_sep_fan_damper",
    "RP_sep_BF_pressure", "RP_sep_BF_damper",
    "mill_energy",
    "mill_feed_c", "mill_feed_cir",
    "MILL_CM_MAIN", "mill_in_temp", "mill_out_temp",      # MILL_CM_MAIN: no 1st-year ANN equivalent
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

# blaine_prev / residue_prev: the most recently MEASURED value of each target (not the current
# row's value), added 2026-08-12 per the reference notebook's "이전품질 적용 후 Chronos" section
# (Downloads/chronos-2-cement_share.ipynb, fill_previous_quality_measurements()). Computed in
# build_dataset.py by shifting(1) among rows where the target is actually non-null (real lab
# readings happen ~every 4h), so this stays SPARSE -- NaN everywhere except at measurement events
# themselves -- deliberately not forward-filled (see build_dataset.py note + chat discussion:
# forward-filling would repeat the same value across every hourly context row, over-exposing the
# model to a "just copy this" pattern during training and risking collapse toward the naive
# persistence baseline). Included as a KNOWN-FUTURE covariate
# (not just past) for BOTH target models, matching the reference notebook's variables_blaine /
# variables_residue, which both include Blaine_prev AND Residue_prev -- see finetune_chronos2.py /
# eval/backtest.py / inference/quality_predictor.py, all updated to pass these at prediction time.
PREV_QUALITY_COLS = [f"{t}_prev" for t in TARGET_COLS]  # ["blaine_prev", "residue_prev"]

# --- Controllable (operator-settable) variables -> Chronos-2 known-future covariates ---
# Renamed 2026-08-12 to the 1st-year ANN's original CONTROL_COLS_9 names (autoControl_b_v2.ipynb)
# at the user's request. The raw xlsx positions were cross-checked against the ANN's
# CONTROL_COLS_9 declaration, including the two bag-filter damper columns.
CONTROL_COLS = [
    "RP_roller_p1",             # = POLYCOM_ROLLER_NO1 (roller press 1) [confirmed]
    "RP_roller_p2",             # = POLYCOM_ROLLER_NO2 (roller press 2) [confirmed]
    "RP_sep_rpm",               # = POLYCOM_SEPOL_SEPOL (separator rpm, 1st stage) [confirmed]
    "RP_sep_fan_damper",        # = POLYCOM_SEPOL_FANDP (separator fan damper, 1st stage) [confirmed]
    "RP_sep_BF_damper",
    "mill_BF_damper",
    "mill_sep_rpm",             # = MILL_SEPOL_SEPOL (separator rpm, 2nd stage) [confirmed]
    "mill_sep_fan_damper",      # = MILL_SEPOL_FANDP (separator fan damper, 2nd stage) [confirmed]
    "grind_aid",                # = Agent (grinding aid dosing) [confirmed]
]

# --- Monitor (observed, not directly set) variables -> past covariates ---
# Renamed alongside CONTROL_COLS using the ANN's MONITOR_COLS_28 names. Names are retained only
# where there's no 1st-year ANN equivalent at all (MILL_CM_MAIN; RP_proc_time, which is new to
# this Chronos-2 pipeline -- RP_proc_time keeps the "RP_" style purely for naming consistency
# with build_dataset.py, not a real ANN column).
MONITOR_COLS = [
    "feed_total", "feed_clinker", "feed_gypsum", "feed_slag", "feed_FA",
    "RP_spac", "RP_skew",
    "RP_roller_energy1", "RP_roller_energy2", "RP_roller_vib",
    "RP_BE_energy1", "RP_BE_energy2",
    "RP_sep_BF_pressure",
    "mill_energy", "mill_feed_c", "mill_feed_cir",
    "MILL_CM_MAIN",           # no 1st-year ANN equivalent
    "mill_in_temp", "mill_out_temp", "mill_BE_energy",
    "mill_out_gas_temp", "mill_out_mater_temp",
    "mill_BF_pressure",
    "mill_sep_BF_pressure", "mill_sep_BF_fan_damper",
    "final_BE_1", "final_BE_2", "final_mater_temp",
    "dosing_BE_energy",
    "RP_proc_time",           # minutes actually operated this hour (0=idle, 60=full hour, partial=transition)
] + PREV_QUALITY_COLS  # blaine_prev / residue_prev -- see PREV_QUALITY_COLS definition above

# Quality spec bands, same thresholds used throughout the existing project's Confusion-Matrix
# accuracy evaluation (autoControl_b_v2.ipynb, 강원대 성과발표자료.pdf).
SPEC_RANGES = {
    "blaine": (3700.0, 3900.0),
    "residue": (7.0, 9.0),
}

# --- Cleaning rules, reproduced from the 1st-year 워크샵 PDF ("데이터 리뷰" 결측치 처리 과정) ---
# All physical/equipment-limit thresholds, NOT statistical (contrast with remove_iqr_outliers(),
# which derives its fences from the data's own quantiles and was found to hurt model performance --
# see notebooks/01_iqr_outlier_removal/). Every threshold below was chosen where the raw data shows
# a large gap between the flagged value(s) and the next-highest/lowest plausible-looking reading
# (checked 2026-08-31), so only unambiguous sensor-fault/sentinel values are removed -- ambiguous
# mid-range tails (e.g. mill_out_gas_temp/mill_out_mater_temp's 1100-1400 cluster,
# mill_out_mater_temp's negative cluster, blaine's own min/max) are deliberately left alone pending
# plant/engineer confirmation rather than guessed at.
CLEANING_RULES = {
    "min_date": "2020-01-15",  # NOTE: our source xlsx only starts 2025-01-01, see caveat in README
    "damper_max_pct": 100.0,       # *_FANDP, *_DP columns: opening ratio must be <= 100%
    "damper_min_pct": 0.0,         # opening ratio can't be negative (observed down to -251%)
    "bag_filter_press_max": 0.0,   # *_BF_Press columns: pressure must be <= 0
    "mill_feed_c_max": 10000.0,
    "blaine_min": 1000.0,
    "blaine_max": 10000.0,
    "residue_max": 20.0,
    "require_prodtype": "내수",
    # sensor-fault sentinels: each is separated from the next-highest real-looking reading by a
    # large gap (100000008 vs 1405; 9100 vs 1285; 83147 vs 5119 -- and RP_roller_p2's own sibling
    # RP_roller_p1 never exceeds 8593), so the cutoff sits safely in that gap either way.
    "mill_out_gas_temp_max": 5000.0,    # real max otherwise 1405; sentinel found at 100000008
    "mill_out_mater_temp_max": 5000.0,  # real max otherwise 1285; sentinel found at 9100
    "roller_p2_max": 20000.0,           # real max otherwise 5119; sentinel found at 83147
}

IQR_OUTLIER_K = 1.5  # standard Tukey fence multiplier, applied per item_id per column

# Experiment 02 (physical-outlier handling A/B, 2026-09-02 -- see 02_physical_outlier_handling/).
# Margin, in percentage points, by which a damper opening-ratio reading may overshoot its physical
# bound (0 or 100%) and still be treated as calibration drift -> clipped to the bound, rather than
# a sensor fault -> NaN. ONLY consulted when clean(damper_oob="clip"); the canonical pipeline
# (damper_oob="nan", the default) NaNs every out-of-range value and ignores this.
DAMPER_CLIP_MARGIN_PCT = 20.0

DAMPER_COLS = [
    "RP_sep_fan_damper", "RP_sep_BF_damper",
    "mill_BF_damper", "mill_sep_fan_damper", "mill_sep_BF_fan_damper",
]
BAG_FILTER_PRESS_COLS = ["RP_sep_BF_pressure", "mill_BF_pressure", "mill_sep_BF_pressure"]
