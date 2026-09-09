"""
Build the long-format, timestamped, per-station time series dataset for the Chronos-2 quality net
from the raw xlsx export (config.SOURCE_XLSX = "운전&품질데이터_실측치.xlsx").

Single canonical pipeline (no experiment branches):
    load raw 2CM/3CM/4CM sheets
      -> build hourly timestamp
      -> clean()          physical/equipment-limit rules + ProdType(내수) filter
      -> numeric coercion
      -> add sparse blaine_prev / residue_prev covariates
      -> reindex each station onto a complete hourly grid (missing hours -> NaN rows)
      -> save data/processed/quality_timeseries.csv

Output columns: item_id, timestamp, <CONTROL_COLS>, <MONITOR_COLS>, blaine, residue
(MONITOR_COLS already includes blaine_prev / residue_prev.)

Run:  python data/build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

OUTPUT_PATH = Path(__file__).resolve().parent / "processed" / "quality_timeseries.csv"


def _find_data_start_row(ws, max_scan_rows: int = 5) -> int:
    """1-indexed row where actual data begins in this sheet.

    Normally cfg.DATA_START_ROW (header "근무일자" at cfg.HEADER_ROW, e.g. 2CM/4CM). The 3CM sheet
    has no header row -- data starts at row 1 -- so detect per-sheet by scanning for "근무일자".
    """
    for row_num, row in enumerate(
        ws.iter_rows(min_row=1, max_row=max_scan_rows, values_only=True), start=1
    ):
        if row and row[0] == "근무일자":
            return row_num + 1
    return 1


def _load_sheet(ws) -> pd.DataFrame:
    data_start_row = _find_data_start_row(ws)
    rows = list(ws.iter_rows(min_row=data_start_row, values_only=True))
    ncols = len(cfg.STD_COLUMNS_BY_POSITION)
    data = [row[:ncols] for row in rows]
    return pd.DataFrame(data, columns=cfg.STD_COLUMNS_BY_POSITION)


def load_raw() -> pd.DataFrame:
    wb = openpyxl.load_workbook(cfg.SOURCE_XLSX, read_only=True, data_only=True)
    frames = []
    for sheet_name in cfg.OPERATING_SHEETS:
        df = _load_sheet(wb[sheet_name])
        df["item_id"] = sheet_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    date = pd.to_datetime(df["Date"], errors="coerce")
    hour = pd.to_numeric(df["WorkTime"], errors="coerce")
    return date + pd.to_timedelta(hour, unit="h")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixed physical/equipment-limit rules from cfg.CLEANING_RULES.

    - damper opening-ratio readings outside [0, 100]%       -> NaN
    - bag-filter pressure readings above 0                  -> NaN
    - mill_feed_c / blaine / residue past their bounds      -> NaN
    - sensor-fault sentinels (gas/mater temp, roller_p2)    -> NaN
    - rows whose ProdType (per-station ffill) != 내수        -> dropped
    """
    df = df.copy()

    damper_hi = cfg.CLEANING_RULES["damper_max_pct"]
    damper_lo = cfg.CLEANING_RULES["damper_min_pct"]
    for col in cfg.DAMPER_COLS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        df.loc[(numeric > damper_hi) | (numeric < damper_lo), col] = np.nan
    for col in cfg.BAG_FILTER_PRESS_COLS:
        df.loc[pd.to_numeric(df[col], errors="coerce") > cfg.CLEANING_RULES["bag_filter_press_max"], col] = np.nan

    mill_coarse = pd.to_numeric(df["mill_feed_c"], errors="coerce")
    df.loc[mill_coarse > cfg.CLEANING_RULES["mill_feed_c_max"], "mill_feed_c"] = np.nan

    blaine = pd.to_numeric(df["blaine"], errors="coerce")
    df.loc[
        (blaine < cfg.CLEANING_RULES["blaine_min"]) | (blaine > cfg.CLEANING_RULES["blaine_max"]),
        "blaine",
    ] = np.nan

    residue = pd.to_numeric(df["residue"], errors="coerce")
    df.loc[residue > cfg.CLEANING_RULES["residue_max"], "residue"] = np.nan

    gas_temp = pd.to_numeric(df["mill_out_gas_temp"], errors="coerce")
    df.loc[gas_temp > cfg.CLEANING_RULES["mill_out_gas_temp_max"], "mill_out_gas_temp"] = np.nan

    mater_temp = pd.to_numeric(df["mill_out_mater_temp"], errors="coerce")
    df.loc[mater_temp > cfg.CLEANING_RULES["mill_out_mater_temp_max"], "mill_out_mater_temp"] = np.nan

    roller_p2 = pd.to_numeric(df["RP_roller_p2"], errors="coerce")
    df.loc[roller_p2 > cfg.CLEANING_RULES["roller_p2_max"], "RP_roller_p2"] = np.nan

    df = df.sort_values(["item_id", "timestamp"])
    blank_prodtype = df["ProdType"].isna() | (df["ProdType"].astype(str).str.strip() == "")
    df.loc[blank_prodtype, "ProdType"] = np.nan
    df["ProdType"] = df.groupby("item_id")["ProdType"].ffill()

    # ProdType is only recorded at the 4-hour measurement rows; without the ffill above this filter
    # would drop most operating hours (and would delete every row for years where ProdType is
    # never filled). NaN still counts as out-of-scope after the ffill.
    out_of_scope = df["ProdType"] != cfg.CLEANING_RULES["require_prodtype"]
    return df[~out_of_scope]


def raw_feature_cols() -> list[str]:
    """CONTROL_COLS + the MONITOR_COLS that come straight from the raw xlsx (i.e. excluding the
    *_prev columns, which are derived later)."""
    return cfg.CONTROL_COLS + [c for c in cfg.MONITOR_COLS if c not in cfg.PREV_QUALITY_COLS]


def remove_iqr_outliers(df: pd.DataFrame, cols: list[str], k: float = cfg.IQR_OUTLIER_K):
    """Per item_id, per column: values outside [Q1 - k*IQR, Q3 + k*IQR] -> NaN.

    Statistical complement to clean()'s fixed physical thresholds. NOT part of the canonical
    pipeline -- only experiments/iqr/ calls this (via build(remove_iqr=True)), to A/B whether
    statistical outlier removal helps on the measured-actual data. Columns with IQR == 0
    (near-constant series) are left untouched. Returns (df, {col: n_values_removed}).
    """
    df = df.copy()
    grouped = df.groupby("item_id")
    removed = {}
    for col in cols:
        q1 = grouped[col].transform(lambda s: s.quantile(0.25))
        q3 = grouped[col].transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        outlier = (((df[col] < lo) | (df[col] > hi)) & (iqr > 0)).fillna(False)
        removed[col] = int(outlier.sum())
        df.loc[outlier, col] = np.nan
    return df, removed


def iqr_cols() -> list[str]:
    """Columns the experiments/iqr/ A/B runs remove_iqr_outliers() on: every raw feature column
    except RP_proc_time (idle-minutes counter -- its low tail is real, not an outlier), plus the
    two quality targets. Matches the old folder's 01_iqr_outlier_removal/build_with_iqr.py."""
    return [c for c in raw_feature_cols() if c != "RP_proc_time"] + cfg.TARGET_COLS


def add_prev_quality_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-quality covariates, present ONLY at scheduled 4-hour measurement rows.

    The original hourly blaine/residue columns are never altered. For each station and target,
    values observed at 00/04/08/12/16/20 are measurement events; *_prev at each event gets the
    preceding event's value. All intervening *_prev rows stay NaN (sparse).
    """
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    scheduled = df["timestamp"].dt.hour.isin(cfg.QUALITY_MEASUREMENT_HOURS)
    for target in cfg.TARGET_COLS:
        prev_col = f"{target}_prev"
        df[prev_col] = np.nan
        measured = scheduled & df[target].notna()
        df.loc[measured, prev_col] = df.loc[measured].groupby("item_id")[target].shift(1)
    return df


def build(*, remove_iqr: bool = False, prev_density: str = "sparse") -> pd.DataFrame:
    """Canonical build. Both knobs default to the canonical behavior -- the canonical CSV gets
    no statistical outlier removal and keeps the prev covariates sparse.

    remove_iqr=True  -- experiments/iqr/ A/B; IQR step slots in right after clean()/numeric
                        coercion and before the prev covariates.
    prev_density     -- "sparse" (default): blaine_prev/residue_prev present only at scheduled
                        measurement rows. "ffill": after the hourly reindex, forward-fill each
                        *_prev per item_id so every row carries the most recent lab reading as an
                        explicit anchor. Leak-safe at prediction_length<=4 (a scheduled row's
                        *_prev is already the measurement from >=4h earlier, so forward-filling
                        only ever exposes readings available >=4h before any row it lands on).
                        experiments/prev_density/ A/B.
    """
    if prev_density not in ("sparse", "ffill"):
        raise ValueError(f"prev_density must be 'sparse' or 'ffill', got {prev_density!r}")
    df = load_raw()
    df["timestamp"] = build_timestamp(df)
    df = df.dropna(subset=["timestamp"])

    df = clean(df)

    feature_cols = raw_feature_cols()
    for col in feature_cols + cfg.TARGET_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if remove_iqr:
        df, removed = remove_iqr_outliers(df, iqr_cols())
        top = sorted(removed.items(), key=lambda kv: -kv[1])[:10]
        print(f"IQR removal: {sum(removed.values())} values -> NaN. Top: {top}")

    df = add_prev_quality_covariates(df)

    keep_cols = ["item_id", "timestamp"] + feature_cols + cfg.PREV_QUALITY_COLS + cfg.TARGET_COLS
    df = df[keep_cols]
    df = df.sort_values(["item_id", "timestamp"]).drop_duplicates(subset=["item_id", "timestamp"])
    df = df.reset_index(drop=True)

    # Real plant data has gaps (downtime, missing logs) -> irregular timestamps. Chronos-2 needs a
    # regular frequency, so reindex each station onto a complete hourly grid; missing hours become
    # NaN rows (the model handles NaN context/covariates natively).
    reindexed = []
    for item_id, g in df.groupby("item_id", sort=False):
        g = g.set_index("timestamp").drop(columns=["item_id"])
        full_index = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_index)
        g["item_id"] = item_id
        g.index.name = "timestamp"
        reindexed.append(g.reset_index())
    df = pd.concat(reindexed, ignore_index=True)

    if prev_density == "ffill":
        df[cfg.PREV_QUALITY_COLS] = df.groupby("item_id", sort=False)[cfg.PREV_QUALITY_COLS].ffill()

    ordered = cfg.CONTROL_COLS + cfg.MONITOR_COLS  # MONITOR_COLS already ends with the *_prev cols
    return df[["item_id", "timestamp"] + ordered + cfg.TARGET_COLS]


def main() -> None:
    print(f"SOURCE_XLSX -> {cfg.SOURCE_XLSX}")
    df = build()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows -> {OUTPUT_PATH}")
    print(df["item_id"].value_counts())
    print("\nTimestamp range per item_id:")
    print(df.groupby("item_id")["timestamp"].agg(["min", "max", "count"]))
    print("\nTarget null rates:")
    print(df[cfg.TARGET_COLS].isna().mean())
    print("\nprev-quality null rates (sparse):")
    print(df[cfg.PREV_QUALITY_COLS].isna().mean())


if __name__ == "__main__":
    main()
