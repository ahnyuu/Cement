"""Shared helpers for loading the processed dataset and splitting it chronologically."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

# Defaults to the canonical processed CSV (data/build_dataset.py main(): clean()'s physical rules
# only, NO statistical IQR removal since 2026-08-31 -- see 01_iqr_outlier_removal/). Override via
# env var CHRONOS_PROCESSED_CSV to point train/finetune_chronos2.py and eval/backtest.py at a
# different processed CSV instead -- e.g. the 01_iqr_outlier_removal / 02_physical_outlier_handling
# branch outputs, to compare preprocessing variants without touching the canonical file.
PROCESSED_PATH = Path(os.environ.get(
    "CHRONOS_PROCESSED_CSV",
    str(Path(__file__).resolve().parent / "processed" / "quality_timeseries.csv"),
))


def load_processed(target: str) -> pd.DataFrame:
    """Load the processed dataset and select feature + one target column, in long format."""
    if target not in cfg.TARGET_COLS:
        raise ValueError(f"target must be one of {cfg.TARGET_COLS}, got {target!r}")

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp"])
    keep = ["item_id", "timestamp"] + cfg.CONTROL_COLS + cfg.MONITOR_COLS + [target]
    df = df[keep].sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each item_id's series chronologically into train/val/test (no shuffling), matching the
    0.7/0.15/0.15 ratio used by the 1st-year ANN pipeline -- but split by TIME instead of randomly
    sampling rows, since these are autocorrelated process series and a random split would leak
    future information into training (the ANN's original train_test_split did this).
    """
    train_parts, val_parts, test_parts = [], [], []
    for item_id, g in df.groupby("item_id", sort=False):
        n = len(g)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train : n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val :])
    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, val, test
