"""Load the processed dataset and split it chronologically."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

# Canonical processed CSV built by data/build_dataset.py. Override via env var
# CHRONOS_PROCESSED_CSV to point train/eval at a different processed CSV (e.g. an experiment
# variant) without touching the canonical file.
PROCESSED_PATH = Path(os.environ.get(
    "CHRONOS_PROCESSED_CSV",
    str(Path(__file__).resolve().parent / "processed" / "quality_timeseries.csv"),
))


def load_processed(target: str, include_prev: bool = True) -> pd.DataFrame:
    """Load the processed dataset and select feature + one target column, in long format.

    include_prev=False drops blaine_prev/residue_prev entirely (not just from the known-covariate
    list -- from_data_frame would otherwise pick leftover columns up as PAST covariates). Used by
    experiments/prev/ to A/B the prev-quality covariate on vs off.
    """
    if target not in cfg.TARGET_COLS:
        raise ValueError(f"target must be one of {cfg.TARGET_COLS}, got {target!r}")

    monitor = cfg.MONITOR_COLS if include_prev else [c for c in cfg.MONITOR_COLS if c not in cfg.PREV_QUALITY_COLS]
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp"])
    keep = ["item_id", "timestamp"] + cfg.CONTROL_COLS + monitor + [target]
    return df[keep].sort_values(["item_id", "timestamp"]).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each item_id's series chronologically into train/val/test (no shuffling) at the
    0.7/0.15/0.15 ratio the 1st-year ANN used -- but by TIME, not random row sampling, since
    these are autocorrelated process series and a random split leaks future info into training.
    """
    train_parts, val_parts, test_parts = [], [], []
    for _item_id, g in df.groupby("item_id", sort=False):
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
