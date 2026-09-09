"""
prev-density A/B -- build the two processed CSVs.

  data/processed/prev_sparse/quality_timeseries.csv  = build(prev_density="sparse")  (== canonical)
  data/processed/prev_ffill/quality_timeseries.csv   = build(prev_density="ffill")

Only difference: blaine_prev / residue_prev are forward-filled per station after the hourly
reindex in the ffill arm (every row carries the most recent lab reading), vs present only at
scheduled measurement rows in the sparse arm. The blaine/residue TARGET columns stay sparse in
both -- only the two *_prev covariate columns change.

Includes a leak audit (mirrors the old folder's smoketest_prev_density.py):
  - ffill non-null superset of sparse non-null, sparse values preserved
  - every ffill'd *_prev value is >= 4h older than the row it lands on (leak-safe at predlen<=4)

    .venv/Scripts/python.exe experiments/prev_density/build_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config as cfg
from data.build_dataset import build

OUT = {
    "sparse": ROOT / "data" / "processed" / "prev_sparse" / "quality_timeseries.csv",
    "ffill": ROOT / "data" / "processed" / "prev_ffill" / "quality_timeseries.csv",
}


def leak_audit(sparse: pd.DataFrame, ffill: pd.DataFrame) -> None:
    print("\n=== leak / consistency audit ===")
    # non-prev columns identical
    non_prev = [c for c in sparse.columns if c not in cfg.PREV_QUALITY_COLS]
    same = sparse[non_prev].equals(ffill[non_prev])
    print(f"[D] all non-_prev columns identical between arms: {same}")

    for col in cfg.PREV_QUALITY_COLS:
        s, f = sparse[col], ffill[col]
        # B: sparse non-null values preserved in ffill, ffill is exactly sparse ffilled per item
        expected = sparse.groupby("item_id", sort=False)[col].ffill()
        b_ok = f.equals(expected)
        preserved = ((s.notna()) & (f == s)).sum() == s.notna().sum()
        print(f"[B] {col}: ffill == sparse.groupby(item).ffill(): {b_ok} | sparse values preserved: {preserved}")
        print(f"    non-null rate  sparse={s.notna().mean():.4f}  ffill={f.notna().mean():.4f}")

        # A: every ffill'd value is >= 4h older than the row it lands on. The value at a scheduled
        # row t_k is target(t_{k-1}) (shift(1) among measurement events), so its SOURCE is the
        # measurement event before t_k -- not t_k itself. Trace each row back to that source ts
        # via merge_asof on the measurement-event timeline.
        target = col.removesuffix("_prev")
        ev = ffill.loc[s.notna(), ["item_id", "timestamp"]].copy()          # rows sparse has a value = scheduled meas events
        ev["source_ts"] = ev.groupby("item_id", sort=False)["timestamp"].shift(1)  # _prev here came from the prior event
        rows = ffill.loc[f.notna(), ["item_id", "timestamp"]].sort_values("timestamp")
        merged = pd.merge_asof(rows, ev.sort_values("timestamp"), on="timestamp", by="item_id", direction="backward")
        lag_h = (merged["timestamp"] - merged["source_ts"]).dt.total_seconds() / 3600
        lag_h = lag_h.dropna()
        print(f"[A] {col}: ffill value age vs its row -- min={lag_h.min():.1f}h median={lag_h.median():.1f}h "
              f"| >=4h everywhere: {bool((lag_h >= 4 - 1e-6).all())}  ({merged['source_ts'].isna().sum()} rows pre-first-source)")


def main() -> None:
    print(f"SOURCE_XLSX -> {cfg.SOURCE_XLSX}")
    dfs = {}
    for name, dens in [("sparse", "sparse"), ("ffill", "ffill")]:
        print(f"\n---------- building prev_{name} (prev_density={dens!r}) ----------")
        df = build(prev_density=dens)
        OUT[name].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT[name], index=False)
        dfs[name] = df
        print(f"saved -> {OUT[name]}  ({len(df)} rows)")

    canon = (ROOT / "data" / "processed" / "quality_timeseries.csv").read_bytes()
    print(f"\nprev_sparse == canonical quality_timeseries.csv : {canon == OUT['sparse'].read_bytes()}")
    leak_audit(dfs["sparse"], dfs["ffill"])


if __name__ == "__main__":
    main()
