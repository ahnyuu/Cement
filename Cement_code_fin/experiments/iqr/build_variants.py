"""
IQR outlier-removal A/B -- build the two processed CSVs.

  data/processed/without_iqr/quality_timeseries.csv   = build(remove_iqr=False)  (== canonical)
  data/processed/with_iqr/quality_timeseries.csv      = build(remove_iqr=True)

Only difference between the two: per-item_id/per-column Tukey 1.5x IQR fence -> NaN on every
raw feature column except RP_proc_time, plus blaine/residue (build_dataset.iqr_cols()).
Everything else (physical clean() rules, sparse prev covariates, hourly reindex) is identical.

    .venv/Scripts/python.exe experiments/iqr/build_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config as cfg
from data.build_dataset import build

OUT = {
    "without_iqr": ROOT / "data" / "processed" / "without_iqr" / "quality_timeseries.csv",
    "with_iqr": ROOT / "data" / "processed" / "with_iqr" / "quality_timeseries.csv",
}


def _summarize(df, name):
    print(f"\n=== {name}: {len(df)} rows ===")
    print("target null rates:")
    print(df[cfg.TARGET_COLS].isna().mean())
    print("feature null rate (mean over CONTROL+MONITOR):",
          round(df[cfg.CONTROL_COLS + cfg.MONITOR_COLS].isna().mean().mean(), 4))


def main() -> None:
    print(f"SOURCE_XLSX -> {cfg.SOURCE_XLSX}")
    for name, remove_iqr in [("without_iqr", False), ("with_iqr", True)]:
        print(f"\n---------- building {name} (remove_iqr={remove_iqr}) ----------")
        df = build(remove_iqr=remove_iqr)
        OUT[name].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT[name], index=False)
        print(f"saved -> {OUT[name]}")
        _summarize(df, name)

    # sanity: without_iqr must be byte-identical to the canonical CSV
    canon = (ROOT / "data" / "processed" / "quality_timeseries.csv").read_bytes()
    wo = OUT["without_iqr"].read_bytes()
    print(f"\nwithout_iqr == canonical quality_timeseries.csv : {canon == wo}")


if __name__ == "__main__":
    main()
