"""
IQR outlier-removal A/B -- build NEW review_v2 CSVs, leaving historical files intact.

  data/processed/without_iqr/quality_timeseries.csv   = build(remove_iqr=False)  (== canonical)
  data/processed/with_iqr/quality_timeseries.csv      = build(remove_iqr=True)

Fit per-item_id/per-column Tukey 1.5x IQR fences on training hourly rows ONLY.
Mask raw features except RP_proc_time using those fixed fences. Quality labels are masked
only in training; validation/test quality observations remain available for honest scoring.
Everything else (physical clean() rules, sparse prev covariates, hourly reindex) is identical.

    .venv/Scripts/python.exe experiments/iqr/build_variants.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config as cfg
from data.build_dataset import build

OUT = {
    "without_iqr": ROOT / "data" / "processed" / "review_v2" / "without_iqr" / "quality_timeseries.csv",
    "with_iqr": ROOT / "data" / "processed" / "review_v2" / "with_iqr" / "quality_timeseries.csv",
}


def _summarize(df, name):
    print(f"\n=== {name}: {len(df)} rows ===")
    print("target null rates:")
    print(df[cfg.TARGET_COLS].isna().mean())
    print("feature null rate (mean over CONTROL+MONITOR):",
          round(df[cfg.CONTROL_COLS + cfg.MONITOR_COLS].isna().mean().mean(), 4))


def main() -> None:
    print(f"SOURCE_XLSX -> {cfg.SOURCE_XLSX}")
    print("review_v2: train-only IQR fences; validation/test target labels preserved.")
    for name, remove_iqr in [("without_iqr", False), ("with_iqr", True)]:
        print(f"\n---------- building {name} (remove_iqr={remove_iqr}) ----------")
        df = build(remove_iqr=remove_iqr)
        OUT[name].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT[name], index=False)
        if remove_iqr:
            OUT[name].with_suffix(".fences.json").write_text(
                json.dumps(df.attrs["iqr_fences"], indent=2), encoding="utf-8"
            )
        print(f"saved -> {OUT[name]}")
        _summarize(df, name)

    # A byte comparison is informative; CSV formatting can differ between pandas versions.
    canon = (ROOT / "data" / "processed" / "quality_timeseries.csv").read_bytes()
    wo = OUT["without_iqr"].read_bytes()
    print(f"\nwithout_iqr == canonical quality_timeseries.csv : {canon == wo}")


if __name__ == "__main__":
    main()
