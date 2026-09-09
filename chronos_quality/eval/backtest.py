"""
Walk-forward backtest for the Chronos-2 quality net, evaluated the same way as the 1st-year ANN
so results are directly comparable:
  - MAE / RMSE / R2 in real units
  - Spec-in/out Confusion Matrix accuracy (blaine 3700-3900, residue 7-9), matching
    autoControl_b_v2.ipynb's categorize_blaine() and the 강원대 성과발표자료.pdf methodology.

Reference baselines already measured on this process (for comparison, not reproduced here):
  - 1st-year ANN (autoControl_b_v2 quality net): Blaine CF accuracy ~90.7-95.1%, Residue ~94.5-95.4%
  - 강원대 XGBoost/LightGBM benchmark: Blaine R2=0.763 (test), MAE=42.2; 44um Residue R2=0.926, MAE=0.436

Unlike the ANN's evaluation (random row split -> leaks future info), this backtest uses a
rolling-origin, 1-step-ahead scheme: at each evaluated timestamp t in the test window, the model
only sees history strictly before t (capped to --context-length) plus the known control settings
at t, and forecasts the quality reading at t.

Usage:
    python backtest.py --target blaine --checkpoint ../checkpoints/blaine/final --stride 4 --tag lora
    python backtest.py --target blaine --checkpoint amazon/chronos-2 --tag zeroshot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data.dataset_utils import chronological_split, load_processed



def make_eval_tasks(
    df: pd.DataFrame,
    item_id: str,
    test_start_idx: int,
    stride: int,
    context_length: int,
    fresh_reading_only: bool = True,
):
    """
    Build one rolling-origin forecast task per evaluated row: (context_df, future_df, actual_value).
    Context is capped to the last `context_length` rows strictly before the forecast timestamp.

    IMPORTANT: blaine/residue are only measured ~every 4h and forward-filled hourly in between
    (see 1st-year 워크샵 PDF, then named BLAINE/Remain: "BLAINE, Remain은 4시간 기준으로 채움").
    A naive "repeat the last value" baseline gets R2=1.0 / MAE=0 on most hourly rows because they simply repeat the prior
    reading -- this also affects the original ANN's evaluation, since its row-level dataset has
    the same value duplicated ~4x per reading and was split randomly, leaking near-duplicate rows
    across train/test. To measure genuine forecast skill, `fresh_reading_only=True` (default)
    restricts evaluation to rows where the target actually changes from the previous timestep,
    i.e. a new lab reading just came in -- the only points where "predict the target" is a
    non-trivial task.
    """
    g = df[df["item_id"] == item_id].reset_index(drop=True)
    target = g.columns[-1]

    if fresh_reading_only:
        is_fresh = g[target].ne(g[target].shift(1)) & g[target].notna()
        candidate_idx = [i for i in range(test_start_idx, len(g)) if is_fresh.iloc[i]]
    else:
        candidate_idx = list(range(test_start_idx, len(g)))

    tasks = []
    for i in candidate_idx[::stride]:
        ctx_start = max(0, i - context_length)
        context_df = g.iloc[ctx_start:i]
        if context_df[target].dropna().empty:
            continue
        future_row = g.iloc[[i]]
        if pd.isna(future_row[target].iloc[0]):
            continue
        future_df = future_row[
            ["item_id", "timestamp"] + cfg.CONTROL_COLS + cfg.PREV_QUALITY_COLS
        ]
        tasks.append((context_df, future_df, float(future_row[target].iloc[0]), future_row["timestamp"].iloc[0]))
    return tasks


def categorize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((values >= lo) & (values <= hi), "IN_SPEC", "OUT_OF_SPEC")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=cfg.TARGET_COLS, required=True)
    parser.add_argument("--checkpoint", default="amazon/chronos-2", help="local fine-tuned path or HF model id")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=4, help="evaluate every Nth eligible row")
    parser.add_argument("--device-map", default=cfg.default_device_map())
    parser.add_argument(
        "--all-hours",
        action="store_true",
        help="evaluate every hourly row instead of only fresh-reading transitions (inflates "
        "accuracy trivially -- see make_eval_tasks docstring; off by default)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="optional suffix for the output csv, e.g. --tag zeroshot -> backtest_{target}_zeroshot.csv "
        "(default: none -> backtest_{target}.csv). Without this, runs against different checkpoints "
        "for the same target silently overwrite each other's result file.",
    )
    args = parser.parse_args()

    df = load_processed(args.target)
    _train_df, _val_df, test_df = chronological_split(df)

    pipeline = cfg.load_chronos_pipeline(args.checkpoint, device_map=args.device_map)

    lo, hi = cfg.SPEC_RANGES[args.target]

    records = []
    for item_id in df["item_id"].unique():
        test_start_idx = len(df[df["item_id"] == item_id]) - len(test_df[test_df["item_id"] == item_id])
        tasks = make_eval_tasks(
            df, item_id, test_start_idx, args.stride, args.context_length,
            fresh_reading_only=not args.all_hours,
        )
        print(f"{item_id}: {len(tasks)} eval points")
        for context_df, future_df, actual, ts in tasks:
            pred = pipeline.predict_df(
                df=context_df,
                future_df=future_df,
                id_column="item_id",
                timestamp_column="timestamp",
                target=args.target,
                prediction_length=1,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            row = pred.iloc[0]
            naive_pred = context_df[args.target].dropna().iloc[-1]
            records.append(
                {
                    "item_id": item_id,
                    "timestamp": ts,
                    "actual": actual,
                    "naive_pred": naive_pred,
                    "pred": float(row["predictions"]),
                    "pred_q10": float(row["0.1"]),
                    "pred_q90": float(row["0.9"]),
                }
            )

    result = pd.DataFrame(records)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = Path(__file__).resolve().parent / f"backtest_{args.target}{suffix}.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved backtest results -> {out_path}")

    def report(pred_col: str, label: str) -> None:
        mae = mean_absolute_error(result["actual"], result[pred_col])
        rmse = np.sqrt(mean_squared_error(result["actual"], result[pred_col]))
        r2 = r2_score(result["actual"], result[pred_col])
        actual_cls = categorize(result["actual"].to_numpy(), lo, hi)
        pred_cls = categorize(result[pred_col].to_numpy(), lo, hi)
        accuracy = float((actual_cls == pred_cls).mean())
        cm = confusion_matrix(actual_cls, pred_cls, labels=["IN_SPEC", "OUT_OF_SPEC"])
        print(f"\n--- {label} ---")
        print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  spec-in/out accuracy={accuracy * 100:.2f}%")
        print(pd.DataFrame(cm, index=["Actual IN", "Actual OUT"], columns=["Pred IN", "Pred OUT"]))

    coverage = float(((result["actual"] >= result["pred_q10"]) & (result["actual"] <= result["pred_q90"])).mean())

    print(f"\n=== {args.target} backtest ({args.checkpoint}) ===")
    print(f"n = {len(result)}  (fresh-reading transitions only: {not args.all_hours})")
    report("naive_pred", "Naive baseline (repeat last observed reading)")
    report("pred", "Chronos-2")
    print(f"\n80% interval (q0.1-q0.9) empirical coverage = {coverage * 100:.2f}% (target: 80%)")


if __name__ == "__main__":
    main()
