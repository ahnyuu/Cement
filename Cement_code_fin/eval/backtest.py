"""
Walk-forward (rolling-origin, 1-step-ahead) backtest for the Chronos-2 quality net:
  - MAE / RMSE / R2 in real units
  - spec-in/out Confusion-Matrix accuracy (blaine 3700-3900, residue 7-9)
  - normalized skill score vs. the naive "repeat last reading" baseline (primary metric)

At each evaluated timestamp t the model sees only history strictly before t (capped to
--context-length) plus the known control settings at t, and forecasts the reading at t. Models
fine-tuned at prediction_length=4 are still evaluated 1-step here -- chronos-2 handles inference
at a shorter horizon than training fine, so this stays comparable across runs.

blaine/residue targets must contain actual measurements only, with unmeasured hours left NaN.
Equal consecutive measurements are valid labels. Default split is validation for parameter
selection; --split test explicitly evaluates the historical, already-inspected test period.
A fixed eligibility history (default 24 hourly rows) defines the same evaluation points for
every context length >=24. Excluded points are recorded, not silently lost.

Usage:
    python eval/backtest.py --target blaine  --checkpoint checkpoints/blaine_ctx1024/final --tag ctx1024
    python eval/backtest.py --target blaine  --checkpoint amazon/chronos-2 --tag zeroshot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from eval.batch_inference import predict_tasks, compare_records
from data.dataset_utils import PROCESSED_PATH, chronological_split, load_processed
from data.research_protocol import PROTOCOL_VERSION, file_sha256, runtime_versions, validate_hourly_frame


def make_eval_tasks(
    df: pd.DataFrame,
    item_id: str,
    test_start_idx: int,
    stride: int,
    context_length: int,
    fresh_reading_only: bool = True,
    include_prev: bool = True,
    target: str | None = None,
    eligibility_context_length: int = 24,
    audit_rows: list | None = None,
):
    """One rolling-origin forecast task per evaluated row: (context_df, future_df, actual, ts).
    Context is capped to the last `context_length` rows strictly before the forecast timestamp."""
    g = df[df["item_id"] == item_id].reset_index(drop=True)
    target = target or g.columns[-1]
    if stride < 1 or eligibility_context_length < 1 or context_length < eligibility_context_length:
        raise ValueError("stride must be positive; context must be >= fixed eligibility context.")
    if not 0 <= test_start_idx <= len(g):
        raise ValueError("Evaluation start is outside the series.")

    if fresh_reading_only:
        # Actual-measurement CSV: observation is independent of a change in numeric value.
        observed = np.isfinite(g[target].to_numpy(dtype=float))
        candidate_idx = [i for i in range(test_start_idx, len(g)) if observed[i]]
    else:
        candidate_idx = list(range(test_start_idx, len(g)))

    tasks = []
    audit_rows = audit_rows if audit_rows is not None else []
    for i in candidate_idx[::stride]:
        audit = {"item_id": item_id, "timestamp": g.timestamp.iloc[i],
                 "actual": g[target].iloc[i], "status": "evaluated"}
        audit_rows.append(audit)
        if not np.isfinite(g[target].iloc[i]):
            audit["status"] = "no_measured_target"
            continue
        eligibility_history = g[target].iloc[max(0, i - eligibility_context_length):i]
        if not np.isfinite(eligibility_history.to_numpy(dtype=float)).any():
            audit["status"] = "no_target_in_fixed_eligibility_history"
            continue
        ctx_start = max(0, i - context_length)
        context_df = g.iloc[ctx_start:i]
        future_row = g.iloc[[i]]
        future_cols = cfg.CONTROL_COLS + (cfg.PREV_QUALITY_COLS if include_prev else [])
        future_df = future_row[["item_id", "timestamp"] + future_cols]
        tasks.append(
            (context_df, future_df, float(future_row[target].iloc[0]), future_row["timestamp"].iloc[0])
        )
    return tasks


def categorize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((values >= lo) & (values <= hi), "IN_SPEC", "OUT_OF_SPEC")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=cfg.TARGET_COLS, required=True)
    parser.add_argument("--checkpoint", default="amazon/chronos-2", help="local fine-tuned path or HF model id")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--eligibility-context-length", type=int, default=24,
                        help="Fixed history used to choose common points; must be <= every compared context.")
    parser.add_argument("--stride", type=int, default=4, help="evaluate every Nth eligible row")
    parser.add_argument("--device-map", default=cfg.default_device_map())
    parser.add_argument("--inference-window-batch-size", type=int, default=1,
                        help="Independent forecast windows per predict_df call. Use 8 to enable batching; 1 preserves serial execution.")
    parser.add_argument("--inference-batch-size", type=int, default=64,
                        help="Chronos inference channel budget (targets AND covariates); independent of training batch size.")
    parser.add_argument("--verify-batch-windows", type=int, default=8,
                        help="Compare the first N windows with serial inference before batched evaluation; 0 disables.")
    parser.add_argument("--all-hours", action="store_true",
                        help="Apply stride to hourly rows instead of measured rows; missing labels are never scored.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate tasks and coverage without loading a model or writing outputs.")
    parser.add_argument("--tag", default="",
                        help="suffix for the output csv, e.g. --tag zeroshot -> "
                        "backtest_{target}_zeroshot.csv")
    parser.add_argument("--no-prev", action="store_true",
                        help="drop blaine_prev/residue_prev (match a --no-prev checkpoint; "
                        "experiments/prev/ A/B)")
    args = parser.parse_args()
    if args.stride < 1 or args.eligibility_context_length < 1 or args.context_length < args.eligibility_context_length:
        parser.error("Positive stride required; context-length must be >= eligibility-context-length.")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.tag):
        parser.error("tag must contain only letters, digits, underscores and hyphens.")

    if args.inference_window_batch_size < 1 or args.inference_batch_size < 1 or args.verify_batch_windows < 0:
        parser.error("Inference batch sizes must be positive; verify-batch-windows must be nonnegative.")

    include_prev = not args.no_prev
    df = load_processed(args.target, include_prev=include_prev)
    validate_hourly_frame(df)
    train_df, val_df, test_df = chronological_split(df)
    if args.split == "validation":
        evaluation_df = val_df
        # Do not pass test rows to task creation when choosing hyperparameters.
        df = pd.concat([train_df, val_df], ignore_index=True).sort_values(["item_id", "timestamp"])
    else:
        evaluation_df = test_df
        print("Historical test period: prior experiments already used it for selection; not an untouched holdout.")
    lo, hi = cfg.SPEC_RANGES[args.target]

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(__file__).resolve().parent / PROTOCOL_VERSION / args.split
    out_path = out_dir / f"backtest_{args.target}{suffix}.csv"
    if not args.dry_run and any(out_path.with_suffix(suffix).exists() for suffix in (".csv", ".json", ".coverage.csv")):
        raise FileExistsError(f"Existing evaluation output: {out_path}. Use a new --tag.")

    audit_rows, task_groups = [], []
    for item_id in df["item_id"].unique():
        start = len(df[df.item_id == item_id]) - len(evaluation_df[evaluation_df.item_id == item_id])
        tasks = make_eval_tasks(
            df, item_id, start, args.stride, args.context_length,
            fresh_reading_only=not args.all_hours, include_prev=include_prev, target=args.target,
            eligibility_context_length=args.eligibility_context_length, audit_rows=audit_rows,
        )
        task_groups.append(tasks)
        print(f"{item_id}: {len(tasks)} eval points ({args.split})")
    audit = pd.DataFrame(audit_rows)
    if not any(task_groups):
        raise ValueError("No evaluation points with a measured target and eligible history.")
    print(audit.groupby(["item_id", "status"]).size().to_string())
    if args.dry_run:
        print("DRY RUN PASSED: no model loaded and no predictions generated.")
        return

    from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
    from eval.metrics_utils import macro_average_mae, normalized_skill_score

    pipeline = cfg.load_chronos_pipeline(args.checkpoint, device_map=args.device_map)
    tasks = [task for group in task_groups for task in group]
    inference_kwargs = dict(target=args.target, context_length=args.context_length,
                            inference_batch_size=args.inference_batch_size)
    verification = {"status": "not_requested"}
    if args.inference_window_batch_size > 1 and args.verify_batch_windows:
        sample = tasks[:args.verify_batch_windows]
        serial = predict_tasks(pipeline, sample, window_batch_size=1, **inference_kwargs)
        batched = predict_tasks(pipeline, sample, window_batch_size=args.inference_window_batch_size,
                                **inference_kwargs)
        verification = compare_records(serial, batched)
        print(f"Serial/batched sample check: {verification}")
    started = time.perf_counter()
    records = predict_tasks(pipeline, tasks, window_batch_size=args.inference_window_batch_size,
                            **inference_kwargs)
    inference_seconds = time.perf_counter() - started

    result = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    audit.to_csv(out_path.with_suffix(".coverage.csv"), index=False)
    metadata = {
        "protocol": PROTOCOL_VERSION, "arguments": vars(args), "n": len(result),
        "processed_csv": str(PROCESSED_PATH.resolve()), "data_sha256": file_sha256(PROCESSED_PATH),
        "versions": runtime_versions(), "prediction_length": 1,
        "inference_seconds": inference_seconds,
        "inference_windows_per_second": len(result) / inference_seconds,
        "batch_verification": verification,
        "mae": float(np.abs(result.actual - result.pred).mean()),
        "naive_mae": float(np.abs(result.actual - result.naive_pred).mean()),
        "excluded_points": int((audit.status != "evaluated").sum()),
        "evaluation_start": str(result.timestamp.min()), "evaluation_end": str(result.timestamp.max()),
        "test_status": "historical test already inspected; not an untouched holdout",
    }
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
        print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}  spec-in/out accuracy={accuracy * 100:.2f}%")
        print(pd.DataFrame(cm, index=["Actual IN", "Actual OUT"], columns=["Pred IN", "Pred OUT"]))

    coverage = float(((result["actual"] >= result["pred_q10"]) & (result["actual"] <= result["pred_q90"])).mean())
    skill = normalized_skill_score(result)
    macro = macro_average_mae(result)

    print(f"\n=== {args.target} backtest ({args.checkpoint}) ===")
    print(f"n = {len(result)}  (split={args.split}, measured-row stride={not args.all_hours})")
    report("naive_pred", "Naive baseline (repeat last observed reading)")
    report("pred", "Chronos-2")
    print(f"\nnormalized skill score (per-station model_MAE / naive_MAE, macro-avg) = {skill:.4f}  "
          f"({'beats' if skill < 1 else 'loses to'} naive)")
    print(f"macro-average MAE (equal per station) = {macro:.4f}")
    print(f"80% interval (q0.1-q0.9) empirical coverage = {coverage * 100:.2f}% (target: 80%)")


if __name__ == "__main__":
    main()
