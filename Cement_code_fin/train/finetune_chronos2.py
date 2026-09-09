"""
Fine-tune Chronos-2 as a quality net for blaine or residue (one model per target, mirroring the
1st-year ANN's separate best_model_blaine.h5 / best_model_residue.h5).

The 9 controllable process variables (cfg.CONTROL_COLS) + the sparse blaine_prev/residue_prev
columns are passed as known-future covariates; the remaining process variables are past
covariates.

Defaults reflect chronos_quality/EXPERIMENT_LOG.md's confirmed choices:
    prediction_length = 4   (hourly steps; missing measurements mean some windows still have
                             no target labels; no guarantee of a label in every window)
    finetune_mode     = full
    num_steps         = 1000
    learning_rate     = 1e-6
context_length is NOT settled for this dataset -- sweep it via --context-length.

Usage:
    python train/finetune_chronos2.py --target blaine  --context-length 1024 --output-tag ctx1024
    python train/finetune_chronos2.py --target residue --context-length 1536 --output-tag ctx1536
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data.dataset_utils import PROCESSED_PATH, chronological_split, load_processed
from data.research_protocol import (
    PROTOCOL_VERSION, file_sha256, runtime_versions, validation_windows,
)
from eval.loss_history import (
    loss_history_from_callback,
    make_loss_history_callback,
    overfit_summary,
    plot_loss_history,
)

MODEL_ID = "amazon/chronos-2"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "checkpoints" / PROTOCOL_VERSION


def build_inputs(df, target: str, prediction_length: int, include_prev: bool = True):
    from chronos.chronos2.preprocess import from_data_frame

    known = cfg.CONTROL_COLS + (cfg.PREV_QUALITY_COLS if include_prev else [])
    return from_data_frame(
        df=df,
        target_columns=[target],
        prediction_length=prediction_length,
        known_covariates_names=known,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=cfg.TARGET_COLS, required=True)
    parser.add_argument("--prediction-length", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512,
                        help="Hourly history length. Historical test-selected values are exploratory.")
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--no-prev", action="store_true",
                        help="drop blaine_prev/residue_prev entirely (experiments/prev/ A/B). "
                        "Default: both are known-future covariates for both target models.")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-windows-per-item", type=int, default=128,
                        help="Deterministic windows spread across validation for each station; 0=all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check data and validation labels without loading/training a model or writing outputs.")
    parser.add_argument("--device-map", default=cfg.default_device_map())
    parser.add_argument(
        "--output-tag",
        default="",
        help="suffix for the checkpoint dir, e.g. --output-tag ctx1024 -> "
        "checkpoints/review_v2/{target}_ctx1024/final. Existing run directories are never overwritten.",
    )
    args = parser.parse_args()
    if min(args.prediction_length, args.context_length, args.num_steps, args.batch_size) < 1:
        parser.error("Lengths, steps and batch size must be positive.")
    if args.learning_rate <= 0 or args.validation_windows_per_item < 0:
        parser.error("Invalid learning rate or validation window count.")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.output_tag):
        parser.error("output-tag must contain only letters, digits, underscores and hyphens.")

    include_prev = not args.no_prev
    df = load_processed(args.target, include_prev=include_prev)
    train_df, val_df, _test_df = chronological_split(df)
    print(f"train rows={len(train_df)} val rows={len(val_df)}  include_prev={include_prev}")

    val_windows, val_manifest = validation_windows(
        train_df, val_df, args.target, args.prediction_length, args.context_length,
        args.validation_windows_per_item,
    )
    print("Validation windows (each includes at least one observed quality label):")
    print(val_manifest.groupby("item_id").agg(
        windows=("window_id", "size"), labels=("observed_labels", "sum"),
        first_forecast=("forecast_start", "min"), last_forecast=("forecast_end", "max"),
    ).to_string())
    if args.dry_run:
        print("DRY RUN PASSED: no model loaded and no training performed.")
        return

    dir_name = f"{args.target}_{args.output_tag}" if args.output_tag else args.target
    output_dir = OUTPUT_ROOT / dir_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output_dir}. Use a new --output-tag.")

    from chronos import Chronos2Pipeline
    from transformers import set_seed

    set_seed(args.seed)
    train_inputs = build_inputs(train_df, args.target, args.prediction_length, include_prev)
    val_inputs = build_inputs(val_windows, args.target, args.prediction_length, include_prev)
    # Verify the actual tensors handed to Chronos, in addition to the source dataframe.
    if len(val_inputs) != len(val_manifest):
        raise ValueError("Validation preprocessing changed the number of windows.")
    import torch
    for window in val_inputs:
        labels = window["context"][:window["n_targets"], -args.prediction_length:]
        if not torch.isfinite(labels).any().item():
            raise ValueError("Chronos validation tensor contains no finite quality labels.")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=args.device_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_manifest.to_csv(output_dir / "validation_windows.csv", index=False)
    metadata = {
        "protocol": PROTOCOL_VERSION, "status": "started", "arguments": vars(args),
        "processed_csv": str(PROCESSED_PATH.resolve()), "data_sha256": file_sha256(PROCESSED_PATH),
        "versions": runtime_versions(), "model_id": MODEL_ID,
        "validation_windows": len(val_manifest),
        "validation_observed_labels": int(val_manifest.observed_labels.sum()),
        "checkpoint_metric": "official eval_loss over observed validation labels; not MAE",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    loss_callback = make_loss_history_callback()
    finetuned = pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        validation_inputs=val_inputs,
        finetune_mode=args.finetune_mode,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_dir=output_dir,
        finetuned_ckpt_name="finetuned-ckpt",
        callbacks=[loss_callback],
        seed=args.seed,
        data_seed=args.seed,
    )

    save_path = output_dir / "final"
    finetuned.save_pretrained(save_path)
    print(f"Saved fine-tuned {args.target} quality net -> {save_path}")

    loss_df = loss_history_from_callback(loss_callback)
    loss_df.to_csv(output_dir / "loss_history.csv", index=False)
    if loss_df["val_loss"].dropna().empty or not torch.isfinite(
        torch.tensor(loss_df["val_loss"].dropna().to_numpy())
    ).all().item():
        raise RuntimeError("No finite validation loss was recorded; run is not marked complete.")
    metadata["status"] = "complete"
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = overfit_summary(loss_df)
    print(f"Saved loss history ({summary.get('n_val_points', 0)} val points) -> {output_dir / 'loss_history.csv'}")
    if summary.get("n_val_points"):
        gap = summary["final_minus_min"]
        flag = " <-- higher than the observed minimum; not proof of overfitting by itself" if gap > 0 else ""
        print(
            f"  val_loss: min={summary['val_loss_min']:.5f} (step {summary['val_loss_min_step']}), "
            f"final={summary['val_loss_final']:.5f} (step {summary['val_loss_final_step']}), "
            f"final-min={gap:+.5f}{flag}"
        )

        # Agg backend: this script has no display and may run over SSH -- set before pyplot import.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = plot_loss_history(loss_df, title=f"{args.target}, ctx={args.context_length}, {args.finetune_mode}")
        plot_path = output_dir / "loss_history.png"
        ax.figure.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(ax.figure)
        print(f"Saved loss curve -> {plot_path}")


if __name__ == "__main__":
    main()
