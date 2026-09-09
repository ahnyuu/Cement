"""
Fine-tune Chronos-2 as a quality net for blaine or residue (one model per target, mirroring the
1st-year ANN's separate best_model_blaine.h5 / best_model_residue.h5).

The 9 controllable process variables (cfg.CONTROL_COLS) + the sparse blaine_prev/residue_prev
columns are passed as known-future covariates; the remaining process variables are past
covariates.

Defaults reflect chronos_quality/EXPERIMENT_LOG.md's confirmed choices:
    prediction_length = 4   (targets sit on a 4-hour grid; predlen>=4 guarantees every training
                             window contains a real measurement -> denser learning signal)
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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
from data.dataset_utils import chronological_split, load_processed
from eval.loss_history import (
    loss_history_from_callback,
    make_loss_history_callback,
    overfit_summary,
    plot_loss_history,
)

from chronos import Chronos2Pipeline
from chronos.chronos2.preprocess import from_data_frame

MODEL_ID = "amazon/chronos-2"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "checkpoints"


def build_inputs(df, target: str, prediction_length: int, include_prev: bool = True):
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
                        help="full 2026-09-06 study (24-1536): residue best at 256, blaine flat "
                        "from ~384, more context hurts residue+zero-shot. 256 recommended; "
                        "default still 512 pending the switch (see EXPERIMENT_LOG)")
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--no-prev", action="store_true",
                        help="drop blaine_prev/residue_prev entirely (experiments/prev/ A/B). "
                        "Default: both are known-future covariates for both target models.")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device-map", default=cfg.default_device_map())
    parser.add_argument(
        "--output-tag",
        default="",
        help="suffix for the checkpoint dir, e.g. --output-tag ctx1024 -> "
        "checkpoints/{target}_ctx1024/final. Without it, runs against different "
        "data/hyperparams for the same target silently overwrite each other's checkpoint.",
    )
    args = parser.parse_args()

    include_prev = not args.no_prev
    df = load_processed(args.target, include_prev=include_prev)
    train_df, val_df, _test_df = chronological_split(df)
    print(f"train rows={len(train_df)} val rows={len(val_df)}  include_prev={include_prev}")

    train_inputs = build_inputs(train_df, args.target, args.prediction_length, include_prev)
    val_inputs = build_inputs(val_df, args.target, args.prediction_length, include_prev)

    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=args.device_map)

    dir_name = f"{args.target}_{args.output_tag}" if args.output_tag else args.target
    output_dir = OUTPUT_ROOT / dir_name

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
    )

    save_path = output_dir / "final"
    finetuned.save_pretrained(save_path)
    print(f"Saved fine-tuned {args.target} quality net -> {save_path}")

    loss_df = loss_history_from_callback(loss_callback)
    loss_df.to_csv(output_dir / "loss_history.csv", index=False)
    summary = overfit_summary(loss_df)
    print(f"Saved loss history ({summary.get('n_val_points', 0)} val points) -> {output_dir / 'loss_history.csv'}")
    if summary.get("n_val_points"):
        gap = summary["final_minus_min"]
        flag = " <-- val loss rose from its min: overfitting" if gap > 0 else ""
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
