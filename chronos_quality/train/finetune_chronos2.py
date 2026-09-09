"""
Fine-tune Chronos-2 as a quality net for blaine or residue.

Usage:
    python finetune_chronos2.py --target blaine --finetune-mode lora --num-steps 2000
    python finetune_chronos2.py --target residue --finetune-mode lora --num-steps 2000

    # Compare data variants (e.g. IQR on vs off) without overwriting each other's checkpoint:
    CHRONOS_PROCESSED_CSV=data/processed/with_iqr/quality_timeseries.csv \
        python finetune_chronos2.py --target blaine --output-tag with_iqr
    CHRONOS_PROCESSED_CSV=data/processed/without_iqr/quality_timeseries.csv \
        python finetune_chronos2.py --target blaine --output-tag without_iqr

Trains one model per target (mirrors the 1st-year ANN's separate best_model_blaine.h5 /
best_model_residue.h5), using the 9 controllable process variables as known-future covariates
(the policy/optimization loop sets these ahead of time) and the remaining process variables as
past covariates. Sparse blaine_prev/residue_prev columns provide the preceding actual lab
measurement at each scheduled measurement row.
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


def build_inputs(df, target: str, prediction_length: int):
    return from_data_frame(
        df=df,
        target_columns=[target],
        prediction_length=prediction_length,
        known_covariates_names=cfg.CONTROL_COLS + cfg.PREV_QUALITY_COLS,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=cfg.TARGET_COLS, required=True)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="lora")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device-map", default=cfg.default_device_map())
    parser.add_argument(
        "--output-tag",
        default="",
        help="optional suffix for the checkpoint dir, e.g. --output-tag with_iqr -> "
        "checkpoints/{target}_with_iqr/final (default: none -> checkpoints/{target}/final). "
        "Without this, runs against different data/hyperparams for the same target silently "
        "overwrite each other's checkpoint.",
    )
    args = parser.parse_args()

    df = load_processed(args.target)
    train_df, val_df, _test_df = chronological_split(df)

    print(f"train rows={len(train_df)} val rows={len(val_df)}")

    train_inputs = build_inputs(train_df, args.target, args.prediction_length)
    val_inputs = build_inputs(val_df, args.target, args.prediction_length)

    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=args.device_map)

    dir_name = f"{args.target}_{args.output_tag}" if args.output_tag else args.target
    output_dir = OUTPUT_ROOT / dir_name

    # Chronos2Pipeline.fit() only returns the finetuned pipeline, not the Trainer's step-by-step
    # loss history -- and reading it back from checkpoint-N/trainer_state.json afterward is
    # unreliable, because save_total_limit=1 + load_best_model_at_end=True (both hardcoded inside
    # fit() whenever validation_inputs is given) delete every checkpoint except the single best
    # one, silently dropping any steps logged after it if training kept going and got worse. This
    # callback records every log entry live as training happens, so the exported loss_history.csv
    # is never missing that data (see eval/loss_history.py docstring for the full story).
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

        # Agg backend: this script has no display and may run over SSH -- must be set before any
        # other matplotlib/pyplot import touches the default backend, so it's imported lazily here
        # rather than at module level (loss_history.py's own matplotlib import is already lazy).
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
