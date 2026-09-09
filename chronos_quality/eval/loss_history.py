"""
Extract train/val loss curves from a fine-tuning checkpoint directory, to check for overfitting.

Chronos2Pipeline.fit() runs a HuggingFace Trainer internally but only returns the finetuned
pipeline (see chronos/chronos2/pipeline.py) -- the Trainer's step-by-step loss history is never
returned to caller code. It IS written to disk, though: every checkpoint-N/ subfolder under
output_dir contains a trainer_state.json with a "log_history" list mixing two kinds of entries,
alternating by step:
    {"step": 100, "loss": ...}        <- train loss (logged every --logging-steps, hardcoded 100)
    {"step": 100, "eval_loss": ...}   <- val loss (logged every eval_steps, hardcoded 100)
This only exists when fit() was called with validation_inputs (finetune_chronos2.py always does).

CAVEAT: with save_total_limit=1 + load_best_model_at_end=True, only ONE checkpoint-N/ dir survives
training -- whichever had the best eval_loss -- and its trainer_state.json is written once, at the
moment that checkpoint was created. If training runs longer afterward and val_loss gets worse
(real overfitting), those later steps' checkpoints get deleted and take their log entries with
them; load_loss_history() below can only ever show history up to the retained best step. Confirmed
happening for some existing checkpoints (2026-08-19: ctx=1024 runs kept training 10-23 more
minutes after their best-saved step, well past what a "wrap-up" gap would explain, vs. ~1.5-5 min
for most other runs -- see time_after_best_checkpoint_s()). For any NEW run, use
LossHistoryCallback instead, which records every log entry live and isn't affected by this at all.

Usage (retrospective, existing checkpoint):
    from eval.loss_history import load_loss_history, plot_loss_history, overfit_summary
    df = load_loss_history("checkpoints/blaine_full_predlen4")
    plot_loss_history(df, title="blaine, ctx=512, full fine-tuning")
    print(overfit_summary(df))

Usage (live, during a new fit() call -- see also finetune_chronos2.py):
    from eval.loss_history import make_loss_history_callback, loss_history_from_callback
    cb = make_loss_history_callback()
    finetuned = pipeline.fit(..., callbacks=[cb])
    df = loss_history_from_callback(cb)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _tidy_log_history(log_history: list[dict]) -> pd.DataFrame:
    train = {e["step"]: e["loss"] for e in log_history if "loss" in e}
    val = {e["step"]: e["eval_loss"] for e in log_history if "eval_loss" in e}
    steps = sorted(set(train) | set(val))
    return pd.DataFrame(
        {
            "step": steps,
            "train_loss": [train.get(s) for s in steps],
            "val_loss": [val.get(s) for s in steps],
        }
    )


def make_loss_history_callback():
    """
    Returns a fresh TrainerCallback instance that records every logged train/eval metric AS IT
    HAPPENS, independent of which checkpoint directories survive save_total_limit pruning.

    WHY THIS EXISTS: with save_total_limit=1 + load_best_model_at_end=True (both hardcoded inside
    Chronos2Pipeline.fit() whenever validation_inputs is given -- see pipeline.py), only the
    single best-eval_loss checkpoint-N/ folder survives on disk. Its trainer_state.json is written
    ONCE, at the moment THAT checkpoint was created, and never touched again -- so if training
    keeps running afterward and val_loss gets worse (real overfitting), those later log entries
    exist only in checkpoints that get deleted, and load_loss_history() below can't see them. That
    made every retrospective check on pre-existing checkpoints potentially blind to "did it get
    worse after the best point" (confirmed happening for the ctx=1024 runs, 2026-08-19).

    Must subclass transformers.TrainerCallback (imported lazily here, so this module has no hard
    `transformers` dependency unless you're actually about to call fit() -- eval/backtest.py etc.
    only need load_loss_history/plotting). Duck-typing just on_log isn't enough: Trainer's
    CallbackHandler.call_event() calls getattr(callback, event) for every lifecycle event
    (on_train_begin, on_step_begin, on_evaluate, on_save, ...), not just on_log, and raises
    AttributeError on any callback missing one; TrainerCallback supplies no-op defaults for all of
    them so only on_log needs overriding here.

    Pass the result via Chronos2Pipeline.fit(..., callbacks=[cb]), then build a tidy DataFrame
    from cb.log_history with loss_history_from_callback(cb) -- this captures the FULL curve,
    including any post-best degradation, regardless of what ends up on disk.
    """
    from transformers.trainer_callback import TrainerCallback

    class LossHistoryCallback(TrainerCallback):
        def __init__(self):
            self.log_history: list[dict] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                self.log_history.append({**logs, "step": state.global_step})
            return control

    return LossHistoryCallback()


def loss_history_from_callback(callback) -> pd.DataFrame:
    """Tidy DataFrame [step, train_loss, val_loss] from a make_loss_history_callback() instance
    that was passed to Chronos2Pipeline.fit(..., callbacks=[cb])."""
    return _tidy_log_history(callback.log_history)


def load_loss_history(checkpoint_dir: str | Path) -> pd.DataFrame:
    """
    Parse the surviving checkpoint-N/trainer_state.json under `checkpoint_dir` into a tidy
    DataFrame with columns [step, train_loss, val_loss] (one row per step at which either was
    logged; NaN where only the other was recorded at that exact step).

    `checkpoint_dir` is the run's output_dir (e.g. "checkpoints/blaine_full_predlen4"), not the
    checkpoint-N/ subfolder itself -- this glob-finds whichever checkpoint-N survived
    save_total_limit=1 pruning.

    CAVEAT (see LossHistoryCallback docstring above for the full explanation): this can only show
    history up to whichever step was last saved as the "best" checkpoint. If training continued
    past that step, any further degradation is invisible here -- use
    time_after_best_checkpoint_s() as an indirect check for whether that's likely, and prefer
    LossHistoryCallback for any run you're about to start.
    """
    checkpoint_dir = Path(checkpoint_dir)
    state_files = sorted(
        checkpoint_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not state_files:
        raise FileNotFoundError(
            f"no checkpoint-N/trainer_state.json under {checkpoint_dir} -- was this run trained "
            "without validation_inputs, or has the checkpoint dir been cleaned up?"
        )
    # if more than one survives (shouldn't normally happen with save_total_limit=1), the highest
    # step number has the most complete accumulated log_history
    with open(state_files[-1], encoding="utf-8") as f:
        log_history = json.load(f)["log_history"]
    return _tidy_log_history(log_history)


def time_after_best_checkpoint_s(checkpoint_dir: str | Path) -> float | None:
    """
    Wall-clock seconds between the retained checkpoint-N/ being written and the run's final/
    pipeline being saved (finetune_chronos2.py writes both). A large gap is an indirect sign that
    training kept going well past the visible best-loss point -- i.e. don't read a clean-looking
    loss_history.csv as proof this run never overfit; it may just mean the evidence for what
    happened afterward was pruned. Returns None if either path is missing.
    """
    checkpoint_dir = Path(checkpoint_dir)
    state_files = sorted(
        checkpoint_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    final_config = checkpoint_dir / "final" / "config.json"
    if not state_files or not final_config.exists():
        return None
    return final_config.stat().st_mtime - state_files[-1].stat().st_mtime


def overfit_summary(df: pd.DataFrame) -> dict:
    """
    Quick numeric signal for overfitting: val_loss's minimum vs. its final value. If the final
    value is meaningfully above the minimum, val loss has started climbing back up after training
    kept going -- the classic overfitting signature (train loss keeps falling, val loss doesn't).
    """
    val = df["val_loss"].dropna()
    if val.empty:
        return {"n_val_points": 0}
    min_idx = val.idxmin()
    return {
        "n_val_points": len(val),
        "val_loss_min": float(val.loc[min_idx]),
        "val_loss_min_step": int(df.loc[min_idx, "step"]),
        "val_loss_final": float(val.iloc[-1]),
        "val_loss_final_step": int(df.loc[val.index[-1], "step"]),
        "final_minus_min": float(val.iloc[-1] - val.loc[min_idx]),
        "train_loss_final": float(df["train_loss"].dropna().iloc[-1]) if df["train_loss"].notna().any() else None,
    }


def plot_loss_history(df: pd.DataFrame, ax=None, title: str | None = None):
    """Plot train_loss and val_loss vs. step on one axes (creates a new figure if ax is None)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    train = df.dropna(subset=["train_loss"])
    val = df.dropna(subset=["val_loss"])
    ax.plot(train["step"], train["train_loss"], marker="o", markersize=3, label="train_loss", color="tab:blue")
    ax.plot(val["step"], val["val_loss"], marker="o", markersize=3, label="val_loss", color="tab:red")
    if not val.empty:
        best_step = val.loc[val["val_loss"].idxmin(), "step"]
        ax.axvline(best_step, color="tab:red", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
