"""
Capture the train/val loss curve of a fine-tuning run live, to check for overfitting.

Chronos2Pipeline.fit() runs a HuggingFace Trainer internally but only returns the finetuned
pipeline -- the step-by-step loss history is never returned. It's written to
checkpoint-N/trainer_state.json, but with save_total_limit=1 + load_best_model_at_end=True (both
hardcoded inside fit() when validation_inputs is given) only the single best-eval_loss checkpoint
survives, so any steps logged after it -- exactly the overfitting signal -- are pruned off disk.

LossHistoryCallback records every log entry as it happens, so the exported curve is never missing
that data regardless of what ends up on disk.
"""

from __future__ import annotations

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
    """A fresh TrainerCallback that records every logged train/eval metric as it happens.

    transformers is imported lazily so this module has no hard dependency on it unless you're
    about to call fit(). Trainer's CallbackHandler calls getattr(callback, event) for every
    lifecycle event and raises on a missing one, so we must subclass TrainerCallback (which
    supplies no-op defaults) rather than duck-typing just on_log.
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
    """Tidy DataFrame [step, train_loss, val_loss] from a make_loss_history_callback() instance."""
    return _tidy_log_history(callback.log_history)


def overfit_summary(df: pd.DataFrame) -> dict:
    """val_loss's minimum vs. its final value. A final value meaningfully above the minimum means
    val loss started climbing after training kept going -- the classic overfitting signature."""
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
