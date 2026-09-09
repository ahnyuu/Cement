"""
Aggregate backtest results across stations (item_id: 2CM/3CM/4CM) when comparing hyperparameters.

Pooling all stations' rows before computing MAE implicitly weights whichever station has larger
raw errors (2CM's readings are intrinsically more volatile). macro_average_mae fixes the unequal
weighting; normalized_skill_score additionally corrects for stations having genuinely different
task difficulty -- use it as the primary metric for choosing between hyperparameter values, and
report pooled/macro alongside for context.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def categorize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((values >= lo) & (values <= hi), "IN_SPEC", "OUT_OF_SPEC")


def compute_metrics(df: pd.DataFrame, spec_range: tuple[float, float], pred_col: str = "pred") -> dict:
    """MAE/RMSE/R2/spec-accuracy (+ 80% interval coverage if pred_q10/pred_q90 present), pooled
    across stations or pre-filtered to one item_id."""
    lo, hi = spec_range
    mae = mean_absolute_error(df["actual"], df[pred_col])
    rmse = np.sqrt(mean_squared_error(df["actual"], df[pred_col]))
    r2 = r2_score(df["actual"], df[pred_col])
    actual_cls = categorize(df["actual"].to_numpy(), lo, hi)
    pred_cls = categorize(df[pred_col].to_numpy(), lo, hi)
    spec_accuracy = float((actual_cls == pred_cls).mean())
    row = {"MAE": mae, "RMSE": rmse, "R2": r2, "spec_accuracy": spec_accuracy}
    if pred_col == "pred" and {"pred_q10", "pred_q90"} <= set(df.columns):
        row["interval_coverage"] = float(
            ((df["actual"] >= df["pred_q10"]) & (df["actual"] <= df["pred_q90"])).mean()
        )
    return row


def macro_average_mae(df: pd.DataFrame, pred_col: str = "pred", id_column: str = "item_id") -> float:
    """MAE averaged equally across stations (each station counts once), instead of pooling rows."""
    per_station = df.groupby(id_column).apply(lambda g: mean_absolute_error(g["actual"], g[pred_col]))
    return float(per_station.mean())


def normalized_skill_score(
    df: pd.DataFrame,
    pred_col: str = "pred",
    naive_col: str = "naive_pred",
    id_column: str = "item_id",
) -> float:
    """Per-station MAE / that station's own naive-baseline MAE (< 1 = beating naive, 1.0 = tied),
    averaged equally across stations. Primary metric for choosing between hyperparameter values:
    unlike pooled or plain macro-average MAE it isn't biased by a station's inherent difficulty."""
    ratios = []
    for _, g in df.groupby(id_column):
        model_mae = mean_absolute_error(g["actual"], g[pred_col])
        naive_mae = mean_absolute_error(g["actual"], g[naive_col])
        ratios.append(model_mae / naive_mae)
    return float(np.mean(ratios))
