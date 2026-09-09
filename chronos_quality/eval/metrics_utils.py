"""Shared helpers for aggregating backtest results across stations (item_id: 2CM/3CM/4CM) when
comparing hyperparameter values (context_length, prediction_length, finetune_mode, ...).

Background (2026-08-27 design discussion): pooling all stations' rows together before computing
MAE (micro-average) implicitly weights whichever station has larger raw errors more heavily --
confirmed that 2CM's blaine/residue readings are intrinsically more volatile than 3CM/4CM's
(bigger swings between consecutive lab readings, so even the naive baseline is worse there), so a
pooled metric leans toward "did this help 2CM" more than the other two stations. macro_average_mae
fixes the unequal-weighting problem; normalized_skill_score additionally corrects for stations
having genuinely different task difficulty. All three (pooled MAE, macro_average_mae,
normalized_skill_score) happened to agree on every hyperparameter sweep re-checked so far
(context_length up to 1536, for both blaine and residue), but normalized_skill_score is the
recommended primary metric going forward since it is the most principled of the three -- use it
by default; report pooled/macro alongside for context if needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def categorize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where((values >= lo) & (values <= hi), "IN_SPEC", "OUT_OF_SPEC")


def compute_metrics(df: pd.DataFrame, spec_range: tuple[float, float], pred_col: str = "pred") -> dict:
    """Standard MAE/RMSE/R2/spec-accuracy (+ 80% interval coverage if pred_q10/pred_q90 present)
    for one DataFrame -- pooled across stations, or pre-filtered to a single item_id."""
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
    """MAE averaged equally across stations (each station counts once toward the average),
    instead of pooling all rows together first (which implicitly weights stations with more rows
    / larger raw errors more heavily)."""
    per_station = df.groupby(id_column).apply(lambda g: mean_absolute_error(g["actual"], g[pred_col]))
    return float(per_station.mean())


def normalized_skill_score(
    df: pd.DataFrame,
    pred_col: str = "pred",
    naive_col: str = "naive_pred",
    id_column: str = "item_id",
) -> float:
    """Per-station MAE scaled by that station's own naive-baseline MAE (skill ratio =
    model_MAE / naive_MAE; < 1 means beating naive, 1.0 means tied with naive), then averaged
    equally across stations.

    Recommended as the primary metric for choosing between hyperparameter values: unlike a
    pooled or plain macro-average MAE, it isn't biased by a station's inherent task difficulty,
    so a hyperparameter value only "wins" here if it genuinely improves on that station's own
    baseline -- not just because that station's raw error scale happens to be larger or smaller.
    """
    ratios = []
    for _, g in df.groupby(id_column):
        model_mae = mean_absolute_error(g["actual"], g[pred_col])
        naive_mae = mean_absolute_error(g["actual"], g[naive_col])
        ratios.append(model_mae / naive_mae)
    return float(np.mean(ratios))
