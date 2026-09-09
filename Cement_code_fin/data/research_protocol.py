"""Deterministic validation windows and run provenance; no model dependency."""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "review_v2"


def validate_hourly_frame(df: pd.DataFrame) -> None:
    if df.empty or df[["item_id", "timestamp"]].isna().any().any():
        raise ValueError("Empty data or missing item_id/timestamp.")
    if df.duplicated(["item_id", "timestamp"]).any():
        raise ValueError("Duplicate item_id/timestamp rows.")
    for item, g in df.groupby("item_id", sort=False):
        if not g.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"{item}: expected sorted hourly rows, retaining gaps as NaN.")


def validation_windows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    prediction_length: int,
    context_length: int,
    windows_per_item: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make independent series whose final horizon contains observed validation labels.

    Anchors are measured rows, selected chronologically with non-overlapping horizons.
    Up to windows_per_item anchors per station are spaced across the validation period;
    zero means all eligible anchors. Each window starts forecasting at its anchor.
    History can include training rows and EARLIER validation rows, never later rows.
    The official dataset still holds out the final prediction_length rows of each series.
    No time compression, target filling, or changes to training-window sampling occur.
    """
    if prediction_length < 1 or context_length < 1 or windows_per_item < 0:
        raise ValueError("Invalid horizon, context length, or window limit.")
    history = pd.concat([train_df, val_df], ignore_index=True)
    history = history.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    validate_hourly_frame(history)
    frames, rows = [], []
    for item, g in history.groupby("item_id", sort=False):
        g = g.reset_index(drop=True)
        vg = val_df.loc[val_df.item_id == item]
        if vg.empty:
            raise ValueError(f"{item}: no validation period.")
        v_start, v_end = vg.timestamp.min(), vg.timestamp.max()
        train_g = train_df.loc[train_df.item_id == item]
        if train_g.empty or train_g.timestamp.max() >= v_start:
            raise ValueError(f"{item}: training must end before validation starts.")
        observed = np.isfinite(g[target].to_numpy(dtype=float))
        candidates, next_free = [], 0
        for i in np.flatnonzero(observed & g.timestamp.between(v_start, v_end).to_numpy()):
            end = int(i) + prediction_length
            if i < next_free or i == 0 or end > len(g) or g.timestamp.iloc[end - 1] > v_end:
                continue
            candidates.append(int(i))
            next_free = end
        if not candidates:
            raise ValueError(f"{item}: no validation windows with observed targets.")
        if windows_per_item and len(candidates) > windows_per_item:
            selected = np.linspace(0, len(candidates) - 1, windows_per_item, dtype=int)
            candidates = [candidates[j] for j in selected]
        for window_number, i in enumerate(candidates):
            start, end = max(0, i - context_length), i + prediction_length
            window = g.iloc[start:end].copy()
            window_id = f"{item}__validation_{window_number:05d}"
            window["item_id"] = window_id
            n_labels = int(observed[i:end].sum())
            if not n_labels:
                raise ValueError("Validation window has no finite target labels.")
            frames.append(window)
            rows.append({
                "window_id": window_id, "item_id": item,
                "context_start": g.timestamp.iloc[start],
                "context_end": g.timestamp.iloc[i - 1],
                "forecast_start": g.timestamp.iloc[i],
                "forecast_end": g.timestamp.iloc[end - 1],
                "context_rows": i - start, "observed_labels": n_labels,
            })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No validation windows.")
    return pd.concat(frames, ignore_index=True), manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict:
    result = {}
    for package in ("chronos-forecasting", "torch", "transformers", "peft", "pandas", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def checkpoint_complete(checkpoint: Path) -> bool:
    """Do not reuse historical or interrupted runs just because a config exists."""
    metadata_path = checkpoint.parent / "run_metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    weights_exist = any(checkpoint.glob("*.safetensors")) or any(checkpoint.glob("*.bin"))
    return metadata.get("protocol") == PROTOCOL_VERSION and metadata.get("status") == "complete" and weights_exist


def evaluation_complete(csv: Path) -> bool:
    metadata_path = csv.with_suffix(".json")
    if not csv.exists() or not metadata_path.exists() or not csv.with_suffix(".coverage.csv").exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (metadata.get("protocol") == PROTOCOL_VERSION
            and metadata.get("arguments", {}).get("split") == "validation"
            and metadata.get("n", 0) > 0)
