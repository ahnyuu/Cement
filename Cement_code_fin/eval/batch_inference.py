"""Batch independent rolling forecast windows without sharing information between origins.

Only execution grouping changes: each task still predicts one hour from its own past.
Keep cross_learning=False: later windows can contain labels unavailable to earlier ones.
"""
import numpy as np
import pandas as pd


def predict_tasks(pipeline, tasks, *, target, context_length, window_batch_size=1,
                  inference_batch_size=64):
    if min(context_length, window_batch_size, inference_batch_size) < 1:
        raise ValueError("Context and batch sizes must be positive.")
    records = []
    for start in range(0, len(tasks), window_batch_size):
        chunk = tasks[start:start + window_batch_size]
        contexts, futures, mapping = [], [], {}
        for offset, (context, future, actual, timestamp) in enumerate(chunk):
            ts = pd.Timestamp(timestamp)
            if len(future) != 1 or pd.Timestamp(future.timestamp.iloc[0]) != ts:
                raise ValueError("Each task must have exactly one matching forecast timestamp.")
            if context.empty or len(context) > context_length or not (context.timestamp < ts).all():
                raise ValueError("Context must contain only the requested history before the forecast.")
            if target in future.columns:
                raise ValueError("The future input must not contain the target column.")
            item = future.item_id.iloc[0]
            if not context.item_id.eq(item).all():
                raise ValueError("Context/future item IDs do not match.")
            if not np.isfinite(actual):
                raise ValueError("Evaluation requires a measured target.")
            observed = context[target].dropna()
            if observed.empty or not np.isfinite(observed.iloc[-1]):
                raise ValueError("No finite previous target for the naive baseline.")
            # Synthetic IDs identify independent requests, not new learned station features.
            request_id = f"window_{start + offset:09d}"
            contexts.append(context.assign(item_id=request_id))
            futures.append(future.assign(item_id=request_id))
            mapping[request_id] = (item, ts, float(actual), float(observed.iloc[-1]))
        pred = pipeline.predict_df(
            df=pd.concat(contexts, ignore_index=True),
            future_df=pd.concat(futures, ignore_index=True),
            id_column="item_id", timestamp_column="timestamp", target=target,
            prediction_length=1, quantile_levels=[0.1, 0.5, 0.9],
            context_length=context_length, batch_size=inference_batch_size,
            cross_learning=False,
        )
        required = {"item_id", "timestamp", "predictions", "0.1", "0.9"}
        if not required.issubset(pred.columns):
            raise ValueError(f"Missing prediction columns: {required - set(pred.columns)}")
        if (len(pred) != len(mapping) or pred.item_id.duplicated().any()
                or set(pred.item_id) != set(mapping)):
            raise ValueError("Missing, duplicate or unexpected forecast window IDs.")
        if not np.isfinite(pred[["predictions", "0.1", "0.9"]].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite model output; refusing partial metrics.")
        indexed = pred.set_index("item_id")
        # Match by ID and timestamp, never by a library's output row order.
        for request_id, (item, ts, actual, naive) in mapping.items():
            row = indexed.loc[request_id]
            if pd.Timestamp(row["timestamp"]) != ts:
                raise ValueError("Prediction timestamp does not match the forecast window.")
            records.append({
                "item_id": item, "timestamp": ts, "actual": actual, "naive_pred": naive,
                "pred": float(row["predictions"]), "pred_q10": float(row["0.1"]),
                "pred_q90": float(row["0.9"]),
            })
    return records


def compare_records(serial, batched, *, rtol=1e-5, atol=1e-5):
    """A smoke check, not proof for every timestamp; allow small floating-point drift."""
    left, right = pd.DataFrame(serial), pd.DataFrame(batched)
    if left.empty or len(left) != len(right):
        raise ValueError("Serial/batch comparison requires equal nonempty results.")
    identity = ["item_id", "timestamp", "actual", "naive_pred"]
    if not left[identity].equals(right[identity]):
        raise ValueError("Serial/batch comparison has different evaluation points or labels.")
    columns = ["pred", "pred_q10", "pred_q90"]
    a, b = left[columns].to_numpy(float), right[columns].to_numpy(float)
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
        raise ValueError("Batched predictions differ beyond tolerance. Use window-batch-size 1 and investigate.")
    return {"status": "passed", "windows": len(left), "rtol": rtol, "atol": atol,
            "max_abs_difference": float(np.max(np.abs(a - b)))}
