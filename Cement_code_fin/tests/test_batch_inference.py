"""Forecast plumbing tests; no claim to measure Chronos accuracy or GPU throughput."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.batch_inference import predict_tasks, compare_records


def sample_tasks():
    tasks = []
    for i in range(7):
        ts = pd.Timestamp('2024-01-02') + pd.Timedelta(hours=i)
        item = ['2CM', '3CM', '4CM'][i % 3]
        context = pd.DataFrame({
            'item_id': item, 'timestamp': pd.date_range(end=ts - pd.Timedelta(hours=1), periods=4, freq='h'),
            'blaine': [3600. + i, np.nan, 3610. + i, np.nan], 'control': float(i),
        })
        future = pd.DataFrame({'item_id': [item], 'timestamp': [ts], 'control': [i + 1.]})
        tasks.append((context, future, 3620. + i, ts))
    return tasks


class FakePipeline:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def predict_df(self, df, future_df, **kwargs):
        assert kwargs['cross_learning'] is False
        assert kwargs['prediction_length'] == 1
        assert kwargs['target'] not in future_df.columns
        self.calls.append(future_df.item_id.nunique())
        rows = []
        for _, f in future_df.iterrows():
            g = df[df.item_id == f.item_id]
            assert (g.timestamp < f.timestamp).all()
            value = g.blaine.dropna().iloc[-1] + f.control
            rows.append({'item_id': f.item_id, 'timestamp': f.timestamp,
                         'predictions': value, '0.1': value - 1, '0.9': value + 1})
        # Intentionally reverse output to exercise key-based matching.
        result = pd.DataFrame(rows[::-1])
        if self.failure == 'missing':
            result = result.iloc[1:]
        elif self.failure == 'duplicate':
            result.loc[result.index[-1], 'item_id'] = result.item_id.iloc[0]
        elif self.failure == 'timestamp':
            result['timestamp'] += pd.Timedelta(hours=1)
        elif self.failure == 'nan':
            result['predictions'] = np.nan
        return result


class BatchTests(unittest.TestCase):
    def test_serial_batch_identical_with_reordered_outputs_and_partial_chunk(self):
        tasks = sample_tasks()
        before = copy.deepcopy(tasks)
        serial, batch = FakePipeline(), FakePipeline()
        a = predict_tasks(serial, tasks, target='blaine', context_length=4)
        b = predict_tasks(batch, tasks, target='blaine', context_length=4, window_batch_size=3)
        self.assertEqual(a, b)
        self.assertEqual(serial.calls, [1] * 7)
        self.assertEqual(batch.calls, [3, 3, 1])
        self.assertEqual(compare_records(a, b)['max_abs_difference'], 0)
        for old, new in zip(before, tasks):
            pd.testing.assert_frame_equal(old[0], new[0])
            pd.testing.assert_frame_equal(old[1], new[1])

    def test_rejects_bad_model_outputs(self):
        for failure in ['missing', 'duplicate', 'timestamp', 'nan']:
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                predict_tasks(FakePipeline(failure), sample_tasks(), target='blaine',
                              context_length=4, window_batch_size=3)

    def test_rejects_future_target_and_context_leak(self):
        for failure in ['target', 'context']:
            tasks = sample_tasks()
            if failure == 'target':
                tasks[0][1]['blaine'] = 9999.
            else:
                tasks[0][0].loc[0, 'timestamp'] = tasks[0][3]
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                predict_tasks(FakePipeline(), tasks, target='blaine', context_length=4)

    def test_comparison_rejects_changed_predictions_or_labels(self):
        records = predict_tasks(FakePipeline(), sample_tasks(), target='blaine', context_length=4)
        for column in ['pred', 'actual']:
            changed = copy.deepcopy(records)
            changed[0][column] += 100
            with self.subTest(column=column), self.assertRaises(ValueError):
                compare_records(records, changed)


if __name__ == '__main__':
    unittest.main()
