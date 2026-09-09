"""Data/protocol regression tests, runnable without torch, Chronos or model downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg
from data.research_protocol import validation_windows, validate_hourly_frame, checkpoint_complete
from data.dataset_utils import chronological_split
from data.build_dataset import remove_iqr_outliers
from eval.backtest import make_eval_tasks
from experiments.summarize_validation import summarize


def sample(n=160):
    frames = []
    for item in ['2CM', '3CM', '4CM']:
        g = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=n, freq='h'), 'item_id': item})
        for col in cfg.CONTROL_COLS + cfg.MONITOR_COLS:
            g[col] = np.arange(n, dtype=float)
        g['blaine'] = np.where(np.arange(n) % 4 == 0, 3600 + np.arange(n), np.nan)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


class ProtocolTests(unittest.TestCase):
    def test_validation_horizons_have_labels_and_never_use_test(self):
        df = sample()
        train, val, test = chronological_split(df)
        for horizon in (1, 4):
            windows, manifest = validation_windows(train, val, 'blaine', horizon, 24, 4)
            self.assertEqual(len(manifest), 12)
            self.assertTrue((manifest.context_end < manifest.forecast_start).all())
            self.assertTrue((manifest.forecast_start >= val.timestamp.min()).all())
            self.assertTrue((manifest.forecast_end < test.timestamp.min()).all())
            for _, g in windows.groupby('item_id'):
                self.assertTrue(g.blaine.tail(horizon).notna().any())
                self.assertTrue(g.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all())
            repeated, second = validation_windows(train, val, 'blaine', horizon, 24, 4)
            pd.testing.assert_frame_equal(windows, repeated)
            pd.testing.assert_frame_equal(manifest, second)

    def test_empty_validation_labels_fail(self):
        train, val, _ = chronological_split(sample())
        val['blaine'] = np.nan
        with self.assertRaisesRegex(ValueError, 'no validation windows'):
            validation_windows(train, val, 'blaine', 4, 24)

    def test_duplicate_or_non_hourly_timeline_rejected(self):
        df = sample()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_hourly_frame(pd.concat([df, df.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, 'hourly'):
            validate_hourly_frame(df.drop(index=2))

    def test_equal_consecutive_measurements_are_kept(self):
        df = sample()
        df.loc[(df.item_id == '2CM') & (df.timestamp == df.timestamp.iloc[121]), 'blaine'] = df.blaine.iloc[120]
        tasks = make_eval_tasks(df, '2CM', 120, 1, 24, target='blaine')
        self.assertIn(df.timestamp.iloc[121], [t[3] for t in tasks])
        for context, future, _, timestamp in tasks:
            self.assertLess(context.timestamp.max(), timestamp)
            self.assertNotIn('blaine', future.columns)
            self.assertEqual(list(future.columns), ['item_id', 'timestamp'] + cfg.CONTROL_COLS + cfg.PREV_QUALITY_COLS)

    def test_same_points_and_baseline_for_different_contexts(self):
        df = sample()
        df.loc[df.timestamp.between(df.timestamp.iloc[90], df.timestamp.iloc[130]), 'blaine'] = np.nan
        signatures, audits = [], []
        for context_length in (24, 73, 128, 1536):
            audit = []
            tasks = make_eval_tasks(df, '2CM', 112, 1, context_length, target='blaine', audit_rows=audit)
            signatures.append([(t[3], t[2], t[0].blaine.dropna().iloc[-1]) for t in tasks])
            audits.append(audit)
        self.assertTrue(all(s == signatures[0] for s in signatures))
        self.assertTrue(all(a == audits[0] for a in audits))
        self.assertTrue(any(a['status'] != 'evaluated' for a in audits[0]))

    def test_stride_counts_measurements(self):
        tasks = make_eval_tasks(sample(), '2CM', 120, 4, 24, target='blaine')
        self.assertEqual([t[3].hour for t in tasks], [0, 16, 8])
        self.assertTrue(all(tasks[i+1][3] - tasks[i][3] == pd.Timedelta(hours=16) for i in range(len(tasks)-1)))

    def test_iqr_heldout_changes_cannot_change_training_fences(self):
        original = sample(100)
        altered = original.copy()
        altered.loc[altered.timestamp >= original.timestamp.iloc[70], 'feed_total'] = 1e9
        cleaned_a, _ = remove_iqr_outliers(original, ['feed_total', 'blaine'])
        cleaned_b, _ = remove_iqr_outliers(altered, ['feed_total', 'blaine'])
        self.assertEqual(cleaned_a.attrs['iqr_fences'], cleaned_b.attrs['iqr_fences'])
        for cleaned, source in ((cleaned_a, original), (cleaned_b, altered)):
            heldout = source.timestamp >= source.timestamp.iloc[70]
            pd.testing.assert_series_equal(cleaned.loc[heldout, 'blaine'], source.loc[heldout, 'blaine'])

    def test_incomplete_checkpoint_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            checkpoint = run / 'final'
            checkpoint.mkdir()
            (checkpoint / 'config.json').write_text('{}')
            self.assertFalse(checkpoint_complete(checkpoint))
            (run / 'run_metadata.json').write_text(json.dumps({'protocol': 'review_v2', 'status': 'started'}))
            (checkpoint / 'model.safetensors').write_bytes(b'test fixture')
            self.assertFalse(checkpoint_complete(checkpoint))
            (run / 'run_metadata.json').write_text(json.dumps({'protocol': 'review_v2', 'status': 'complete'}))
            self.assertTrue(checkpoint_complete(checkpoint))

    def test_every_experiment_uses_validation_and_new_paths(self):
        for path in (ROOT / 'experiments').rglob('run*.py'):
            text = path.read_text(encoding='utf-8')
            self.assertIn('"eval/backtest.py", "--split", "validation"', text, str(path))
            self.assertIn('"checkpoints" / "review_v2"', text, str(path))
            self.assertIn('"eval" / "review_v2" / "validation"', text, str(path))

    def test_summary_rejects_test_results_and_mismatched_points(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = pd.DataFrame({'item_id': ['2CM', '2CM'],
                                   'timestamp': pd.date_range('2024-02-01', periods=2, freq='4h'),
                                   'actual': [10., 12.], 'pred': [11., 11.], 'naive_pred': [9., 10.]})
            metadata = {'protocol': 'review_v2', 'processed_csv': 'fixture.csv',
                        'arguments': {'target': 'blaine', 'tag': 'a', 'context_length': 24,
                                      'checkpoint': 'fixture', 'split': 'test'}}
            path = directory / 'backtest_blaine_a.csv'
            result.to_csv(path, index=False)
            path.with_suffix('.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'only review_v2 validation'):
                summarize(directory)
            metadata['arguments']['split'] = 'validation'
            path.with_suffix('.json').write_text(json.dumps(metadata))
            self.assertEqual(summarize(directory).MAE.iloc[0], 1.)
            second_path = directory / 'backtest_blaine_b.csv'
            result.iloc[:1].to_csv(second_path, index=False)
            second_path.with_suffix('.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'evaluation points/labels/baseline differ'):
                summarize(directory)

    def test_no_previous_quality_mode_removes_future_columns(self):
        df = sample().drop(columns=cfg.PREV_QUALITY_COLS)
        tasks = make_eval_tasks(df, '2CM', 120, 4, 24, target='blaine', include_prev=False)
        self.assertTrue(tasks)
        self.assertTrue(all(not set(cfg.PREV_QUALITY_COLS).intersection(t[1].columns) for t in tasks))


if __name__ == '__main__':
    unittest.main()
