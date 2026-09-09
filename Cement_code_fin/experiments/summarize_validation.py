"""Summarize only review_v2 validation predictions on identical points; never rank test runs."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.research_protocol import PROTOCOL_VERSION


def summarize(directory: Path, pattern: str = 'backtest_*.csv') -> pd.DataFrame:
    rows, reference = [], {}
    for path in sorted(directory.glob(pattern)):
        if path.name.endswith('.coverage.csv'):
            continue
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        args = metadata['arguments']
        if metadata.get('protocol') != PROTOCOL_VERSION or args.get('split') != 'validation':
            raise ValueError(f'{path.name}: only review_v2 validation results may be ranked.')
        df = pd.read_csv(path, parse_dates=['timestamp']).sort_values(['item_id', 'timestamp']).reset_index(drop=True)
        if df.empty or df.duplicated(['item_id', 'timestamp']).any():
            raise ValueError(f'{path.name}: empty or duplicate evaluation rows.')
        if not np.isfinite(df[['actual', 'pred', 'naive_pred']].to_numpy()).all():
            raise ValueError(f'{path.name}: non-finite scores.')
        target = args['target']
        points = df[['item_id', 'timestamp', 'actual', 'naive_pred']]
        if target in reference:
            try:
                pd.testing.assert_frame_equal(points, reference[target], check_exact=True)
            except AssertionError as error:
                raise ValueError(f'{path.name}: evaluation points/labels/baseline differ; do not rank together.') from error
        else:
            reference[target] = points
        errors = df.assign(error=(df.actual - df.pred).abs(), naive_error=(df.actual - df.naive_pred).abs())
        station = errors.groupby('item_id')[['error', 'naive_error']].mean()
        ratios = station.error / station.naive_error.replace(0, np.nan)
        rows.append({
            'target': target, 'tag': args['tag'], 'context_length': args['context_length'],
            'n': len(df), 'MAE': float(errors.error.mean()),
            'macro_MAE': float(station.error.mean()), 'naive_MAE': float(errors.naive_error.mean()),
            'normalized_MAE_ratio': float(ratios.mean()) if ratios.notna().all() else np.nan,
            'checkpoint': args['checkpoint'], 'no_prev': args.get('no_prev', False),
            'processed_csv': metadata['processed_csv'], 'results_csv': str(path.resolve()),
        })
    if not rows:
        raise ValueError('No completed validation results matched the pattern.')
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='backtest_*.csv', help='Limit comparison to the intended study.')
    parser.add_argument('--metric', choices=['MAE', 'macro_MAE', 'normalized_MAE_ratio'], default='MAE')
    args = parser.parse_args()
    directory = ROOT / 'eval' / PROTOCOL_VERSION / 'validation'
    summary = summarize(directory, args.pattern)
    if summary[args.metric].isna().any():
        raise ValueError('Selected metric is undefined for some runs; choose another metric.')
    summary = summary.sort_values(['target', args.metric, 'tag']).reset_index(drop=True)
    summary.to_csv(directory / 'validation_summary.csv', index=False)
    selected = summary.groupby('target', sort=False).head(1)
    (directory / 'selected_by_validation.json').write_text(json.dumps({
        'protocol': PROTOCOL_VERSION, 'selection_split': 'validation', 'metric': args.metric,
        'pattern': args.pattern, 'selected': selected.to_dict(orient='records'),
        'note': 'Selection among matching completed runs only; historical test remains previously inspected.',
    }, indent=2), encoding='utf-8')
    print(summary[['target', 'tag', 'n', args.metric]].to_string(index=False))
    print('Saved validation_summary.csv and selected_by_validation.json; no test evaluation was run.')


if __name__ == '__main__':
    main()
