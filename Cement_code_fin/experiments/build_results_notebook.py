"""
Assemble experiments/results_ctx_sweep.ipynb from whatever eval/backtest_*_ctx*_*.csv files
exist, then execute it in place.

    .venv/Scripts/python.exe experiments/build_results_notebook.py

Safe to run while the sweep is still going -- it just reports on the CSVs present so far.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "experiments" / "results_ctx_sweep.ipynb"

CONTEXT_LENGTHS = [512, 768, 1024, 1280, 1536]
TARGETS = ["blaine", "residue"]


def _md(text):
    return nbf.v4.new_markdown_cell(text)


def _code(src):
    return nbf.v4.new_code_cell(src)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Cement_code_fin (.venv)",
        "language": "python",
        "name": "cement_code_fin",
    }
    cells = []

    cells.append(_md(
        "# context_length sweep — blaine / residue\n\n"
        "Chronos-2 full fine-tune (predlen 4 train / 1-step eval, 1000 steps, lr 1e-6, batch 64), "
        "sweeping `context_length` over 512 / 768 / 1024 / 1280 / 1536. Each ctx is also evaluated "
        "zero-shot (no fine-tune), since context_length changes the inference input itself.\n\n"
        "Primary metric: **normalized skill score** = per-station `model_MAE / naive_MAE`, "
        "macro-averaged across the 3 stations (2CM/3CM/4CM). `< 1` beats the naive "
        "\"repeat last lab reading\" baseline.\n\n"
        "Source: `eval/backtest_{target}_ctx{N}_{zeroshot,full}.csv` written by "
        "`experiments/run_ctx_sweep.py`."
    ))

    cells.append(_code(
        "import sys\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'experiments' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import config as cfg\n"
        "from eval.metrics_utils import compute_metrics, macro_average_mae, normalized_skill_score\n\n"
        f"CONTEXT_LENGTHS = {CONTEXT_LENGTHS}\n"
        f"TARGETS = {TARGETS}\n"
        "pd.set_option('display.float_format', lambda v: f'{v:.4f}')"
    ))

    cells.append(_code(
        "def load_backtest(target, ctx, method):\n"
        "    p = ROOT / 'eval' / f'backtest_{target}_ctx{ctx}_{method}.csv'\n"
        "    if not p.exists():\n"
        "        return None\n"
        "    return pd.read_csv(p, parse_dates=['timestamp'])\n\n"
        "rows = []\n"
        "for target in TARGETS:\n"
        "    lo, hi = cfg.SPEC_RANGES[target]\n"
        "    for ctx in CONTEXT_LENGTHS:\n"
        "        for method in ['zeroshot', 'full']:\n"
        "            df = load_backtest(target, ctx, method)\n"
        "            if df is None:\n"
        "                continue\n"
        "            m = compute_metrics(df, (lo, hi), pred_col='pred')\n"
        "            naive = compute_metrics(df.assign(pred=df['naive_pred']), (lo, hi), pred_col='pred')\n"
        "            rows.append({\n"
        "                'target': target, 'ctx': ctx, 'method': method, 'n': len(df),\n"
        "                'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],\n"
        "                'spec_acc': m['spec_accuracy'], 'coverage80': m.get('interval_coverage', np.nan),\n"
        "                'macro_MAE': macro_average_mae(df),\n"
        "                'skill_score': normalized_skill_score(df),\n"
        "                'naive_MAE': naive['MAE'], 'naive_R2': naive['R2'], 'naive_spec_acc': naive['spec_accuracy'],\n"
        "            })\n\n"
        "summary = pd.DataFrame(rows)\n"
        "if summary.empty:\n"
        "    print('no backtest CSVs found yet')\n"
        "summary"
    ))

    for target in TARGETS:
        cells.append(_md(f"## {target}"))
        cells.append(_code(
            f"t = summary[summary['target'] == '{target}'].copy()\n"
            "if not t.empty:\n"
            "    view = t.set_index(['method', 'ctx'])[\n"
            "        ['n', 'MAE', 'RMSE', 'R2', 'spec_acc', 'coverage80', 'macro_MAE', 'skill_score']\n"
            "    ].sort_index()\n"
            "    display(view)\n"
            "    nb = t.groupby('ctx')[['naive_MAE', 'naive_R2', 'naive_spec_acc']].first()\n"
            "    print('naive baseline (per ctx = per eval set):')\n"
            "    display(nb)\n"
            "else:\n"
            "    print('no rows yet')"
        ))
        cells.append(_code(
            f"t = summary[summary['target'] == '{target}'].copy()\n"
            "if not t.empty:\n"
            "    fig, ax = plt.subplots(1, 3, figsize=(15, 4))\n"
            "    for method, mk in [('zeroshot', 'o--'), ('full', 'o-')]:\n"
            "        s = t[t['method'] == method].sort_values('ctx')\n"
            "        if s.empty:\n"
            "            continue\n"
            "        ax[0].plot(s['ctx'], s['skill_score'], mk, label=method)\n"
            "        ax[1].plot(s['ctx'], s['MAE'], mk, label=f'{method} (model)')\n"
            "        ax[2].plot(s['ctx'], s['coverage80'], mk, label=method)\n"
            "    ax[0].axhline(1.0, color='grey', lw=1, ls=':')\n"
            "    ax[0].set_title('normalized skill score (<1 beats naive)'); ax[0].set_xlabel('context_length'); ax[0].legend()\n"
            "    nmae = t.groupby('ctx')['naive_MAE'].first().sort_index()\n"
            "    ax[1].plot(nmae.index, nmae.values, 'k:', label='naive')\n"
            "    ax[1].set_title('pooled MAE'); ax[1].set_xlabel('context_length'); ax[1].legend()\n"
            "    ax[2].axhline(0.8, color='grey', lw=1, ls=':')\n"
            "    ax[2].set_title('80% interval empirical coverage'); ax[2].set_xlabel('context_length'); ax[2].legend()\n"
            f"    fig.suptitle('{target}')\n"
            "    plt.tight_layout(); plt.show()"
        ))

    cells.append(_md("## Best context_length per target"))
    cells.append(_code(
        "if not summary.empty:\n"
        "    full = summary[summary['method'] == 'full']\n"
        "    best = full.loc[full.groupby('target')['skill_score'].idxmin()]\n"
        "    display(best[['target', 'ctx', 'MAE', 'R2', 'spec_acc', 'coverage80', 'skill_score']])\n"
        "    for _, r in best.iterrows():\n"
        "        verdict = 'beats' if r['skill_score'] < 1 else 'loses to'\n"
        "        print(f\"{r['target']}: best ctx = {int(r['ctx'])}  \"\n"
        "              f\"(skill {r['skill_score']:.4f}, {verdict} naive; MAE {r['MAE']:.3f}, R2 {r['R2']:.4f})\")"
    ))

    cells.append(_md(
        "## Conclusion (2026-09-06)\n\n"
        "**`context_length = 512` for both targets.**\n\n"
        "- **blaine** is essentially flat across 512-1536 (full-finetune skill 0.824-0.832, all "
        "  inside retrain noise ~0.004). ctx1536 is nominally best on MAE/R2 but not meaningfully.\n"
        "- **residue** gets *monotonically worse* with more context (full-finetune skill "
        "  0.825 -> 0.857 as ctx 512 -> 1536). ctx512 is the clear best.\n"
        "- **zero-shot** degrades monotonically with context for both targets "
        "  (blaine 0.873 -> 0.914, residue 0.857 -> 0.976) -- longer history adds noise, not signal, "
        "  on this measured-actual dataset.\n"
        "- Every full-finetune combo beats the naive baseline (skill < 1); blaine ~0.83, residue ~0.83-0.86.\n"
        "- 80% interval coverage stays 49-68% (target 80%) at every ctx -- unchanged open issue.\n\n"
        "This **contradicts the old pipeline's finding** (blaine 1024 / residue 1536 best), which "
        "was measured on the pre-cleanup xlsx that had non-measurement-hour values bleeding in. "
        "On the measured-actual data, short context wins.\n\n"
        "ctx512 also trains fastest by far (~2.6 min vs ~92 min at ctx1536 on this 8GB GPU).\n\n"
        "Follow-up worth a look: sweep *below* 512 (256/384) -- the whole >=512 range is flat-to-worse, "
        "so the optimum may be shorter still.\n\n"
        "_Per-run timings: `experiments/ctx_sweep.log`. Write-up: `EXPERIMENT_LOG.md`._"
    ))

    nb.cells = cells
    return nb


def main() -> None:
    nb = build_notebook()
    NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {NB_PATH}")

    print("executing...")
    res = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=600", str(NB_PATH)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(res.stdout)
    print(res.stderr)
    if res.returncode != 0:
        raise SystemExit("nbconvert failed")
    print(f"executed -> {NB_PATH}")


if __name__ == "__main__":
    main()
