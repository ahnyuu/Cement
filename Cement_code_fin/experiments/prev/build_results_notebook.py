"""
Assemble experiments/prev/results_prev_comparison.ipynb from the 4 prev-A/B backtests, execute it.

    .venv/Scripts/python.exe experiments/prev/build_results_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "experiments" / "prev" / "results_prev_comparison.ipynb"


def _md(t):
    return nbf.v4.new_markdown_cell(t)


def _code(s):
    return nbf.v4.new_code_cell(s)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Cement_code_fin (.venv)", "language": "python", "name": "cement_code_fin",
    }
    c = []
    c.append(_md(
        "# prev-quality covariate 유무 A/B — 실측치 데이터\n\n"
        "`blaine_prev` / `residue_prev` (직전 lab 실측값, sparse) 를 **known-future covariate로 "
        "넣을 때 vs 아예 뺄 때** 성능 비교. 구 folder 2026-08-12 비교의 실측치 재현.\n\n"
        "- context_length = 2026-09-06 스윕 최적값 (**blaine 512 / residue 256**)\n"
        "- full fine-tune / predlen 4 / 1000 steps / lr 1e-6 / batch 64 / stride 4\n"
        "- `prev` arm = 현행 canonical (sparse prev, 두 타깃 모델 다 blaine_prev+residue_prev 사용)\n"
        "- `noprev` arm = `--no-prev` (두 컬럼 완전 제거 — past covariate로도 안 들어감)\n"
        "- 같은 canonical CSV, 같은 평가 지점 / naive baseline → 공정 비교\n\n"
        "정규화 skill-score `< 1` = naive(직전값 반복) 이김. GPU 안 씀."
    ))
    c.append(_code(
        "import sys\nfrom pathlib import Path\n\n"
        "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
        "try:\n    plt.rcParams['font.family'] = 'Malgun Gothic'\n    plt.rcParams['axes.unicode_minus'] = False\n"
        "except Exception:\n    pass\n\n"
        "ROOT = Path.cwd()\n"
        "while not (ROOT / 'config.py').exists() and ROOT != ROOT.parent:\n    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import config as cfg\n"
        "from eval.metrics_utils import compute_metrics, macro_average_mae, normalized_skill_score\n\n"
        "CTX = {'blaine': 512, 'residue': 256}\n"
        "ARMS = ['noprev', 'prev']\n"
        "LABEL = {'noprev': 'prev 미사용', 'prev': 'prev 사용 (sparse)'}\n"
        "COLOR = {'noprev': 'tab:red', 'prev': 'tab:green'}\n"
    ))

    c.append(_md("## 1. 결과 로딩"))
    c.append(_code(
        "res, missing = {}, []\n"
        "for t in cfg.TARGET_COLS:\n"
        "    for arm in ARMS:\n"
        "        p = ROOT / 'eval' / f'backtest_{t}_ctx{CTX[t]}_{arm}.csv'\n"
        "        if p.exists():\n"
        "            res[(t, arm)] = pd.read_csv(p, parse_dates=['timestamp'])\n"
        "        else:\n"
        "            missing.append(p.name)\n"
        "print('missing:', missing or 'NONE', '| loaded:', len(res), '/ 4')\n"
    ))

    c.append(_md("## 2. 요약표"))
    c.append(_code(
        "rows = []\n"
        "for t in cfg.TARGET_COLS:\n"
        "    for arm in ARMS:\n"
        "        df = res.get((t, arm))\n"
        "        if df is None:\n            continue\n"
        "        m = compute_metrics(df, cfg.SPEC_RANGES[t])\n"
        "        rows.append({'target': t, 'arm': arm, 'ctx': CTX[t], 'n': len(df),\n"
        "                     'MAE': round(m['MAE'], 4), 'R2': round(m['R2'], 4),\n"
        "                     'spec%': round(m['spec_accuracy'] * 100, 2),\n"
        "                     'cov80%': round(m['interval_coverage'] * 100, 2),\n"
        "                     'macro_MAE': round(macro_average_mae(df), 4),\n"
        "                     'skill': round(normalized_skill_score(df), 4),\n"
        "                     'naive_MAE': round(float(np.abs(df['actual'] - df['naive_pred']).mean()), 4)})\n"
        "summary = pd.DataFrame(rows).set_index(['target', 'arm'])\n"
        "summary\n"
    ))

    c.append(_md("## 3. Δ (prev 사용 − 미사용)\n\n음수 = prev 사용이 나음 (MAE/skill 기준)."))
    c.append(_code(
        "for t in cfg.TARGET_COLS:\n"
        "    a, b = res[(t, 'noprev')], res[(t, 'prev')]\n"
        "    ma, mb = compute_metrics(a, cfg.SPEC_RANGES[t]), compute_metrics(b, cfg.SPEC_RANGES[t])\n"
        "    sa, sb = normalized_skill_score(a), normalized_skill_score(b)\n"
        "    print(f'{t:8s} (ctx {CTX[t]}):  ΔMAE={mb[\"MAE\"]-ma[\"MAE\"]:+.4f}  ΔR2={mb[\"R2\"]-ma[\"R2\"]:+.4f}  '\n"
        "          f'Δskill={sb-sa:+.4f}   |  noprev skill={sa:.4f}  prev skill={sb:.4f}')\n"
    ))

    c.append(_md("## 4. 타깃별 비교 그래프\n\n빨강 = prev 미사용, 초록 = prev 사용. MAE 패널 회색 점선 = naive baseline."))
    c.append(_code(
        "def plot_target(t):\n"
        "    d = {arm: res[(t, arm)] for arm in ARMS}\n"
        "    m = {arm: compute_metrics(d[arm], cfg.SPEC_RANGES[t]) for arm in ARMS}\n"
        "    sk = {arm: normalized_skill_score(d[arm]) for arm in ARMS}\n"
        "    nmae = float(np.abs(d['prev']['actual'] - d['prev']['naive_pred']).mean())\n"
        "    panels = [('skill-score', sk, None), ('MAE', {k: m[k]['MAE'] for k in ARMS}, nmae),\n"
        "              ('R2', {k: m[k]['R2'] for k in ARMS}, None),\n"
        "              ('spec_accuracy', {k: m[k]['spec_accuracy'] for k in ARMS}, None)]\n"
        "    fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n"
        "    fig.suptitle(f'{t}  (ctx={CTX[t]}, full fine-tuning, n={len(d[\"prev\"])})',\n"
        "                 fontsize=13, fontweight='bold', y=1.05)\n"
        "    for ax, (name, vals, naive) in zip(axes, panels):\n"
        "        xs = np.arange(len(ARMS))\n"
        "        bars = ax.bar(xs, [vals[a] for a in ARMS], color=[COLOR[a] for a in ARMS], width=0.55)\n"
        "        for bb, a in zip(bars, ARMS):\n"
        "            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), f'{vals[a]:.4f}',\n"
        "                    ha='center', va='bottom', fontsize=9)\n"
        "        if naive is not None:\n"
        "            ax.axhline(naive, color='grey', ls=':', label=f'naive {naive:.3f}'); ax.legend(fontsize=8)\n"
        "        if name == 'skill-score':\n"
        "            ax.axhline(1.0, color='red', ls='--', lw=1)\n"
        "        ax.set_xticks(xs); ax.set_xticklabels([LABEL[a] for a in ARMS])\n"
        "        ax.set_title(name, fontsize=11, fontweight='bold')\n"
        "    plt.tight_layout(); plt.show()\n\n"
        "for t in cfg.TARGET_COLS:\n"
        "    plot_target(t)\n"
    ))

    c.append(_md("## 5. 결론"))
    c.append(_code(
        "for t in cfg.TARGET_COLS:\n"
        "    a, b = res[(t, 'noprev')], res[(t, 'prev')]\n"
        "    sa, sb = normalized_skill_score(a), normalized_skill_score(b)\n"
        "    ma = compute_metrics(a, cfg.SPEC_RANGES[t])['MAE']\n"
        "    mb = compute_metrics(b, cfg.SPEC_RANGES[t])['MAE']\n"
        "    win = 'prev 사용' if sb < sa else 'prev 미사용'\n"
        "    print(f'{t:8s}: noprev skill={sa:.4f} MAE={ma:.4f}  |  prev skill={sb:.4f} MAE={mb:.4f}  -> {win} 우세 '\n"
        "          f'(Δskill={sb-sa:+.4f})')\n"
    ))
    c.append(_md("<!-- 결론 텍스트는 EXPERIMENT_LOG.md 참조 -->"))
    nb.cells = c
    return nb


def main() -> None:
    NB_PATH.write_text(nbf.writes(build()), encoding="utf-8")
    print(f"wrote {NB_PATH}")
    r = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=600", str(NB_PATH)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("nbconvert failed")
    print(f"executed -> {NB_PATH}")


if __name__ == "__main__":
    main()
