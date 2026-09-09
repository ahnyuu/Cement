"""
Assemble experiments/predlen/results_predlen_comparison.ipynb from the 4 predlen-A/B backtests.

    .venv/Scripts/python.exe experiments/predlen/build_results_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "experiments" / "predlen" / "results_predlen_comparison.ipynb"


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
        "# prediction_length (train) 1 vs 4 A/B — 실측치 데이터\n\n"
        "학습 시 `--prediction-length` 를 4(현행) vs 1 로 두고 비교. 구 folder 2026-08-18 predlen "
        "스윕의 실측치 재현. context_length = 스윕 최적값 (**blaine 512 / residue 256**).\n\n"
        "- full fine-tune / 1000 steps / lr 1e-6 / batch 64 / stride 4\n"
        "- **평가는 두 arm 모두 1-step** (`backtest.py`가 `predict_df` 를 `prediction_length=1` 로 "
        "  하드코딩 — chronos-2는 학습보다 짧은 horizon 추론 OK). 그래서 \"1-step 예측 정확도\"라는 "
        "  잣대가 두 arm에 동일하게 적용됨.\n"
        "- 같은 canonical CSV, 같은 평가 지점 / naive baseline\n\n"
        "**가설**: 타깃(blaine/residue)이 4시간 grid에만 존재 → predlen=1이면 랜덤 학습 윈도우의 다음 "
        "1시간이 실측 시각일 확률이 낮아 학습 신호 낭비. predlen≥4면 어떤 4시간 윈도우든 실측 시각 "
        "포함. 실측치는 타깃이 더 sparse(82.6% NaN)라 이 효과가 더 클 것.\n\n"
        "정규화 skill-score `< 1` = naive 이김. GPU 안 씀."
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
        "ARMS = ['predlen1', 'predlen4']\n"
        "LABEL = {'predlen1': 'predlen=1', 'predlen4': 'predlen=4 (현행)'}\n"
        "COLOR = {'predlen1': 'tab:red', 'predlen4': 'tab:green'}\n"
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

    c.append(_md("## 3. Δ (predlen4 − predlen1)\n\n음수 = predlen4가 나음 (MAE/skill 기준)."))
    c.append(_code(
        "for t in cfg.TARGET_COLS:\n"
        "    a, b = res[(t, 'predlen1')], res[(t, 'predlen4')]\n"
        "    ma, mb = compute_metrics(a, cfg.SPEC_RANGES[t]), compute_metrics(b, cfg.SPEC_RANGES[t])\n"
        "    sa, sb = normalized_skill_score(a), normalized_skill_score(b)\n"
        "    print(f'{t:8s} (ctx {CTX[t]}):  ΔMAE={mb[\"MAE\"]-ma[\"MAE\"]:+.4f}  ΔR2={mb[\"R2\"]-ma[\"R2\"]:+.4f}  '\n"
        "          f'Δskill={sb-sa:+.4f}   |  predlen1 skill={sa:.4f}  predlen4 skill={sb:.4f}')\n"
    ))

    c.append(_md("## 4. 타깃별 비교 그래프\n\n빨강 = predlen1, 초록 = predlen4. MAE 패널 회색 점선 = naive baseline."))
    c.append(_code(
        "def plot_target(t):\n"
        "    d = {arm: res[(t, arm)] for arm in ARMS}\n"
        "    m = {arm: compute_metrics(d[arm], cfg.SPEC_RANGES[t]) for arm in ARMS}\n"
        "    sk = {arm: normalized_skill_score(d[arm]) for arm in ARMS}\n"
        "    nmae = float(np.abs(d['predlen4']['actual'] - d['predlen4']['naive_pred']).mean())\n"
        "    panels = [('skill-score', sk, None), ('MAE', {k: m[k]['MAE'] for k in ARMS}, nmae),\n"
        "              ('R2', {k: m[k]['R2'] for k in ARMS}, None),\n"
        "              ('spec_accuracy', {k: m[k]['spec_accuracy'] for k in ARMS}, None)]\n"
        "    fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n"
        "    fig.suptitle(f'{t}  (ctx={CTX[t]}, full fine-tuning, n={len(d[\"predlen4\"])})',\n"
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
        "    a, b = res[(t, 'predlen1')], res[(t, 'predlen4')]\n"
        "    sa, sb = normalized_skill_score(a), normalized_skill_score(b)\n"
        "    ma = compute_metrics(a, cfg.SPEC_RANGES[t])['MAE']\n"
        "    mb = compute_metrics(b, cfg.SPEC_RANGES[t])['MAE']\n"
        "    win = 'predlen=4' if sb < sa else 'predlen=1'\n"
        "    print(f'{t:8s}: predlen1 skill={sa:.4f} MAE={ma:.4f}  |  predlen4 skill={sb:.4f} MAE={mb:.4f}  -> {win} 우세 '\n"
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
