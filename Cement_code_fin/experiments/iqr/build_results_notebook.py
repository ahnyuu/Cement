"""
Assemble experiments/iqr/results_iqr_comparison.ipynb from the 8 IQR-A/B backtests, then
execute it. Mirrors Cement_code/chronos_quality/01_iqr_outlier_removal/results_iqr_comparison.ipynb
(the dual raw/clean-test-set version).

    .venv/Scripts/python.exe experiments/iqr/build_results_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "experiments" / "iqr" / "results_iqr_comparison.ipynb"


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
        "# IQR 이상치 제거 유무 A/B — 실측치 데이터\n\n"
        "`Cement_code/chronos_quality/01_iqr_outlier_removal/`의 재현판. 데이터는 "
        "**운전&품질데이터_실측치.xlsx**, context_length는 2026-09-06 스윕 최적값 "
        "(**blaine 512 / residue 256**).\n\n"
        "**IQR 제거**: `clean()`의 물리 규칙을 통과한 뒤, 호기별·컬럼별 Tukey 1.5×IQR 펜스 밖 값을 "
        "NaN 처리 (`build_dataset.iqr_cols()` = `RP_proc_time` 제외한 전 raw feature + blaine/residue). "
        "실측치 데이터에서 총 **115,210개** 값이 NaN 처리됨 (최다: feed_slag 29k — SLAG 미투입 배치가 "
        "통계적으로 이상치로 잡힘).\n\n"
        "**설계**: 4 모델 = {blaine, residue} × {withiqr 학습, withoutiqr 학습}, full fine-tune / "
        "predlen 4 / 1000 steps / lr 1e-6 / batch 64. 각 모델을 **두 테스트셋 모두**에서 backtest:\n"
        "- `evalon_raw` = without_iqr 테스트셋 (극단값 포함, 실측 그대로) = **배포 현실, 주지표**\n"
        "- `evalon_clean` = with_iqr 테스트셋 (IQR로 극단값 제거된 정상운전 구간)\n\n"
        "같은 테스트셋에서 두 모델을 평가해야 naive baseline / skill-score 분모가 동일 → 공정 비교.\n\n"
        "정규화 skill-score < 1 이면 naive(직전값 반복) baseline보다 낫다는 뜻. GPU 안 씀."
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
        "TRAINVARS = ['withoutiqr', 'withiqr']\n"
        "EVALVARS = ['raw', 'clean']\n"
        "TRAIN_LABEL = {'withoutiqr': 'IQR 미적용', 'withiqr': 'IQR 적용'}\n"
        "COLOR = {'withoutiqr': 'tab:blue', 'withiqr': 'tab:orange'}\n"
    ))

    c.append(_md("## 1. 결과 로딩 (4 모델 × 2 테스트셋 = 8 backtest)"))
    c.append(_code(
        "res = {}\nmissing = []\n"
        "for t in cfg.TARGET_COLS:\n"
        "    ctx = CTX[t]\n"
        "    for tv in TRAINVARS:\n"
        "        for ev in EVALVARS:\n"
        "            p = ROOT / 'eval' / f'backtest_{t}_ctx{ctx}_{tv}_evalon_{ev}.csv'\n"
        "            if p.exists():\n"
        "                res[(t, tv, ev)] = pd.read_csv(p, parse_dates=['timestamp'])\n"
        "            else:\n"
        "                missing.append(p.name)\n"
        "print('missing:', missing or 'NONE')\n"
        "print('loaded :', len(res), '/ 8')\n"
    ))

    c.append(_md(
        "## 2. 요약표\n\n"
        "행 = (타깃, 학습 데이터, 평가 테스트셋). `evalon_raw`가 주지표. 같은 (타깃, 평가셋) 안에서 "
        "`withoutiqr` vs `withiqr` 를 비교하면 됨."
    ))
    c.append(_code(
        "rows = []\n"
        "for t in cfg.TARGET_COLS:\n"
        "    for ev in EVALVARS:\n"
        "        for tv in TRAINVARS:\n"
        "            df = res.get((t, tv, ev))\n"
        "            if df is None:\n                continue\n"
        "            m = compute_metrics(df, cfg.SPEC_RANGES[t])\n"
        "            nmae = float(np.abs(df['actual'] - df['naive_pred']).mean())\n"
        "            rows.append({'target': t, 'eval_set': ev, 'train_data': tv, 'n': len(df),\n"
        "                         'MAE': round(m['MAE'], 4), 'R2': round(m['R2'], 4),\n"
        "                         'spec%': round(m['spec_accuracy'] * 100, 2),\n"
        "                         'cov80%': round(m['interval_coverage'] * 100, 2),\n"
        "                         'macro_MAE': round(macro_average_mae(df), 4),\n"
        "                         'skill': round(normalized_skill_score(df), 4),\n"
        "                         'naive_MAE': round(nmae, 4)})\n"
        "summary = pd.DataFrame(rows).set_index(['target', 'eval_set', 'train_data'])\n"
        "summary\n"
    ))

    c.append(_md("## 3. Δ (IQR 적용 − IQR 미적용), 테스트셋별\n\n음수 = IQR 적용이 나쁨 (MAE/skill 기준)."))
    c.append(_code(
        "for ev in EVALVARS:\n"
        "    print(f'=== evalon_{ev} ===')\n"
        "    for t in cfg.TARGET_COLS:\n"
        "        a = res.get((t, 'withoutiqr', ev)); b = res.get((t, 'withiqr', ev))\n"
        "        if a is None or b is None:\n            continue\n"
        "        ma, mb = compute_metrics(a, cfg.SPEC_RANGES[t]), compute_metrics(b, cfg.SPEC_RANGES[t])\n"
        "        sa, sb = normalized_skill_score(a), normalized_skill_score(b)\n"
        "        print(f'  {t:8s}  ΔMAE={mb[\"MAE\"]-ma[\"MAE\"]:+.4f}  ΔR2={mb[\"R2\"]-ma[\"R2\"]:+.4f}  '\n"
        "              f'Δskill={sb-sa:+.4f}  (withoutiqr skill={sa:.4f}, withiqr skill={sb:.4f})')\n"
        "    print()\n"
    ))

    c.append(_md(
        "## 4. 타깃별 비교 그래프 (주지표 = evalon_raw)\n\n"
        "패널 = Skill-score / MAE / R² / Spec 정확도. 파랑 = IQR 미적용, 주황 = IQR 적용. "
        "MAE 패널 회색 점선 = naive baseline."
    ))
    c.append(_code(
        "def plot_target(t, ev='raw'):\n"
        "    d = {tv: res[(t, tv, ev)] for tv in TRAINVARS}\n"
        "    m = {tv: compute_metrics(d[tv], cfg.SPEC_RANGES[t]) for tv in TRAINVARS}\n"
        "    sk = {tv: normalized_skill_score(d[tv]) for tv in TRAINVARS}\n"
        "    nmae = float(np.abs(d['withoutiqr']['actual'] - d['withoutiqr']['naive_pred']).mean())\n"
        "    panels = [('skill-score', sk, None), ('MAE', {k: m[k]['MAE'] for k in TRAINVARS}, nmae),\n"
        "              ('R2', {k: m[k]['R2'] for k in TRAINVARS}, None),\n"
        "              ('spec_accuracy', {k: m[k]['spec_accuracy'] for k in TRAINVARS}, None)]\n"
        "    fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n"
        "    fig.suptitle(f'{t}  (ctx={CTX[t]}, full fine-tuning, evalon_{ev}, n={len(d[\"withoutiqr\"])})',\n"
        "                 fontsize=13, fontweight='bold', y=1.05)\n"
        "    for ax, (name, vals, naive) in zip(axes, panels):\n"
        "        xs = np.arange(len(TRAINVARS))\n"
        "        bars = ax.bar(xs, [vals[tv] for tv in TRAINVARS], color=[COLOR[tv] for tv in TRAINVARS], width=0.55)\n"
        "        for bb, tv in zip(bars, TRAINVARS):\n"
        "            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), f'{vals[tv]:.4f}',\n"
        "                    ha='center', va='bottom', fontsize=9)\n"
        "        if naive is not None:\n"
        "            ax.axhline(naive, color='grey', ls=':', label=f'naive {naive:.3f}'); ax.legend(fontsize=8)\n"
        "        if name == 'skill-score':\n"
        "            ax.axhline(1.0, color='red', ls='--', lw=1)\n"
        "        ax.set_xticks(xs); ax.set_xticklabels([TRAIN_LABEL[tv] for tv in TRAINVARS])\n"
        "        ax.set_title(name, fontsize=11, fontweight='bold')\n"
        "    plt.tight_layout(); plt.show()\n\n"
        "for t in cfg.TARGET_COLS:\n"
        "    plot_target(t, 'raw')\n"
    ))

    c.append(_md("## 5. 결론\n\n_아래는 build 스크립트 하드코딩 요약 — 숫자는 위 셀에서 재확인._"))
    c.append(_code(
        "for t in cfg.TARGET_COLS:\n"
        "    print(f'--- {t} (ctx={CTX[t]}) ---')\n"
        "    for ev in EVALVARS:\n"
        "        a, b = res[(t, 'withoutiqr', ev)], res[(t, 'withiqr', ev)]\n"
        "        sa, sb = normalized_skill_score(a), normalized_skill_score(b)\n"
        "        ma, mb = compute_metrics(a, cfg.SPEC_RANGES[t])['MAE'], compute_metrics(b, cfg.SPEC_RANGES[t])['MAE']\n"
        "        win = 'IQR 미적용' if sa < sb else 'IQR 적용'\n"
        "        print(f'  evalon_{ev:5s}: withoutiqr skill={sa:.4f} MAE={ma:.4f} | withiqr skill={sb:.4f} MAE={mb:.4f}  -> {win} 우세')\n"
        "    print()\n"
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
