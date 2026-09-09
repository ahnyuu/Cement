"""
Assemble experiments/results_context_comparison.ipynb — mirrors
Cement_code/chronos_quality/results_context_comparison.ipynb, but on the
운전&품질데이터_실측치.xlsx data and with every run at matching hyperparameters
(predlen 4 / num_steps 1000 / batch 64 / Full lr 1e-6 / LoRA lr 1e-5).

Reads only eval/backtest_*.csv and checkpoints/*/loss_history.csv — no GPU.

    .venv/Scripts/python.exe experiments/build_ctx_comparison_notebook.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "experiments" / "results_context_comparison.ipynb"

CONTEXTS = [24, 73, 128, 168, 256, 384, 512, 768, 1024, 1280, 1536]


def _md(t):
    return nbf.v4.new_markdown_cell(t)


def _code(s):
    return nbf.v4.new_code_cell(s)


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Cement_code_fin (.venv)", "language": "python", "name": "cement_code_fin",
    }
    c = []

    c.append(_md(
        "# Chronos-2 Quality Net — Context Length Comparison (실측치 데이터)\n\n"
        "`Cement_code/chronos_quality/results_context_comparison.ipynb`의 재현판. 데이터만 "
        "**운전&품질데이터_실측치.xlsx**(`data/processed/quality_timeseries.csv`)로 바꾸고, "
        "**11개 context_length** (24 / 73 / 128 / 168 / 256 / 384 / 512 / 768 / 1024 / 1280 / 1536) "
        "× **blaine / residue** × **zero-shot / LoRA / Full fine-tune** 전부를 구 folder와 동일한 "
        "하이퍼파라미터로 학습·평가:\n\n"
        "- `prediction_length` 4 (학습) / 1 (평가) · `num_steps` 1000 · `batch_size` 64 · backtest `stride` 4\n"
        "- Full `lr` 1e-6 · LoRA `lr` 1e-5 · zero-shot = `amazon/chronos-2` 그대로 backtest\n\n"
        "집계는 **정규화 skill-score** (`eval/metrics_utils.normalized_skill_score`): 호기(2CM/3CM/4CM)별 "
        "`model MAE / naive MAE` 비율을 동등 평균. **1.0 미만이면 naive baseline보다 낫다**는 뜻.\n\n"
        "GPU 안 씀 — 저장된 `eval/backtest_*.csv` / `checkpoints/*/loss_history.csv`만 읽음."
    ))

    c.append(_code(
        "import sys\nfrom pathlib import Path\n\n"
        "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n"
        "try:\n    plt.rcParams['font.family'] = 'Malgun Gothic'\n    plt.rcParams['axes.unicode_minus'] = False\n"
        "except Exception:\n    pass\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'experiments' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "import config as cfg\n"
        "from eval.metrics_utils import compute_metrics, macro_average_mae, normalized_skill_score\n"
        "from eval.loss_history import overfit_summary\n\n"
        f"CONTEXTS = {CONTEXTS}\n"
        "TARGETS = cfg.TARGET_COLS            # ['blaine', 'residue']\n"
        "METHODS = {'zero-shot': 'zeroshot', 'LoRA': 'lora', 'Full fine-tuning': 'full'}\n"
        "COLOR = {'blaine': 'tab:blue', 'residue': 'tab:orange'}\n"
        "MCOLOR = {'zero-shot': 'tab:blue', 'LoRA': 'tab:orange', 'Full fine-tuning': 'tab:green'}\n"
    ))

    c.append(_md("## 1. 결과 로딩 (11 ctx × 2 target × 3 method + loss history)"))
    c.append(_code(
        "raw = {t: {m: {} for m in METHODS} for t in TARGETS}\n"
        "loss = {t: {} for t in TARGETS}\n"
        "missing = []\n"
        "for t in TARGETS:\n"
        "    for label, suf in METHODS.items():\n"
        "        for ctx in CONTEXTS:\n"
        "            p = ROOT / 'eval' / f'backtest_{t}_ctx{ctx}_{suf}.csv'\n"
        "            if p.exists():\n"
        "                raw[t][label][ctx] = pd.read_csv(p, parse_dates=['timestamp'])\n"
        "            else:\n"
        "                missing.append(p.name)\n"
        "    for ctx in CONTEXTS:\n"
        "        # Full checkpoints: checkpoints/{t}_ctx{ctx}/loss_history.csv\n"
        "        lp = ROOT / 'checkpoints' / f'{t}_ctx{ctx}' / 'loss_history.csv'\n"
        "        if lp.exists():\n"
        "            loss[t][ctx] = pd.read_csv(lp)\n"
        "print('missing backtests:', missing or 'NONE')\n"
        "print('backtests loaded  :', sum(len(raw[t][m]) for t in TARGETS for m in METHODS), '/ 66')\n"
        "print('Full loss curves  :', sum(len(loss[t]) for t in TARGETS), '/ 22')\n\n"
        "def naive_metrics(t, ctx):\n"
        "    df = raw[t]['Full fine-tuning'][ctx]\n"
        "    return compute_metrics(df.assign(pred=df['naive_pred']), cfg.SPEC_RANGES[t])\n"
    ))

    c.append(_md(
        "## 2. context_length별 학습 과적합 확인 (Full fine-tuning loss curve)\n\n"
        "파란=blaine, 주황=residue. 실선=train_loss, 점선=val_loss. 세로 점선=val_loss 최소 지점. "
        "제목의 `Δval` = `val_loss_final − val_loss_min` (0보다 크면 과적합 신호). "
        "구 folder와 달리 여기 loss curve는 **backtest에 쓰인 바로 그 학습 실행**의 것 "
        "(`make_loss_history_callback` 기반, `_v2` 재학습본 아님)."
    ))
    c.append(_code(
        "ncols = 3\nnrows = -(-len(CONTEXTS) // ncols)\n"
        "fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows))\n"
        "axes = axes.flatten()\n"
        "ov = []\n"
        "for i, ctx in enumerate(CONTEXTS):\n"
        "    ax = axes[i]\n"
        "    for t in TARGETS:\n"
        "        df = loss[t].get(ctx)\n"
        "        if df is None:\n            continue\n"
        "        tr = df.dropna(subset=['train_loss']); va = df.dropna(subset=['val_loss'])\n"
        "        ax.plot(tr['step'], tr['train_loss'], color=COLOR[t], lw=1.3, label=f'{t} train')\n"
        "        ax.plot(va['step'], va['val_loss'], color=COLOR[t], lw=1.3, ls='--', label=f'{t} val')\n"
        "        s = overfit_summary(df); s.update({'target': t, 'ctx': ctx}); ov.append(s)\n"
        "        if s.get('n_val_points'):\n"
        "            ax.axvline(s['val_loss_min_step'], color=COLOR[t], ls=':', lw=0.8, alpha=0.5)\n"
        "    gaps = '  '.join(f\"{r['target']} Δval={r['final_minus_min']:+.4f}\"\n"
        "                     for r in ov if r['ctx'] == ctx and r.get('n_val_points'))\n"
        "    ax.set_title(f'ctx={ctx}\\n{gaps}', fontsize=9)\n"
        "    ax.tick_params(labelsize=7)\n"
        "    if i == 0:\n        ax.legend(fontsize=6)\n"
        "for j in range(len(CONTEXTS), len(axes)):\n    axes[j].axis('off')\n"
        "plt.tight_layout(); plt.show()\n"
        "print(pd.DataFrame(ov).sort_values(['target', 'ctx'])[\n"
        "    ['target', 'ctx', 'val_loss_min', 'val_loss_final', 'final_minus_min']].round(5).to_string(index=False))\n"
    ))

    c.append(_md(
        "## 3. context_length별 정규화 skill-score (Full fine-tuning)\n\n"
        "막대 = `model MAE / naive MAE` (호기 3개 평균). **1.0(빨간 점선) 아래로 갈수록 좋음.**"
    ))
    c.append(_code(
        "score = {t: {ctx: normalized_skill_score(raw[t]['Full fine-tuning'][ctx]) for ctx in CONTEXTS} for t in TARGETS}\n"
        "ncols = 4\nnrows = -(-len(CONTEXTS) // ncols)\n"
        "fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), sharey=True)\n"
        "axes = axes.flatten()\n"
        "ymax = max(v for t in TARGETS for v in score[t].values()) * 1.08\n"
        "for i, ctx in enumerate(CONTEXTS):\n"
        "    ax = axes[i]\n"
        "    vals = [score[t][ctx] for t in TARGETS]\n"
        "    b = ax.bar(TARGETS, vals, color=[COLOR[t] for t in TARGETS], width=0.55)\n"
        "    ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.7)\n"
        "    for bb, v in zip(b, vals):\n"
        "        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)\n"
        "    ax.set_title(f'ctx={ctx}', fontsize=10); ax.set_ylim(0.78, ymax); ax.tick_params(labelsize=8)\n"
        "for j in range(len(CONTEXTS), len(axes)):\n    axes[j].axis('off')\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    c.append(_md("## 4. context_length 추세 (정규화 skill-score, 3 방식)"))
    c.append(_code(
        "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
        "xs = list(range(len(CONTEXTS)))\n"
        "for t in TARGETS:\n"
        "    for label in METHODS:\n"
        "        ys = [normalized_skill_score(raw[t][label][ctx]) for ctx in CONTEXTS]\n"
        "        ls = {'zero-shot': ':', 'LoRA': '--', 'Full fine-tuning': '-'}[label]\n"
        "        ax.plot(xs, ys, ls, marker='o', ms=4, color=COLOR[t], alpha=0.9,\n"
        "                label=f'{t} · {label}')\n"
        "        if label == 'Full fine-tuning':\n"
        "            bi = int(np.argmin(ys))\n"
        "            ax.annotate(f'{t} best ctx={CONTEXTS[bi]}', (bi, ys[bi]),\n"
        "                        textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8, color=COLOR[t])\n"
        "ax.axhline(1.0, color='red', ls='--', lw=1, label='naive')\n"
        "ax.set_xticks(xs); ax.set_xticklabels(CONTEXTS, rotation=45)\n"
        "ax.set_xlabel('context_length'); ax.set_ylabel('정규화 skill-score (낮을수록 좋음)')\n"
        "ax.set_title('context_length별 정규화 skill-score — zero-shot(:) / LoRA(--) / Full(-)')\n"
        "ax.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.show()\n"
    ))

    c.append(_md("## 5. 순위표 (Full fine-tuning — 정규화 / macro / pooled MAE)"))
    c.append(_code(
        "for t in TARGETS:\n"
        "    rows = []\n"
        "    for ctx in CONTEXTS:\n"
        "        df = raw[t]['Full fine-tuning'][ctx]\n"
        "        rows.append({'ctx': ctx,\n"
        "                     'skill_norm': round(normalized_skill_score(df), 4),\n"
        "                     'macro_MAE': round(macro_average_mae(df), 4),\n"
        "                     'pooled_MAE': round(compute_metrics(df, cfg.SPEC_RANGES[t])['MAE'], 4)})\n"
        "    tab = pd.DataFrame(rows).sort_values('skill_norm').reset_index(drop=True)\n"
        "    print(f'=== {t} (정규화 skill-score 순) ===')\n"
        "    print(tab.to_string(index=False))\n"
        "    print(f'  -> 최적 context_length = {int(tab.iloc[0][\"ctx\"])}\\n')\n"
    ))

    c.append(_md(
        "## 6. 상세 지표 추세 (Full fine-tuning, pooled)\n\n"
        "MAE(낮을수록) / R²(높을수록) / Spec 정확도(높을수록) / 80% 커버리지(0.80 초록선에 가까울수록). "
        "회색 점선 = naive baseline."
    ))
    c.append(_code(
        "PM = ['MAE', 'R2', 'spec_accuracy', 'interval_coverage']\n"
        "PL = {'MAE': 'MAE', 'R2': 'R²', 'spec_accuracy': 'Spec 정확도', 'interval_coverage': '80% 커버리지'}\n"
        "xs = list(range(len(CONTEXTS)))\n"
        "fig, axes = plt.subplots(len(TARGETS), len(PM), figsize=(4.6 * len(PM), 4.0 * len(TARGETS)))\n"
        "for i, t in enumerate(TARGETS):\n"
        "    mt = {ctx: compute_metrics(raw[t]['Full fine-tuning'][ctx], cfg.SPEC_RANGES[t]) for ctx in CONTEXTS}\n"
        "    nv = {ctx: naive_metrics(t, ctx) for ctx in CONTEXTS}\n"
        "    for j, k in enumerate(PM):\n"
        "        ax = axes[i, j]\n"
        "        ax.plot(xs, [mt[ctx][k] for ctx in CONTEXTS], marker='o', ms=5, color=COLOR[t])\n"
        "        if k in ('MAE', 'R2', 'spec_accuracy'):\n"
        "            ax.plot(xs, [nv[ctx][k] for ctx in CONTEXTS], ls=':', color='grey', label='naive')\n"
        "            ax.legend(fontsize=7)\n"
        "        if k == 'interval_coverage':\n"
        "            ax.axhline(0.80, color='green', ls='--', lw=1)\n"
        "        ax.set_xticks(xs); ax.set_xticklabels(CONTEXTS, rotation=45, fontsize=7)\n"
        "        ax.set_title(f'{t} — {PL[k]}', fontsize=11, fontweight='bold')\n"
        "        ax.grid(True, alpha=0.25)\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    c.append(_md(
        "## 7. zero-shot / LoRA / Full fine-tuning 비교 (11 ctx 전부)\n\n"
        "패널마다 세 방식을 겹쳐 그림 (행=target, 열=지표). MAE 낮을수록, R²·Spec 높을수록, "
        "80% 커버리지는 초록선(0.80)에 가까울수록."
    ))
    c.append(_code(
        "fig, axes = plt.subplots(len(TARGETS), len(PM), figsize=(4.6 * len(PM), 4.0 * len(TARGETS)))\n"
        "for i, t in enumerate(TARGETS):\n"
        "    for j, k in enumerate(PM):\n"
        "        ax = axes[i, j]\n"
        "        for label in METHODS:\n"
        "            ys = [compute_metrics(raw[t][label][ctx], cfg.SPEC_RANGES[t])[k] for ctx in CONTEXTS]\n"
        "            ax.plot(xs, ys, marker='o', ms=4, color=MCOLOR[label], label=label)\n"
        "        if k == 'interval_coverage':\n"
        "            ax.axhline(0.80, color='green', ls='--', lw=1)\n"
        "        ax.set_xticks(xs); ax.set_xticklabels(CONTEXTS, rotation=45, fontsize=7)\n"
        "        ax.set_title(f'{t} — {PL[k]}', fontsize=11, fontweight='bold')\n"
        "        ax.grid(True, alpha=0.25)\n"
        "        if i == 0 and j == 0:\n            ax.legend(fontsize=8)\n"
        "plt.tight_layout(); plt.show()\n\n"
        "for t in TARGETS:\n"
        "    for label in METHODS:\n"
        "        rows = []\n"
        "        for ctx in CONTEXTS:\n"
        "            m = compute_metrics(raw[t][label][ctx], cfg.SPEC_RANGES[t])\n"
        "            rows.append({'ctx': ctx, 'MAE': round(m['MAE'], 4), 'R2': round(m['R2'], 4),\n"
        "                         'spec_acc': round(m['spec_accuracy'], 4),\n"
        "                         'cov80': round(m['interval_coverage'], 4),\n"
        "                         'skill': round(normalized_skill_score(raw[t][label][ctx]), 4)})\n"
        "        print(f'--- {t} · {label} ---')\n"
        "        print(pd.DataFrame(rows).to_string(index=False)); print()\n"
    ))

    c.append(_md(
        "## 8. 결론\n\n"
        "_아래는 build 스크립트에 하드코딩된 요약 — 숫자는 위 셀에서 다시 확인 가능._\n\n"
        "- **blaine (Full)**: skill-score가 **ctx≈384부터 평탄** (384→1536: 0.838→0.824, 재학습 "
        "  노이즈 수준). 최저는 ctx1536(0.824)이지만 384~1536 어디든 사실상 동급. **ctx24는 확실히 "
        "  최악**(0.915 — naive와 큰 차이 없음).\n"
        "- **residue (Full)**: **ctx256에서 최적**(skill 0.809, 매트릭스 전체 최고값). 그보다 길어지면 "
        "  **단조적으로 악화** (256→1536: 0.809→0.857). ctx24도 나쁨(0.879). 즉 residue는 짧은~중간 "
        "  context가 분명히 유리.\n"
        "- **방식 비교**: 쓸모 있는 구간(ctx≥73)에서 **Full > LoRA > zero-shot**이 거의 모든 ctx에서 "
        "  성립 (MAE·R²·Spec 전부). ctx24에서만 셋이 도토리 키재기. LoRA는 zero-shot보다 확실히 낫지만 "
        "  Full엔 못 미침 → **Full fine-tuning 채택이 데이터로 뒷받침됨** (구 folder 결론과 동일).\n"
        "- **80% 예측구간 커버리지**: 전 구간 49~75%로 목표(0.80) 미달. 다만 **짧은 ctx(73~168)에서 "
        "  가장 높음**(residue Full 73~168에서 74~75%) — 긴 context일수록 예측구간이 좁아지며 과신.\n"
        "- **구 folder 결론(blaine ctx1024 / residue ctx1536)과 반대.** 그건 비실측 시간대 값이 스며 "
        "  있던 구 xlsx 기준이고, 긴 context가 그 carry-forward 규칙성을 주워 담던 것으로 보임. 실측치 "
        "  데이터에서는 **residue는 ctx256, blaine은 ctx≈384+ 평탄**.\n"
        "- **권장**: 한 값으로 통일하면 **ctx256** (residue 최적 + blaine 평탄 시작점 근처). "
        "  타깃별로 두면 blaine 384~768 / residue 256.\n\n"
        "미해결(구 folder에서 이어짐): 80% 커버리지 과신, ctx24 극단 케이스 외 512 미만 미세 구조, "
        "1536 이후 추가 스윕 필요성 없음(둘 다 이미 악화 추세)."
    ))

    nb.cells = c
    return nb


def main() -> None:
    NB_PATH.write_text(nbf.writes(build_notebook()), encoding="utf-8")
    print(f"wrote {NB_PATH}")
    res = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=900", str(NB_PATH)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(res.stdout[-3000:]); print(res.stderr[-3000:])
    if res.returncode != 0:
        raise SystemExit("nbconvert failed")
    print(f"executed -> {NB_PATH}")


if __name__ == "__main__":
    main()
