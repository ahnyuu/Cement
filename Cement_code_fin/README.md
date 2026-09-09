# Cement Quality Net (Chronos-2) — `_fin`


> **2026-09-09 검증·평가 수정본:** 실행 전 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)를 읽어 주세요.
> 새 학습은 여러 날짜의 정답으로 검증하며, 실험 실행기는 validation 구간만 평가합니다.
> 새 모델은 `checkpoints/review_v2/`, 새 평가 결과는 `eval/review_v2/validation/`에 저장됩니다.
> 아래 기존 결과 표·실험 로그·노트북은 과거 test 기반 탐색 기록이며 수정 후 결과가 아닙니다.
> 코드 변경만으로 과거 test 구간이 미사용 데이터가 되지는 않습니다.

Cleaned-up rebuild of `Cement_code/chronos_quality/`. Fine-tunes **Chronos-2** as a per-target
quality net (blaine, residue) for the cement grinding process. Evaluation reports Chronos-2
and a last-observation baseline; it does not perform a matched ANN comparison.

Only the canonical pipeline is kept here. The old folder's A/B experiment machinery (statistical
IQR removal, damper clip-vs-NaN, ProdType drop-vs-mask, prev-quality sparse-vs-ffill, dozens of
checkpoint dirs, 100+ backtest CSVs) is gone — add a branch back only when an experiment needs it.

## What carried over as settled (see `Cement_code/chronos_quality/EXPERIMENT_LOG.md`)

| decision | value |
|---|---|
| source xlsx | `운전&품질데이터_실측치.xlsx` (measured-actual only; swapping to it made every zero-shot/LoRA/full combo beat naive) |
| statistical IQR outlier removal | **off** (only fixed physical/equipment limits in `config.CLEANING_RULES`) |
| `blaine_prev` / `residue_prev` covariate | **sparse** (present only at scheduled 4-hour measurement rows) |
| `prediction_length` (train) | 4 |
| `prediction_length` (eval) | 1 |
| `finetune_mode` | full |
| `num_steps` / `learning_rate` | 1000 / 1e-6 |
| primary metric | normalized skill score (`eval/metrics_utils.py`) |
| `context_length` | full study 2026-09-06 (11 values 24–1536 × 3 methods): **residue best at 256**, blaine flat from ~384; more context hurts both zero-shot and residue. Recommend **256**; script default still 512 pending the switch (EXPERIMENT_LOG next-steps #1). |

## Layout

```
config.py                     column mapping, control/monitor split, spec bands, cleaning rules
data/build_dataset.py         raw xlsx -> data/processed/quality_timeseries.csv (single pipeline)
data/dataset_utils.py         load_processed() + chronological_split()
data/processed/quality_timeseries.csv   pre-built; travels with the folder
train/finetune_chronos2.py    fine-tune one target
eval/backtest.py              rolling-origin 1-step backtest + naive baseline + skill score
eval/metrics_utils.py         macro-avg MAE, normalized skill score
eval/loss_history.py          live train/val loss capture (overfit check)
```

`checkpoints/` and `eval/backtest_*.csv` are created by the scripts.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

GPU/CPU is auto-detected (`config.default_device_map()`).

## Run

```bash
# 1) (only if regenerating the processed CSV from the raw xlsx)
python data/build_dataset.py

# 2) fine-tune, per target — sweep --context-length
python train/finetune_chronos2.py --target blaine  --context-length 1024 --output-tag ctx1024
python train/finetune_chronos2.py --target residue --context-length 1536 --output-tag ctx1536

# 3) backtest (auto-compares against the naive baseline)
python eval/backtest.py --target blaine  --checkpoint checkpoints/blaine_ctx1024/final  --tag ctx1024
python eval/backtest.py --target residue --checkpoint checkpoints/residue_ctx1536/final --tag ctx1536

# zero-shot baseline for reference
python eval/backtest.py --target blaine --checkpoint amazon/chronos-2 --tag zeroshot
```

Chronos-2 (`amazon/chronos-2`, 120M) downloads from HuggingFace on first run (set `HF_TOKEN` to
lift the rate limit).

To try a different processed CSV without touching the canonical one, set `CHRONOS_PROCESSED_CSV`
and pass matching `--output-tag` / `--tag`.
