# Cement Quality Net (`_fin`) — Experiment Log

New log for the cleaned-up rebuild. The full history of the earlier pipeline (2026-08-04 →
2026-09-05, ~15 experiments: first LoRA finetune, column rename, prev covariate, ProdType bug,
num_steps sweep, full finetune, context_length sweeps, IQR on/off A/B, physical-outlier A/B,
prev-density sparse/ffill A/B, measured-actual xlsx swap + controlled comparison) lives in
`../Cement_code/chronos_quality/EXPERIMENT_LOG.md`.

## Settled going in (from that log)

- **Source xlsx = `운전&품질데이터_실측치.xlsx`.** Controlled comparison (2026-09-05): pipeline
  fixed, xlsx the only change → skill-score dropped 0.07–0.19 across all 6 zero-shot/LoRA/full ×
  blaine/residue combos; with the old xlsx only blaine-full beat naive, with the measured-actual
  xlsx all 6 do. The old file had non-measurement-hour values bleeding in, inflating the naive
  baseline.
- **No statistical IQR removal** (2026-08-28, re-confirmed 2026-08-31). Chronos-2 masks NaN
  natively; deleting extreme-but-real values is a net loss of signal. Fixed physical limits only.
- **prediction_length = 4 for training** (2026-08-18). Targets sit on a 4-hour grid, so predlen≥4
  guarantees each sampled training window contains a real measurement. Eval stays 1-step.
- **finetune_mode = full** (> LoRA in every sweep), num_steps 1000, lr 1e-6.
- **prev-quality covariate = sparse** is the current default. NOTE: the last A/B on the *old*
  xlsx (2026-09-02) found **ffill** meaningfully better (Δskill ≈ −0.023 blaine / −0.026 residue,
  ~6× retrain noise) but it was never promoted to default and never re-checked on the
  measured-actual data. Re-verifying ffill here is an open item.

## Open items inherited

1. `context_length` — the whole sweep needs redoing on the measured-actual data (old bests were
   blaine 1024 / residue 1536; non-monotonic dip near ctx≈128 was never explained).
2. prev-density sparse vs ffill — re-verify on this dataset.
3. 80% prediction-interval coverage sat at 0.45–0.68 (target 0.80) — uncertainty still overconfident.
4. Ambiguous physical tails (mill_out gas/mater temp 1100–1400 cluster, blaine own min/max) left
   in pending plant-engineer confirmation.

---

## 2026-09-06 — `_fin` skeleton built

Rebuilt `chronos_quality/` as `Cement_code_fin/` with only the canonical path. `config.py` points
at `운전&품질데이터_실측치.xlsx`; `data/build_dataset.py` collapsed to a single `build()` (no
`damper_oob` / `prodtype_mode` / `prev_density` / IQR branches); `finetune_chronos2.py` defaults
updated (predlen 4, full, 1000 steps, lr 1e-6); `backtest.py` now prints normalized skill score
directly. `data/processed/quality_timeseries.csv` pre-built and included.

Dedicated venv built (`.venv`, Python 3.13, torch 2.11+cu128, RTX 5060 8GB) + Jupyter kernel
`cement_code_fin`. `Cement_code/` to be deleted — this folder is now standalone.

---

## 2026-09-06 — context_length sweep {512, 768, 1024, 1280, 1536} × {blaine, residue}

First real experiment on `_fin`. Full fine-tune (predlen 4 train / 1-step eval, 1000 steps,
lr 1e-6, batch 64), every ctx also evaluated zero-shot (ctx changes the inference input, so
zero-shot isn't reusable across ctx). Orchestrated by `experiments/run_ctx_sweep.py`; results
notebook `experiments/results_ctx_sweep.ipynb`; per-run log `experiments/ctx_sweep.log`.
n ≈ 851 (blaine) / 856 (residue) fresh-reading eval points, stride 4, same eval set across ctx
(naive baseline: blaine MAE 83.77 / R² 0.163 / spec 67.0%; residue MAE 0.569 / R² 0.578 / spec 78.5%).

**blaine — full fine-tune**

| ctx | MAE | R² | spec | 80% cov | skill |
|---|---|---|---|---|---|
| 512  | 69.63 | 0.474 | 70.2% | 66.2% | 0.832 |
| 768  | 69.29 | 0.480 | 70.8% | 63.0% | 0.828 |
| 1024 | 69.47 | 0.479 | 71.0% | 66.1% | 0.830 |
| 1280 | 69.54 | 0.477 | 71.4% | 63.0% | 0.831 |
| 1536 | 68.93 | 0.488 | 70.7% | 68.0% | 0.824 |

**residue — full fine-tune**

| ctx | MAE | R² | spec | 80% cov | skill |
|---|---|---|---|---|---|
| 512  | 0.467 | 0.761 | 82.6% | 68.5% | **0.825** |
| 768  | 0.471 | 0.760 | 81.5% | 70.4% | 0.833 |
| 1024 | 0.479 | 0.757 | 81.5% | 66.4% | 0.845 |
| 1280 | 0.481 | 0.755 | 81.0% | 67.2% | 0.848 |
| 1536 | 0.486 | 0.754 | 81.8% | 67.5% | 0.857 |

zero-shot (both targets) degrades monotonically with ctx: blaine skill 0.873 → 0.914,
residue 0.857 → 0.976 (512 → 1536).

### 결론 — **context_length = 512 for both targets**
- **blaine**: 512–1536 전부 평탄 (skill 0.824–0.832, 재학습 노이즈 ~0.004 안쪽). ctx1536이 MAE/R²
  기준 근소 우위지만 유의미하지 않음.
- **residue**: context가 길수록 **단조적으로 악화** (skill 0.825 → 0.857). ctx512가 명확히 최고.
- **zero-shot**: 두 타깃 모두 context가 길수록 단조 악화 → 실측치 데이터에서 긴 이력은 신호가
  아니라 노이즈.
- full fine-tune은 전 ctx에서 naive를 이김 (skill < 1). blaine ~0.83, residue 0.83–0.86.
- 80% 예측구간 커버리지는 전 ctx에서 49–68% (목표 80% 미달) — 기존 미해결 이슈 그대로.
- **기존 파이프라인 결론(blaine 1024 / residue 1536)과 반대.** 그건 비실측 시간대 값이 스며 있던
  구 xlsx 기준이었고(2026-09-05 통제 비교 참고), 실측치 데이터에서는 짧은 context가 유리.
- ctx512는 학습도 압도적으로 빠름 (2.6분 vs ctx1536 92분, 이 8GB GPU 기준).
- 학습 시간(이 머신): ctx 512/768/1024/1280/1536 ≈ 2.6 / 3.6 / 18 / 67 / 93 분. ctx1024부터
  8GB VRAM 한계로 급격히 느려짐.

### 다음에 해볼 것
1. **512 미만 스윕 (256/384)** — ≥512 구간이 평탄~악화라 최적점이 더 짧을 수 있음.
2. `finetune_chronos2.py` / `README` 기본 `--context-length`를 512로 확정 (현재도 512).
3. prev-density sparse vs ffill 재확인 (실측치 데이터 기준, ctx512 고정).
4. 80% 커버리지 (conformal / quantile-loss 가중).

체크포인트: `checkpoints/{blaine,residue}_ctx{512,768,1024,1280,1536}/`.
backtest: `eval/backtest_{blaine,residue}_ctx{N}_{zeroshot,full}.csv`.

---

## 2026-09-06 (이어서) — 전체 context_length 파라미터 스터디 재현 (구 folder와 동일 매트릭스)

사용자 요청: `Cement_code/chronos_quality/results_context_comparison.ipynb`의 파라미터 스터디를
**데이터만 실측치로 바꿔** 동일하게 재현. 매트릭스: **ctx {24, 73, 128, 168, 256, 384, 512, 768,
1024, 1280, 1536} × {blaine, residue} × {zero-shot, LoRA, Full}**. 하이퍼파라미터 구 folder와
동일 (predlen 4 학습 / 1 평가, num_steps 1000, batch 64, Full lr 1e-6, LoRA lr 1e-5, stride 4).
위 스윕의 Full/zero-shot 5개 ctx(512–1536) 재사용, 나머지(Full+zeroshot 6 ctx, LoRA 11 ctx) 추가.
오케스트레이터 `experiments/run_ctx_matrix.py`, 결과 노트북 `experiments/results_context_comparison.ipynb`
(구 folder 노트북 구조 미러: loss curve / skill-score / 상세지표 추세 / 순위표 / 방식비교 / 결론).
66개 backtest 전부 완료, 실패 0. 학습 시간: Full ctx24–384 각 ~2분 / ctx1024 18분 / ctx1280 67분 /
ctx1536 93분; **LoRA는 전 ctx 2–7분** (ctx1536 LoRA 7.3분 vs Full 93분 — LoRA가 긴 context에서
훨씬 빠름). 전체 매트릭스 wall-clock ~2시간.

### 정규화 skill-score (model MAE / naive MAE, 호기 평균; <1 = naive 이김)

**blaine** (naive MAE 83.77)

| ctx | zero-shot | LoRA | Full |
|---|---|---|---|
| 24   | 0.915 | 0.914 | 0.915 |
| 73   | 0.887 | 0.882 | 0.857 |
| 128  | 0.892 | 0.887 | 0.860 |
| 168  | 0.875 | 0.871 | 0.855 |
| 256  | 0.865 | 0.862 | 0.848 |
| 384  | 0.862 | 0.856 | **0.838** |
| 512  | 0.873 | 0.860 | 0.832 |
| 768  | 0.886 | 0.843 | 0.828 |
| 1024 | 0.903 | 0.852 | 0.830 |
| 1280 | 0.907 | 0.835 | 0.831 |
| 1536 | 0.914 | 0.831 | **0.824** |

**residue** (naive MAE 0.568)

| ctx | zero-shot | LoRA | Full |
|---|---|---|---|
| 24   | 0.892 | 0.886 | 0.879 |
| 73   | 0.861 | 0.855 | 0.855 |
| 128  | 0.858 | 0.854 | 0.852 |
| 168  | 0.857 | 0.844 | 0.838 |
| 256  | 0.847 | 0.834 | **0.809** |
| 384  | **0.844** | **0.832** | 0.815 |
| 512  | 0.857 | 0.840 | 0.825 |
| 768  | 0.879 | 0.841 | 0.833 |
| 1024 | 0.915 | 0.847 | 0.845 |
| 1280 | 0.944 | 0.856 | 0.848 |
| 1536 | 0.976 | 0.876 | 0.857 |

### 결론
- **blaine (Full)**: skill-score가 **ctx≈384부터 평탄** (384→1536: 0.838→0.824, 재학습 노이즈
  수준). 384~1536 어디든 사실상 동급. ctx24는 확실히 최악(0.915).
- **residue (Full)**: **ctx256이 명확한 최적** (0.809 — 매트릭스 전체 최고). 그보다 길면 **단조
  악화** (256→1536: 0.809→0.857). 짧은~중간 context가 유리.
- **방식**: ctx≥73에서 **Full > LoRA > zero-shot**이 거의 모든 ctx에서 성립. LoRA는 zero-shot보다
  낫지만 Full엔 못 미침 → Full 채택이 데이터로 뒷받침됨 (구 folder 결론과 동일). ctx24만 셋이 동급.
- **zero-shot은 두 타깃 다 ctx 길수록 단조 악화** (blaine 0.86→0.91, residue 0.84→0.98).
- **80% 커버리지**: 전 구간 49–75%로 목표 미달. 짧은 ctx(73–168)에서 가장 높음 — 긴 context일수록
  예측구간이 좁아지며 과신.
- **구 folder 결론(blaine 1024 / residue 1536)과 반대.** 그건 carry-forward 오염된 구 xlsx 기준.
  실측치에서는 residue ctx256, blaine ctx≈384+ 평탄.
- **권장**: 한 값 통일 시 **ctx256** (residue 최적 + blaine 평탄 시작 근처). 타깃별이면 blaine
  384–768 / residue 256.

### 다음에 해볼 것 (갱신)
1. `finetune_chronos2.py` / `README` 기본 `--context-length`를 256으로 변경 (현재 512).
2. residue ctx256 부근(192/224/288/320) 미세 스윕으로 최적점 정밀화 — 선택.
3. prev-density sparse vs ffill 재확인 (실측치, ctx256 고정).
4. 80% 커버리지 (conformal / quantile-loss 가중) — 미해결.

산출물: `experiments/run_ctx_matrix.py`, `experiments/ctx_matrix.log`,
`experiments/results_context_comparison.ipynb`, `experiments/build_ctx_comparison_notebook.py`.
체크포인트: `checkpoints/{blaine,residue}_ctx{N}/` (Full), `checkpoints/{blaine,residue}_ctx{N}_lora/` (LoRA), N ∈ 11개 값.
backtest: `eval/backtest_{blaine,residue}_ctx{N}_{zeroshot,lora,full}.csv` (66개).

---

## 2026-09-07 — IQR 이상치 제거 유무 A/B (실측치 데이터)

사용자 요청: 구 folder `01_iqr_outlier_removal`의 A/B를 실측치 데이터로 재현. context_length는
2026-09-06 스윕 최적값 사용 (**blaine 512 / residue 256**, 구 folder는 1024/1536이었음).

- `config.IQR_OUTLIER_K = 1.5` + `build_dataset.remove_iqr_outliers()` / `iqr_cols()` /
  `build(remove_iqr=True)` 옵션 추가 (canonical 동작 불변, 기본값 False). `iqr_cols()` =
  `RP_proc_time` 제외 전 raw feature + blaine/residue — 구 folder와 동일.
- `experiments/iqr/build_variants.py` → `data/processed/{with_iqr,without_iqr}/quality_timeseries.csv`.
  `without_iqr`는 canonical과 byte 동일. `with_iqr`는 **115,210개 값 NaN 처리**
  (최다: feed_slag 29,189 = SLAG 미투입 배치가 통계적 이상치로 잡힘 — 구 folder와 같은 현상).
  실측 타깃도 일부 제거: blaine null 0.8257→0.8288, residue 0.8255→0.8339.
- `experiments/iqr/run_iqr_experiment.py`: 4 모델 {타깃 × withiqr/withoutiqr 학습} full/predlen4/
  1000steps/lr1e-6/batch64, 각각 raw/clean 두 테스트셋 backtest = 8 backtest. wall-clock ~14분.

### 결과 (정규화 skill-score / MAE; <1 = naive 이김)

| target | eval set | IQR 미적용 | IQR 적용 | Δskill (적용−미적용) |
|---|---|---|---|---|
| blaine (ctx512) | **raw** (주지표) | **0.8319** / MAE 69.63 | 0.8361 / MAE 69.96 | +0.0042 |
| blaine (ctx512) | clean | **0.8652** / MAE 69.26 | 0.8715 / MAE 69.76 | +0.0063 |
| residue (ctx256) | **raw** (주지표) | **0.8086** / MAE 0.4585 | 0.8093 / MAE 0.4587 | +0.0007 |
| residue (ctx256) | clean | **0.8723** / MAE 0.4145 | 0.8749 / MAE 0.4160 | +0.0026 |

### 결론
- **IQR 이상치 제거는 실측치 데이터에서도 도움이 안 된다.** 4개 (타깃 × 테스트셋) 조합 전부에서
  IQR 미적용이 MAE·R²·skill-score 우세 (또는 residue-raw처럼 사실상 동률). 방향이 100% 일관.
- 다만 **효과 크기가 구 데이터보다 더 작아짐** — Δskill 0.0007~0.0063으로 재학습 노이즈(~0.004)와
  같은 수준. 구 folder에선 blaine이 "IQR 껐을 때만 naive를 이김"이었지만(Δskill ~0.011), 실측치에선
  blaine이 IQR on/off 둘 다 naive를 여유 있게 이김(0.832 vs 0.836) — carry-forward 오염이 없어지니
  이 축의 중요도 자체가 낮아짐.
- spec 정확도·80% 커버리지는 조합마다 승패 엇갈림 (IQR 적용이 커버리지 근소 우세인 경우 있음)
  — 절대 수준이 낮은 저신호 지표라 결론 안 뒤집음.
- **canonical 파이프라인 = IQR 미적용 유지** 결정 유효 (구 folder 결론 재확인).
- 한계: 조합당 단일 run (시드 고정 안 함), Δ가 노이즈 수준이라 "미적용이 확실히 낫다"기보다
  "IQR을 켤 이유가 없다"에 가까움.

산출물: `experiments/iqr/{build_variants,run_iqr_experiment,build_results_notebook}.py`,
`experiments/iqr/iqr_experiment.log`, `experiments/iqr/results_iqr_comparison.ipynb`,
`data/processed/{with_iqr,without_iqr}/quality_timeseries.csv`,
체크포인트 `checkpoints/{blaine_ctx512,residue_ctx256}_{withiqr,withoutiqr}/`,
backtest `eval/backtest_{blaine_ctx512,residue_ctx256}_{withiqr,withoutiqr}_evalon_{raw,clean}.csv` (8개).

---

## 2026-09-07 — prev-quality covariate 유무 A/B (실측치 데이터)

사용자 요청: `blaine_prev`/`residue_prev`를 known-future covariate로 **넣을 때 vs 아예 뺄 때**
비교. 구 folder 2026-08-12 no-prev vs prev 비교의 실측치 재현. context_length는 스윕 최적값
(blaine 512 / residue 256).

- `dataset_utils.load_processed(target, include_prev=True)` — `include_prev=False`면 두 컬럼 완전
  제거 (past covariate로도 안 들어가게). `finetune_chronos2.py --no-prev` / `backtest.py --no-prev`
  플래그 추가. canonical 동작 불변.
- `experiments/prev/run_prev_experiment.py`: 4 모델 {타깃 × prev/noprev} full/predlen4/1000steps/
  lr1e-6/batch64, 같은 canonical CSV·같은 평가 지점. wall-clock ~11분.

### 결과 (정규화 skill-score / MAE; <1 = naive 이김)

| target | prev 미사용 | prev 사용 (sparse) | Δskill (사용−미사용) |
|---|---|---|---|
| blaine (ctx512)  | 0.8363 / MAE 70.00 | **0.8319** / MAE 69.63 | −0.0044 |
| residue (ctx256) | 0.8090 / MAE 0.4588 | **0.8086** / MAE 0.4585 | −0.0004 |

(naive MAE: blaine 83.77 / residue 0.5685 — 두 arm 동일)

### 결론
- **실측치 데이터에서는 prev covariate(sparse) 유무가 거의 차이 없다.** 두 타깃 다 Δskill
  −0.0004~−0.0044 = 재학습 노이즈(~0.004) 수준. 방향은 "prev 사용 ≥ 미사용"으로 구 folder와
  같지만, 크기가 무의미.
- 구 folder 2026-08-12에선 prev 추가가 "확실한 효과"(두 타깃 다 처음으로 MAE 포함 전 지표에서
  naive 앞섬)였는데, 실측치에선 그 효과가 사실상 사라짐.
- 해석: 실측치 데이터는 타깃(blaine/residue)이 sparse라 context window 안에 실측 시각 값들이 이미
  들어있음 → "4h 전 실측값"이라는 sparse prev covariate는 그 in-context 정보와 대부분 중복. 구
  (오염) 데이터는 비측정 시각에 carry-forward 값이 섞여 context가 노이지했고, 그때는 명시적 prev가
  더 유용했던 것으로 보임.
- **함의**: sparse prev ≈ no-prev 라면, 실제 신호는 **ffill prev**(매시간 최근 실측값을 명시적
  앵커로)에 있을 가능성 — 구 folder 2026-09-02가 sparse→ffill에서 Δskill ~0.025를 봤음. sparse vs
  ffill A/B 재확인이 다음 우선순위.
- prev covariate 자체는 비용이 거의 없고(컬럼 2개) 방향이 해롭지 않아 canonical에서 유지.
- 한계: 조합당 단일 run.

산출물: `experiments/prev/{run_prev_experiment,build_results_notebook}.py`,
`experiments/prev/prev_experiment.log`, `experiments/prev/results_prev_comparison.ipynb`,
체크포인트 `checkpoints/{blaine_ctx512,residue_ctx256}_{prev,noprev}/`,
backtest `eval/backtest_{blaine_ctx512,residue_ctx256}_{prev,noprev}.csv` (4개).

---

## 2026-09-07 — prediction_length (train) 1 vs 4 A/B (실측치 데이터)

사용자 요청: 학습 `--prediction-length` 4(현행) vs 1 비교. 구 folder 2026-08-18 predlen 스윕의
실측치 재현. context_length = 스윕 최적값 (blaine 512 / residue 256). 평가는 두 arm 모두 1-step
(`backtest.py` 하드코딩 — chronos-2는 학습보다 짧은 horizon 추론 OK). full/1000steps/lr1e-6/batch64.
`experiments/predlen/run_predlen_experiment.py`, wall-clock ~11분.

### 결과 (정규화 skill-score / MAE / R²; <1 = naive 이김)

| target | predlen=1 | predlen=4 (현행) | Δskill (4−1) | ΔMAE (%) |
|---|---|---|---|---|
| blaine (ctx512)  | skill 0.8491 / MAE 71.05 / R² 0.451 | **skill 0.8319 / MAE 69.63 / R² 0.474** | **−0.0172** | −2.0% |
| residue (ctx256) | skill 0.8423 / MAE 0.4782 / R² 0.754 | **skill 0.8086 / MAE 0.4585 / R² 0.770** | **−0.0337** | −4.1% |

(naive MAE: blaine 83.77 / residue 0.5685 — 두 arm 동일)

### 결론
- **predlen=4가 두 타깃 모두 확실히 우세.** Δskill −0.017 / −0.034 = 재학습 노이즈(~0.004)의
  4~8배 — **지금까지 실측치에서 돌린 A/B 중 가장 큰 효과** (context_length는 평탄, IQR·prev는
  노이즈 수준이었음). MAE −2.0% / −4.1%, R² +0.02.
- **예측대로 실측치에서 효과가 구 데이터보다 더 큼** (구: blaine MAE −3.1% / residue −1.6%,
  이번: blaine −2.0% / residue **−4.1%** — residue가 이번에 더 큼). 타깃이 더 sparse해서
  predlen=1의 학습 신호 낭비가 심하다는 가설과 일치.
- predlen=1이 80% 커버리지만 근소 우세 (blaine 67.2 vs 66.2, residue 72.9 vs 69.7) — 저신호
  지표이고 둘 다 목표 미달이라 결론 안 뒤집음.
- **predlen=4 유지 결정 강하게 재확인** — 이 프로젝트에서 선택이 실제로 중요한 유일한
  하이퍼파라미터. `finetune_chronos2.py` 기본값 4 그대로 둠.
- 한계: 조합당 단일 run (하지만 Δ가 노이즈의 4~8배라 방향은 확실).

산출물: `experiments/predlen/{run_predlen_experiment,build_results_notebook}.py`,
`experiments/predlen/predlen_experiment.log`, `experiments/predlen/results_predlen_comparison.ipynb`,
체크포인트 `checkpoints/{blaine_ctx512,residue_ctx256}_predlen{1,4}/`,
backtest `eval/backtest_{blaine_ctx512,residue_ctx256}_predlen{1,4}.csv` (4개).

---

## 2026-09-07 — prev-density: sparse vs ffill A/B (실측치 데이터)

사용자 요청: `blaine_prev`/`residue_prev` covariate 를 sparse(현행) vs ffill(직전 실측값 매시간
forward-fill) 비교. 구 folder 2026-09-02 실험의 실측치 재현. ctx = 스윕 최적값 (blaine 512 /
residue 256).

- `build_dataset.build(prev_density="sparse"|"ffill")` 옵션 추가 (canonical=sparse 불변).
  ffill = 시간 그리드 재색인 후 `groupby("item_id")[PREV_QUALITY_COLS].ffill()` 한 줄.
- `experiments/prev_density/build_variants.py` → `data/processed/prev_{sparse,ffill}/`.
  `_prev` non-null: sparse 17.2% → ffill 99.99%. 타깃 컬럼은 두 arm 다 sparse 유지.
- **누출 감사 통과**: ffill `_prev` 값은 모든 행에서 그 행보다 ≥4h 오래됨 (min 4.0h / median 7.0h)
  → predlen≤4에서 미래 실측 노출 0. 구 folder 감사와 일치.
- 4 모델 {타깃 × sparse/ffill} full/predlen4/1000steps/lr1e-6/batch64. wall-clock ~11분.

### 결과 (정규화 skill-score / MAE / R²; <1 = naive 이김)

| target | sparse (현행) | ffill | Δskill (ffill−sparse) |
|---|---|---|---|
| blaine (ctx512)  | **skill 0.8319 / MAE 69.63 / R² 0.474** | skill 0.8316 / MAE 69.61 / R² 0.473 | −0.0003 |
| residue (ctx256) | **skill 0.8086 / MAE 0.4585 / R² 0.770** | skill 0.8355 / MAE 0.4749 / R² 0.760 | **+0.0269** |

### 결론 — 구 folder와 반대
- **blaine**: sparse ≈ ffill (Δskill −0.0003, 무의미).
- **residue**: **ffill이 유의미하게 나쁨** (Δskill +0.027 = 재학습 노이즈 ~7배, MAE +3.6%, R² −0.010).
- **구 folder 2026-09-02 결론(ffill이 두 타깃 다 유의미하게 나음, Δskill ~−0.025)이 실측치에서
  뒤집힘** — blaine은 무효과, residue는 역방향.
- 해석: 구(오염) 데이터는 타깃이 forward-fill돼 있어 context가 (가짜) dense였고, sparse `_prev`가
  더할 게 적었음. 실측치는 타깃이 진짜 sparse라 모델이 실제 추론을 해야 하는데, ffill `_prev`를
  매시간 앵커로 주면 residue가 "직전값 추종"으로 과의존 → `fresh_reading_only` 평가(전환 지점만)에서
  손해. residue ffill의 80% 커버리지가 올라간 것(69.7→72.1)도 예측이 앵커 쪽으로 당겨진 것과 일치.
- 이는 2026-08-12에 sparse를 고른 **원래 근거**("매시간 반복 노출 → naive persistence로 붕괴 위험")가
  깨끗한 데이터에선 다시 유효하다는 뜻. 구 folder의 ffill 권고는 오염 데이터의 인공물이었음.
- **canonical = sparse 유지** 결정. `build()` 기본값 sparse 그대로.
- 한계: 조합당 단일 run (하지만 residue Δ가 노이즈의 7배).

산출물: `experiments/prev_density/{build_variants,run_prev_density_experiment,build_results_notebook}.py`,
`experiments/prev_density/prev_density_experiment.log`,
`experiments/prev_density/results_prev_density_comparison.ipynb`,
`data/processed/prev_{sparse,ffill}/quality_timeseries.csv`,
체크포인트 `checkpoints/{blaine_ctx512,residue_ctx256}_prevdensity_{sparse,ffill}/`,
backtest `eval/backtest_{blaine_ctx512,residue_ctx256}_prevdensity_{sparse,ffill}.csv` (4개).

<!-- next entries go here -->
