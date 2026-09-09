# 03 · `_prev` covariate 밀도 A/B (2026-09-02)

`blaine_prev` / `residue_prev` 를 hourly 그리드에 **얼마나 촘촘히 채울지**를 비교한다.

## 배경

현행 `add_prev_quality_covariates()`([../data/build_dataset.py](../data/build_dataset.py))는
직전 실측 품질값을 **실측 scheduled 행(4시간 간격)에만** 넣고 그 사이 hourly 행은 NaN 으로 둔다
(**sparse**). 참고 노트북(`chronos-2-cement_share.ipynb` "이전품질 적용 후 Chronos")은 반대로
`df_fill_NA(option="copy_back")` = **backward-fill** 로 모든 행을 채웠다(**dense**).

- sparse 를 고른 원래 이유(EXPERIMENT_LOG 2026-08-12): 매 시간 같은 값이 반복 노출되면 모델이
  naive persistence 로 수렴할 위험. 단, **직접 A/B 로 확인한 적은 없음** — 추론으로 정한 기본값.
- 참고 노트북의 **bfill 은 누출**이다: 예측 구간(horizon)이 측정 경계를 넘으면 아직 관측되지 않은
  미래 실측값을 known-future covariate 로 노출한다. 그래서 이 실험의 dense 분기는 **ffill**(누출 없음,
  아래 참조)로 한다.

## 2개 분기

| 분기 | 내용 | 코드 차이 |
|---|---|---|
| **sparse** | 현행 그대로. 정식 파이프라인 `build_dataset.py main()` 과 동일 | `finalize_and_save(..., prev_density="sparse")` |
| **ffill** (dense) | 시간 그리드 재색인 후 `_prev` 를 item_id 별 forward-fill. 모든 행이 "가장 최근 lab 값" 앵커를 명시적으로 보유 | `finalize_and_save(..., prev_density="ffill")` |

두 분기 모두 **행 수 / 타임스탬프 / 타깃값 / 학습·평가 분할 지점이 동일**하고, backtest 평가 지점
(=실측 갱신 시점, 그 행의 `_prev` 는 원래 non-NaN)의 `_prev` 값도 **두 분기가 완전히 동일**하다
→ naive / skill-score 분모가 같아 공정 비교. 분기 간 차이는 오직 **비-실측 hourly 행의 `_prev` 2개
컬럼 값**뿐(sparse: NaN, ffill: 직전 실측값).

### ffill 이 누출이 아닌 이유 (prediction_length ≤ 4)

scheduled 행 `S` 의 sparse `_prev` 는 **이미 `S` 의 4시간 전 측정값**이다. 이를 앞으로 채우면 어떤
행 `h` 에 놓이든 그 값의 출처 timestamp 는 `≤ h − 4h`. origin `t`, horizon `t+1..t+4` 인 학습
윈도우에서 horizon 각 행의 출처는 항상 `≤ t` → origin 시점에 관측 가능. `smoketest_prev_density.py`
가 합성 데이터로 이 성질(누출 0건)을 검증한다.

## 실행

### 권장: 올인원 노트북 (빌드 · 누출 감사 · 학습 · 비교 한 번에)

```bash
jupyter notebook 03_prev_density_handling/run_and_compare_prev_density.ipynb
```

1절이 raw xlsx 에서 두 CSV 를 만들고, **2절 누출 감사(불변식 D/B/A/C)** 가 통과해야만
3절 학습으로 넘어간다. 3~4절은 정식 스크립트(`train/finetune_chronos2.py`, `eval/backtest.py`)를
그대로 subprocess 호출한다 — 노트북이 학습 경로를 재구현하지 않는다. 체크포인트/CSV 가 있으면
해당 단계는 건너뛴다(재실행 안전). GPU 로 약 4.5시간.

### CLI 로 나눠 돌리기 (동등)

```bash
python 03_prev_density_handling/smoketest_prev_density.py     # 합성 데이터 로직 확인
python 03_prev_density_handling/build_sparse.py               # 실데이터 CSV 2개
python 03_prev_density_handling/build_dense_ffill.py          # (build 로그의 null rate: sparse ~0.82, ffill ~0.00)
bash train/run_prev_density_experiment.sh                     # 학습 + backtest, ~4.5h
jupyter notebook 03_prev_density_handling/results_prev_density_comparison.ipynb  # 비교만
```

두 경로 모두 같은 파일 경로(`data/processed/prev_*`, `checkpoints/{target}_..._prevdensity_*`,
`eval/backtest_{target}_prevdensity_*.csv`)를 쓰므로 섞어 써도 된다.

## 판단 기준

- **1차 지표**: normalized skill-score(`../eval/metrics_utils.py`), 보조로 MAE / R² / spec 정확도.
- 개선폭이 **재학습 노이즈(blaine skill-score 기준 ~0.004)보다 작으면** "차이 없음" → 현행 sparse 유지.
  유의미하면 seed 2~3개로 재확인 후 채택.
- ffill 이 유의미하게 나으면 정식 파이프라인 기본값을 `prev_density="ffill"` 로 전환 검토
  (`data/build_dataset.py` `main()`, `data/dataset_utils.py`, `config.py` 주석). `EXPERIMENT_LOG.md` 기록.

## 되돌리기

`data/build_dataset.py` 에 추가된 `prev_density` 인자는 기본값이 `"sparse"`(현행 동작)이라, 이 실험을
접어도 정식 파이프라인은 그대로다.

## 결과

_(실행 후 채움 — `EXPERIMENT_LOG.md` 에도 기록)_
