# 02 · 물리적 이상치 처리 방식 A/B (2026-09-02)

`01_iqr_outlier_removal`에서 **"통계적 IQR 제거는 하지 않는다"**가 확정된 뒤, 남은 축인
**물리적 이상치를 어떻게 처리하는가**를 비교한다.

## 배경

현행 `clean()`([../data/build_dataset.py](../data/build_dataset.py))은 물리적으로 불가능한 값을
전부 **NaN 처리**한다. 참고 노트북(`chronos-2-cement_share.ipynb`)은 대신 **경계값으로 clip**하거나
행을 유지한다(다만 그 노트북의 댐퍼 clip은 변수 참조 버그로 실제 데이터엔 미반영 상태).

Chronos-2는 NaN을 네이티브로 마스킹하므로 **"구멍을 불필요하게 만들지 말라"**는 원칙
(= IQR 실험의 결론)을, 실제 값이 복구 가능한 두 지점에 적용해 본다.

데이터 품질 5개 차원(정확성 / 완전성 / 일관성 / 적시성 / 유효성) 중 이상치 제거는
**정확성·유효성**을 담당하며 **완전성과 트레이드오프** 관계다. 아래 두 실험은
"유효성 위반값을 NaN으로 날릴지(완전성 ↓) vs 경계값으로 교정해 레코드를 살릴지(완전성 유지)"를
검증한다.

## 3개 분기

| 분기 | 내용 | 코드 차이 |
|---|---|---|
| **baseline** | 현행 그대로. 정식 파이프라인 `build_dataset.py main()`과 동일 | `prepare_base()` |
| **expA** (댐퍼 clip) | 댐퍼 개도율이 0~100%를 `DAMPER_CLIP_MARGIN_PCT`(config.py, 기본 20%p) 이내로 벗어나면 NaN 대신 경계값(0/100)으로 clip. 그 이상은 기존대로 NaN(센서 고장) | `prepare_base(damper_oob="clip")` |
| **expB** (비-내수 유지) | 비-내수 행을 삭제하지 않고 유지, 품질 타깃(blaine/residue)만 NaN. 운전 조건 covariate 컨텍스트 보존 | `prepare_base(prodtype_mode="mask_target")` |

세 분기 모두 **행 수 / 타임스탬프 / 타깃값 / 학습·평가 분할 지점이 동일** → backtest 평가 지점이
같아 skill-score 비교가 공정하다. 분기 간 차이는 오직 feature 컬럼 값뿐:
- **expA**: 댐퍼 컬럼의 소수 셀 (NaN → 경계값)
- **expB**: 비-내수 시각의 전체 feature 컬럼 (NaN → 실제 값)

## 실행

```bash
# 1) 원본 xlsx가 있는 머신에서 3개 CSV 생성
python 02_physical_outlier_handling/build_baseline.py
python 02_physical_outlier_handling/build_expA_damper_clip.py
python 02_physical_outlier_handling/build_expB_prodtype_keep.py

# 2) GPU 머신에서 학습 + backtest (blaine ctx1024, residue ctx1536, 각 3분기)
bash train/run_phys_outlier_experiment.sh

# 3) 결과 비교
jupyter notebook 02_physical_outlier_handling/results_phys_outlier_comparison.ipynb
```

## 판단 기준

- **1차 지표**: normalized skill-score(`../eval/metrics_utils.py`), 보조로 MAE / R² / spec 정확도.
- 개선폭이 **재학습 노이즈(blaine skill-score 기준 ~0.004)보다 작으면** 해당 실험은 "차이 없음"으로
  보고 baseline 유지. 유의미하면 seed 2~3개로 재확인 후 채택.
- **expB 사전 확인**: 비-내수 행이 실제 몇 %인지, 그 시각에 품질 실측이 있는지. 비중이 매우 작으면
  (< 5%) expB는 스킵해도 무방하다. `build_baseline.py` 실행 로그의 행 수를,
  `build_expB_prodtype_keep.py` 실행 로그의 행 수(및 target null rate)와 비교하면 바로 보인다.
- 최종: 세 분기 중 성능이 가장 좋은 구성을 정식 파이프라인 기본값으로 채택하고 `EXPERIMENT_LOG.md`에 기록.

## 되돌리기

`config.py` / `data/build_dataset.py`의 추가된 인자는 전부 기본값이 현행 동작이라, 이 실험을 접어도
정식 파이프라인은 그대로다. 편집 전 백업: `config.py.bak_20260902`,
`data/build_dataset.py.bak_20260902`, `data/processed/quality_timeseries.csv.bak_20260902_before_phys_outlier_exp`.

## 결과

_(실행 후 채움 — `EXPERIMENT_LOG.md`에도 기록)_