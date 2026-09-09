# Chronos-2 Quality Net — Experiment Log

## 2026-08-04 — 새 컴퓨터 환경 설정 + 첫 LoRA 파인튜닝

### 환경 설정
- 이 컴퓨터(GPU: RTX A6000 x2, driver 552.74, CUDA 12.4)에 `chronos_quality_net` 프로젝트를 처음 실행.
- 기존 Python 3.10/3.11 설치가 손상돼 있어(핵심 `python.exe` 누락) `winget install Python.Python.3.11`로 재설치.
- `chronos_quality_net\.venv` (Python 3.11.9)에 `torch==2.5.1+cu121`, `chronos-forecasting==2.3.1`, `peft==0.20.0` 설치. CUDA 정상 인식 확인.
- 참고: 이 컴퓨터에는 기존에 `mygpu`라는 이름의 venv(`C:\virtual enviroment\mygpu`, Python 3.9.13)가 있었으나 **TensorFlow 전용**(1차년도 Keras ANN용으로 추정)이고 torch는 CPU 빌드뿐이라 이 프로젝트엔 재사용하지 않음. 대신 전용 `.venv`를 새로 만듦.

### 코드 검증
- `config.py`의 `CONTROL_COLS`(9개)를 1차년도 원본(`20260113_reverse modeling.../Reverse_Blaine_v2/autoControl_b_v2.ipynb`의 `CONTROL_COLS_9`)과 직접 대조 — 7/9는 명확히 일치, 2개(`POLYCOM_160_BF_FANDP`, `MILL_186_BF_DP`)는 원매핑 불확실성이 코드 주석에 이미 명시됨(정직하게 문서화).
- Train/Val/Test를 시간순 분할로 개선(원본 ANN의 랜덤 분할은 미래 정보 누출 위험 있었음).
- 결론: 구조적으로 문제 없음, 파인튜닝 진행 판단.

### 결과: Zero-shot vs LoRA 파인튜닝 (2000 steps, 기본 하이퍼파라미터)

**BLAINE** (spec 3700-3900)

| 방식 | MAE | RMSE | R² | Spec 정확도 | 80% 구간 커버리지 |
|---|---|---|---|---|---|
| Naive baseline (직전값 반복) | 61.28 | 85.90 | 0.432 | **78.15%** | — |
| Chronos-2 zero-shot | 68.55 | 89.28 | 0.386 | 74.17% | 39.74% |
| Chronos-2 LoRA 파인튜닝 | 68.08 | 88.90 | 0.391 | 74.83% | 39.07% |

**Remain / 44µm 잔사** (spec 7-9)

| 방식 | MAE | RMSE | R² | Spec 정확도 | 80% 구간 커버리지 |
|---|---|---|---|---|---|
| Naive baseline (직전값 반복) | 0.37 | 0.53 | 0.142 | **73.20%** | — |
| Chronos-2 zero-shot | 0.43 | 0.56 | 0.045 | 70.59% | 37.25% |
| Chronos-2 LoRA 파인튜닝 | 0.42 | 0.56 | 0.059 | 71.24% | 33.33% |

### 결론
- LoRA 파인튜닝(기본 설정, 2000 steps ≈ 1 epoch 상당)은 zero-shot 대비 **거의 개선 없음** (1% 이내).
- 두 타깃 모두 여전히 naive baseline(직전 값 반복)보다 성능이 낮음.
- 1차년도 ANN(group MLP, Blaine 95.1% / Residue 94.5% CF accuracy, KTL 공인 95%+)과 격차 큼 — 아직 Chronos-2가 이 데이터를 ANN만큼 활용 못 하는 상태.
- 80% 예측구간 실제 커버리지도 33~40%대로 목표(80%)에 크게 못 미쳐 불확실성 추정이 과신 상태.
- 결과 파일: `eval/backtest_BLAINE.csv`, `eval/backtest_Remain.csv`(=파인튜닝 결과), `eval/backtest_BLAINE_zeroshot.csv`, `eval/backtest_Remain_zeroshot.csv`(=zero-shot 보존본). 체크포인트: `checkpoints/BLAINE/final`, `checkpoints/Remain/final`.

### 다음에 시도해볼 것 (아직 미실행)
1. `--finetune-mode full` (LoRA 대신 전체 파인튜닝)
2. `--num-steps` 증가 또는 `--learning-rate` 조정
3. `BLAINE_prev`/`Remain_prev`처럼 "직전 품질값"을 명시적 covariate로 추가하는 방안 재검토 (1차년도 ANN이 쓴 핵심 트릭이자, 원본 공유 노트북의 "이전품질 적용 후 Chronos" 섹션과 동일한 아이디어)

---

## 2026-08-12 — 세 번째 컴퓨터로 이전 + 2026 xlsx 갱신 + prev covariate + ProdType 버그 수정

### 환경 설정
- 또 다른 컴퓨터(GPU: RTX 5060, Blackwell/sm_120)로 이전. `.venv`/`__pycache__`는 안 가져옴(원래 안 가져와도 되는 게 맞음 — venv는 머신마다 새로 만들어야 함).
- PyTorch 2.11.0+cu128 설치(Blackwell 지원에 cu121 등 예전 빌드는 안 맞음) — GPU 실제 matmul 연산까지 검증 완료.

### 컬럼명을 1차년도 ANN 이름으로 리네임 (사용자 요청)
- 기존 `config.py`는 POLYCOM_/MILL_ 원문 이름을 표준으로 썼는데, 1차년도 ANN이 쓰던 `RP_*`/`mill_*`/`blaine`/`residue` 이름으로 전면 리네임(불확실 매핑 4개는 원문 유지: `POLYCOM_160_BF_FANDP`, `MILL_186_BF_DP`, `POLYCOM_PCOutletBE_142/145`, `POLYCOM_Impal`).
- 이 리네임이 `build_dataset.py`에 있던 기존 버그(옛 이름 참조하다 컴퓨터 이전 중 리팩토링이 멈춘 상태)를 부수적으로 해결함.

### 새 xlsx (2차년도 소스) 반영
- `운전 & 품질 데이터_2026.xlsx`로 전환. 헤더 구조가 이전과 달라짐(헤더 2행/데이터 3행부터 시작, 그리고 결정적으로 **"근무자" 컬럼이 아예 빠짐** → 그대로 쓰면 2번째 컬럼부터 전부 밀려서 조용히 데이터가 깨짐). `HEADER_ROW`/`DATA_START_ROW`/`STD_COLUMNS_BY_POSITION`을 새 45컬럼 레이아웃에 맞게 재작성, 실제 원본 행과 값 대조로 검증 완료.
- 참고: 파일명은 "_2026"이지만 실제 데이터 범위는 2017/2020~2025-11-17로 기존과 거의 동일 — 신규 2026년 실측치가 추가된 게 아니라 헤더 포맷만 재정리된 재추출 파일.

### blaine_prev / residue_prev covariate 추가 (sparse)
- 참고 노트북(`Downloads/chronos-2-cement_share.ipynb`, "이전품질 적용 후 Chronos" 섹션)의 `fill_previous_quality_measurements()`를 이식. 실측 행(4시간 간격)끼리만 shift(1), bfill 없이 sparse 유지. Dense(forward-fill) 대안도 논의했으나, (a) 학습 컨텍스트 전체에 반복 노출되면 naive persistence로의 수렴을 더 강화할 위험, (b) idle 여부는 이미 `restart_hr`/`prior_gap_hr`가 별도로 맡고 있어 NaN에 그 의미를 얹을 필요가 없다는 이유로 sparse로 결정.
- `finetune_chronos2.py`/`eval/backtest.py`/`inference/quality_predictor.py`에서 known-future covariate로도 전달하도록 수정(원본 노트북처럼 past+future 양쪽에 사용, blaine/residue 모델 양쪽에 두 covariate 모두 포함).
- 검증: 13,316개 실측 행 전부 "직전 실측값"과 정확히 일치(mismatch 0), 비실측 37,893행 전부 NaN(누출 없음).

### ProdType 필터 버그 발견 및 수정 (중요)
- `clean()`의 `ProdType == '내수'` 필터가 원본 행의 **68.8%를 제거**하고 있었음. 원인: `ProdType`이 품질 실측 시각(4시간마다)에만 기록되고 나머지 시간은 빈 값 — 그런데 필터가 매 행마다 정확히 "내수"를 요구해서, 실제로 가동 중인 시간(빈 ProdType 행의 95.1%가 `RP_proc_time>0`)까지 통째로 걸러짐. **2025년 데이터는 ProdType이 단 한 행도 안 채워져 있어 그대로 뒀으면 2025년 전체가 삭제될 뻔함.**
- 수정: `ProdType`을 item_id별로 forward-fill 후 필터 적용. 결과: 센서/제어 컬럼 결측률이 대폭 개선(예: `RP_roller_p1` 결측률 대폭 하락), `restart_hr==0`(가짜 재가동 이벤트) 34,894회 → 6,579회로 정상화 — 즉 idle 감지 로직 자체가 이 버그로 왜곡되어 있었음.

### 결과 비교 (2000 steps, LoRA, 새 2026 데이터 기준)

**blaine** (spec 3700-3900)

| 방식 | MAE | RMSE | R² | Spec 정확도 | 80% 구간 커버리지 |
|---|---|---|---|---|---|
| Naive baseline | 63.43 | 91.54 | 0.416 | 74.86% | — |
| Chronos-2 (covariate 없음) | 64.12 | 87.18 | 0.471 | 74.35% | 47.40% |
| Chronos-2 (+prev, ProdType 버그 있음) | 61.13 | 85.27 | 0.494 | 74.75% | 42.98% |
| Chronos-2 (+prev, ProdType 수정) | 62.05 | 86.06 | 0.484 | 74.80% | 39.86% |

**residue** (spec 7-9)

| 방식 | MAE | RMSE | R² | Spec 정확도 | 80% 구간 커버리지 |
|---|---|---|---|---|---|
| Naive baseline | 0.41 | 0.61 | 0.785 | 81.84% | — |
| Chronos-2 (covariate 없음) | 0.42 | 0.59 | 0.800 | 81.17% | 47.46% |
| Chronos-2 (+prev, ProdType 버그 있음) | 0.40 | 0.57 | 0.809 | 81.23% | 45.58% |
| Chronos-2 (+prev, ProdType 수정) | 0.40 | 0.57 | 0.809 | 80.27% | 44.37% |

### 결론
- **`_prev` covariate 추가는 확실히 효과 있음** — 두 타깃 모두 naive baseline을 처음으로 MAE 포함 전 지표에서 앞섬(이전까진 RMSE/R²만 이기고 MAE는 지는 경우 있었음).
- **ProdType 버그 수정은 데이터 정확성 측면에서는 중요한 수정**(가짜 idle/재가동 왜곡 해소, 2025년 데이터 유실 방지)**이지만, 이번 backtest 지표엔 뚜렷한 개선으로 안 나타남**(blaine 소폭 하락, residue 거의 동일). 가설: 데이터가 거의 3배 늘었는데 학습 스텝은 2000으로 그대로라 상대적으로 덜 학습된 상태일 가능성 — `--num-steps` 증가 재시도가 다음 후보.
- 80% 예측구간 커버리지는 여전히 40~47%대로 목표(80%)에 크게 못 미침 — 불확실성 과신 문제는 이번 작업들로 해결 안 됨, 별도 이슈로 남음.
- 1차년도 ANN(95%대 spec 정확도)과는 여전히 큰 격차.

### 결과/체크포인트 파일 위치
- 현재(최종) 체크포인트: `checkpoints/blaine/final`, `checkpoints/residue/final` (prev + ProdType 수정 버전)
- 비교용 백업: `checkpoints/blaine_no_prev`, `checkpoints/residue_no_prev` (covariate 없음), `checkpoints/blaine_prev_prodtypebug`, `checkpoints/residue_prev_prodtypebug` (prev만 있고 ProdType 버그 있는 버전)
- 대응 backtest 결과: `eval/backtest_{blaine,residue}.csv`(최종), `eval/backtest_{blaine,residue}_no_prev.csv`, `eval/backtest_{blaine,residue}_prev_prodtypebug.csv`
- 8/4 구버전(다른 컴퓨터·구 데이터) 체크포인트: `checkpoints/_archive_20260804_BLAINE_old-data`, `_archive_20260804_Remain_old-data`

### 다음에 시도해볼 것
1. `--num-steps` 늘려서(예: 4000~5000) ProdType 수정 버전 재시도 — 데이터 3배 증가분을 학습이 못 따라간 것인지 확인
2. 80% 구간 커버리지 문제(불확실성 과신) 별도 조사
3. `--finetune-mode full` (LoRA 대신 전체 파인튜닝) — 아직 미실행

---

## 2026-08-14 — `--num-steps 4000` 재시도

8/12 "다음에 시도해볼 것" 1번 실행. ProdType 버그 수정 버전(+prev covariate)을 2000 → 4000 steps로 재학습.

| 타깃 | 방식 | MAE | RMSE | R² | Spec 정확도 |
|---|---|---|---|---|---|
| blaine | 2000 steps (기존) | 66.50 | 90.85 | 0.4179 | 70.21% |
| blaine | 4000 steps | 66.44 | 90.72 | 0.4195 | 70.41% |
| residue | 2000 steps (기존) | 0.4392 | 0.6053 | 0.8182 | 80.49% |
| residue | 4000 steps | 0.4405 | 0.6090 | 0.8160 | 80.29% |

**결론**: blaine은 미세하게 개선, residue는 미세하게 악화 — 사실상 유의미한 차이 없음. 데이터 3배 증가를
스텝 2배로는 못 따라잡는다기보다, 이 하이퍼파라미터 축 자체가 별 영향이 없는 것으로 보임. `--num-steps`는
이후 실험에서 계속 기본값 2000 사용. 결과 파일: `eval/backtest_{blaine,residue}_4000steps.csv`.

---

## 2026-08-18 — 4번째 컴퓨터로 재이전(8/4 GPU 머신으로 복귀, E: 드라이브) + 환경 정리

- 8/12 항목의 "또 다른 컴퓨터"(RTX 5060)에서, 이번엔 8/4에 처음 세팅했던 GPU 머신(RTX A6000 x2,
  `E:\2_AI CODE\10_Cement\20260804_notglass\cement_code\`)으로 다시 이전. 폴더가 그 사이 한글명
  (`삼표시멘트_코드`)에서 영문명(`cement_code`)으로 리네임되어 있었음.
- 이 리네임 때문에 발견된 문제 2건, 둘 다 수정:
  - `config.py`의 `SOURCE_XLSX` 기본 경로 주석이 "이 컴퓨터는 E: 드라이브가 아님"이라고 적혀 있었는데,
    실제로는 이 GPU 머신이 E: 드라이브라 사실과 반대로 적혀 있었음 — 주석을 현재 상태에 맞게 정정
    (원본 xlsx 자체는 이 머신에 없는 게 정상 — README 참고, processed CSV만 있으면 학습/평가 가능).
  - Jupyter 커널 `chronos_quality_net`의 `kernel.json`이 옛 폴더명(`삼표시멘트_코드`) 경로의
    `python.exe`를 가리키고 있어 깨져 있었음 — 현재 폴더명(`cement_code`) 경로로 수정.
- venv(`torch==2.5.1+cu121`, CUDA 정상 인식)·`chronos`/`peft`/`pandas` 등 의존성은 이 머신에 이미
  설치돼 있었고, `data/processed/quality_timeseries.csv`도 그대로 포함되어 있어 재설치/재빌드 없이 바로
  이어서 작업 가능했음.

---

## 2026-08-18 — Full fine-tuning 추가 (context_length=512, prediction_length=1)

8/12 "다음에 시도해볼 것" 3번 실행. `--finetune-mode full --learning-rate 1e-6`(LoRA보다 10배 낮은 lr,
나머지 기본값 동일)로 blaine/residue 재학습.

| 타깃 | 방식 | MAE | RMSE | R² | Spec 정확도 | 80% 커버리지 |
|---|---|---|---|---|---|---|
| blaine | Naive | 63.21 | 92.94 | 0.3908 | 72.59% | — |
| blaine | Zero-shot | 66.48 | 90.75 | 0.4192 | 70.31% | 41.01% |
| blaine | LoRA | 66.50 | 90.85 | 0.4179 | 70.21% | 41.41% |
| blaine | Full fine-tuning | 66.24 | 91.03 | 0.4156 | 70.61% | 41.21% |
| residue | Naive | 0.415 | 0.6203 | 0.8091 | 81.31% | — |
| residue | Zero-shot | 0.4396 | 0.6071 | 0.8172 | 79.67% | 39.63% |
| residue | LoRA | 0.4392 | 0.6053 | 0.8182 | 80.49% | 39.63% |
| residue | Full fine-tuning | 0.4370 | 0.5987 | 0.8222 | 81.10% | 41.88% |

**결론**: Full fine-tuning이 LoRA보다 소폭 우세(특히 residue R²가 처음으로 0.82를 넘김). blaine은
여전히 naive baseline을 못 이김. 결과를 `results.ipynb`로 정리(GPU 재사용 없이 CSV만 읽어 표/그래프
생성, 이후 모든 `results_*.ipynb`의 템플릿이 됨). 체크포인트: `checkpoints/{blaine,residue}_full/final`.

---

## 2026-08-18 — context_length=24 실험 (prediction_length=1)

context_length 512(기본, 약 21일) vs 24(하루)를 zero-shot/LoRA/Full 전부 비교.

| 타깃 | 방식 | MAE (ctx512→ctx24) | R² (ctx512→ctx24) |
|---|---|---|---|
| blaine | Zero-shot | 66.48 → 68.11 | 0.4192 → 0.3731 |
| blaine | Full fine-tuning | 66.24 → 67.80 | 0.4156 → 0.3723 |
| residue | Zero-shot | 0.4396 → 0.4420 | 0.8172 → 0.8005 |
| residue | Full fine-tuning | 0.4370 → 0.4501 | 0.8222 → 0.7975 |

**결론**: context를 하루치로 줄이면 전반적으로 나빠짐(80% 커버리지만 소폭 상승, 41→42~43% 수준이지만
여전히 목표 미달). context_length=512(기본값) 유지가 맞다는 결론. 결과: `results_context24.ipynb`.

---

## 2026-08-18 — prediction_length 스윕: 1(기존) vs 4 (context_length=512 고정) → **prediction_length=4로 고정 결정**

**가설**: blaine/residue 실측치는 `QUALITY_MEASUREMENT_HOURS=[0,4,8,12,16,20]`, 즉 정확히 4시간
간격으로만 존재하고 나머지는 NaN. `chronos/chronos2/dataset.py`의 학습 샘플링(`np.random.randint`로
전체 시계열에서 랜덤 시점을 뽑아 그 다음 `prediction_length`시간을 타깃으로 사용)을 코드 레벨에서
확인한 결과, prediction_length=1이면 뽑힌 시점의 바로 다음 1시간이 실측 시각일 확률이 25%뿐이라
학습 신호의 상당수가 낭비되는 구조. prediction_length≥4면 4시간 grid 특성상 어떤 4시간 구간을 뽑아도
반드시 실측 시각이 하나 포함되어 신호 밀도가 개선될 것이라는 가설을 세우고 검증.

| 타깃 | 방식 | MAE (pred=1→pred=4) | R² (pred=1→pred=4) | 80% 커버리지 (pred=1→pred=4) |
|---|---|---|---|---|
| blaine | LoRA | 66.50 → 66.31 | 0.4179 → 0.4260 | 41.41% → 42.80% |
| blaine | Full fine-tuning | 66.24 → 64.18 | 0.4156 → 0.4550 | 41.21% → 48.36% |
| residue | LoRA | 0.4392 → 0.4396 | 0.8182 → 0.8197 | 39.63% → 43.92% |
| residue | Full fine-tuning | 0.4370 → 0.4298 | 0.8222 → 0.8295 | 41.88% → 51.17% |

**결론**: 가설대로 prediction_length=4가 1보다 나음 — 특히 full fine-tuning에서 뚜렷(blaine MAE
-3.1%, R² +0.039; residue MAE -1.6%, R² +0.007). 80% 구간 커버리지도 두 타깃 모두 큰 폭 개선(지금까지
실험 중 최대 개선폭) — 불확실성 과신 문제가 처음으로 의미 있게 완화됨. LoRA는 효과가 약함(파라미터가
적어 밀도 높은 신호를 못 살리는 것으로 추정). blaine은 여전히 naive(63.21)를 못 이김. **이후 모든
실험은 prediction_length=4로 고정하기로 결정**(사용자 확정). 평가는 기존 `eval/backtest.py`를 그대로
사용(`predict_df`의 `prediction_length=1` 하드코딩은 유지 — chronos-2는 학습 때보다 짧은
prediction_length로 추론해도 문제없음을 `chronos/chronos2/pipeline.py`에서 확인, 그래서 "1-step 예측
정확도"라는 잣대는 이전 실험들과 그대로 비교 가능). 결과: `results_context512.ipynb`. 체크포인트:
`checkpoints/{blaine,residue}_predlen4`, `_full_predlen4`.

⚠️ **미반영 TODO**: `train/finetune_chronos2.py --prediction-length` 기본값(현재 1)과 `README.md`는
아직 4로 안 바꿈 — 다음 세션에서 정리 필요.

---

## 2026-08-18 — context_length 스윕 (prediction_length=4 고정): 512 / 168 / 73 / 128

prediction_length=4를 고정한 채 context_length를 4개 지점(512, 168, 73, 128)에서 비교. 128/73/168은
zero-shot도 다시 평가(context_length는 파인튜닝 여부와 무관하게 추론 입력 자체를 바꾸는 값이라 재사용
불가 — prediction_length 스윕 때 zero-shot을 재사용했던 것과 다른 점).

**Full fine-tuning 기준 요약** (context_length 내림차순):

| context_length | blaine MAE | blaine R² | residue MAE | residue R² | 결과 노트북 |
|---|---|---|---|---|---|
| **512** (1위) | **64.18** | **0.4550** | **0.4298** | **0.8295** | `results_context512.ipynb` |
| 168 (2위) | 65.89 | 0.4164 | 0.4386 | 0.8144 | `results_context168.ipynb` |
| 73 (3위) | 66.66 | 0.4010 | 0.4371 | 0.8158 | `results_context73.ipynb` |
| 128 (4위) | 68.51 | 0.3635 | 0.4401 | 0.8111 | `results_context128.ipynb` |

**결론**:
- context_length=512가 압도적 1위. 512와 168 사이 격차가 커서(blaine MAE 64.18→65.89), 그 사이
  (256/384 등)에 성능이 급격히 좋아지는 지점이 있을 가능성.
- **context_length와 성능이 단조적으로 비례하지 않음** — blaine은 512>168>73>128 순이지만 residue는
  512>73>168>128 순(73과 168의 순서가 다름). 128 부근에 원인 불명의 국소적 dip이 있고, 73에서 다시
  약간 회복하는, 매끄럽지 않은 곡선. 4개 지점만으로는 이 dip이 노이즈인지 구조적 패턴인지 판단 어려움.
- 종합: **context_length=512, prediction_length=4가 지금까지 실험한 조합 중 최고 성능** — 이후
  실험/배포 기본값으로 확정. 512 미만 구간의 정확한 형태(단조적인지 dip이 있는지)는 미해결.

체크포인트: `checkpoints/{blaine,residue}_predlen4_ctx{168,73,128}`, `_full_predlen4_ctx{168,73,128}`.

### 다음에 시도해볼 것 (2026-08-18 기준, 최신)
1. context_length 256/384 등 512 근처 추가 스윕 — 512-168 사이 급격한 개선 구간의 정확한 위치 확인
2. context_length=128 부근(96/160 등) dip의 원인 확인 — 노이즈인지 실제 구조적 패턴인지
3. `train/finetune_chronos2.py --prediction-length` 기본값과 `README.md`를 4로 갱신(위 미반영 TODO)
4. 80% 구간 커버리지 문제 — prediction_length=4로 상당히 개선됐지만(ctx512 기준 48~51%) 여전히 목표(80%)엔
   못 미침, 별도 조사 필요
5. `--finetune-mode full`이 LoRA를 계속 앞서고 있음 — LoRA rank/alpha 등 하이퍼파라미터 조정으로 격차를
   좁힐 수 있는지 확인 (아직 미착수)

---

## 2026-08-28 — IQR 이상치 제거 유무 A/B (blaine=ctx1024, residue=ctx1536, predlen4, full) → **IQR 제거 안 함**

**배경**: context_length 스윕 결론으로 타깃별 최적값 확정(사용자 2026-08-28: blaine 1024, residue 1536).
그 위에서 `remove_iqr_outliers()`(호기별·컬럼별 Tukey 1.5·IQR 펜스, 타깃 포함)를 켠 CSV와 끈 CSV로
각각 full fine-tuning. 하이퍼파라미터는 `run_ctx_sweep_v2.sh`와 100% 동일(1000 steps / lr 1e-6 /
batch 64) — 두 arm의 유일한 차이는 입력 CSV. 실행: `train/run_iqr_experiment.sh`.

**평가 설계**: 4개 모델을 각각 두 테스트셋에서 backtest(같은 테스트셋 = 같은 평가 지점 = naive/
skill-score 분모 동일 = 공정 비교).
- `raw` = without_iqr 테스트셋(극단값 포함) = 배포 현실, 주 지표
- `clean` = with_iqr 테스트셋(IQR로 정제된 정상운전 구간)

**데이터 레벨**: IQR이 feature 셀의 2.92%(117,672개) + 실측 blaine 922개(2.3%)·residue 956개(2.3%)를
NaN 처리. 최다 제거 컬럼은 feed_slag(29k, SLAG 미투입 배치가 통계적으로 이상치로 잡힘).

**결과 — raw 테스트셋 (Full fine-tuning, pooled)**:

| target | variant | MAE | R² | Spec 정확도 | 80% 커버리지 | skill-score |
|---|---|---|---|---|---|---|
| blaine  | IQR 제거   | 69.88 | 0.4755 | 70.34% | 45.51% | 1.0085 |
| blaine  | IQR 미제거 | **69.14** | **0.4893** | 70.05% | 44.64% | **0.9973** |
| residue | IQR 제거   | 0.4119 | 0.8642 | 82.72% | 52.51% | 1.0426 |
| residue | IQR 미제거 | **0.4084** | **0.8658** | **83.01%** | 49.81% | **1.0340** |

(clean 테스트셋도 결론 동일: 미제거가 MAE·R²·skill-score 우세, 제거가 Spec·커버리지 근소 우세)

**결론**:
- **IQR 이상치 제거는 하지 않는다.** raw·clean 두 테스트셋 모두에서 IQR 미제거가 점 예측 지표
  (MAE / R² / skill-score) 전부 우세. blaine·residue 각각 -1.1% / -0.9% MAE 개선.
- **blaine이 이 프로젝트에서 처음으로 naive baseline을 이김** (skill-score 0.997 < 1.0). IQR-off
  개선폭(1.0085→0.9973, Δ0.011)이 재학습 노이즈(ctx1024 재현시 1.0048 vs 1.0085, Δ0.004)보다 큼.
- IQR 제거가 근소하게 나은 건 Spec 정확도(blaine만)·80% 커버리지뿐인데 둘 다 절대 수준이 낮아
  (커버리지 0.45~0.53, 목표 0.80) 의미 없음.
- 이유: Chronos-2는 NaN을 네이티브 마스킹하므로 극단값을 "정리"한다고 지우면 학습 신호 순손실.
  물리적으로 불가능한 값 제거(`clean()`의 고정 임계값 규칙)는 유지, 통계적 IQR 제거만 뺌.

**한계**: arm당 단일 run, 차이 ~1%. 방향은 4개 조합에서 일관. 확실히 하려면 blaine(~17분/run) seed 재확인.

체크포인트: `checkpoints/{blaine_ctx1024,residue_ctx1536}_full_predlen4_{withiqr,withoutiqr}` (→ `_full_predlen4_ctx{N}_{withiqr,withoutiqr}`).
backtest: `eval/backtest_{target}_{withiqr,withoutiqr}_evalon_{raw,clean}.csv`.
결과 노트북: `notebooks/01_iqr_outlier_removal/results_iqr_comparison.ipynb`.

### 다음에 시도해볼 것 (2026-08-28 기준)
1. 정식 파이프라인을 IQR-off로 전환 — `data/build_dataset.py`의 `main()`에서 `remove_iqr_outliers()`
   호출 제거(또는 학습/평가 기본 CSV를 `without_iqr/`로), config의 `IQR_OUTLIER_K` 등 관련 코드 정리
2. (선택) blaine IQR-off를 seed 2~3개로 재확인 — skill-score < 1.0이 재현되는지
3. 80% 구간 커버리지 문제(전 실험 공통, 0.45~0.53) — conformal 보정 / quantile-loss 가중 등 별도 조사
4. 512 미만 context_length 구간의 비단조 패턴(128 근처 dip)
5. `--finetune-mode full` vs LoRA 격차 — LoRA rank/alpha 튜닝 여지

---

## 2026-08-31 — 물리 정제 규칙 확장 + 정식 파이프라인 IQR-off 전환

08-28 "다음에 시도해볼 것" 1번 실행.

- `config.py`의 `CLEANING_RULES`에 센서 고장 sentinel/설비 한계 규칙 추가(`mill_out_gas_temp` 5000,
  `mill_out_mater_temp` 5000, `RP_roller_p2` 20000, damper 하한 0). 각 컷오프는 "flagged 값과 그다음
  정상값 사이 큰 갭"을 손으로 확인해 결정(주석에 근거 기록). 애매한 중간 꼬리(gas/mater temp 1100~1400
  클러스터, mater temp 음수, blaine 자체 min/max)는 현장 확인 전까지 보류.
- `data/build_dataset.py` `main()`을 IQR-off로 전환(통계적 `remove_iqr_outliers()` 미호출). `clean()`의
  고정 임계값 규칙만 적용. canonical CSV 재생성(`data/processed/quality_timeseries.csv`, 구버전은
  `.bak_before_20260831_cleanup`).
- 확장된 물리 정제 데이터 위에서 IQR on/off를 다시 A/B(blaine ctx1024, residue ctx1536, full/predlen4).
  결과: 08-28 결론 그대로 재현 — 두 타깃 모두 IQR 미적용이 MAE/R²/skill-score 우세, blaine은 IQR을
  껐을 때만 naive를 이김(skill 0.9965 < 1.0). 결과 노트북:
  `01_iqr_outlier_removal/results_iqr_comparison_cleaned.ipynb`.

---

## 2026-09-02 — 물리적 이상치 처리 방식 A/B 실험 스캐폴드 (⚠️ 아직 미실행)

IQR을 뺀 뒤 남은 축("물리적 이상치를 NaN으로 날릴지 / 실제 값을 살릴지")을 3-way로 비교하기 위한
코드/스크립트만 먼저 준비. 학습은 GPU 머신에서 별도 실행 예정.

- `clean()` / `prepare_base()`에 인자 2개 추가(둘 다 기본값 = 현행 동작이라 정식 파이프라인 무변화):
  - `damper_oob="nan"|"clip"` — 댐퍼 개도율 경계 초과값을 NaN(현행) vs `DAMPER_CLIP_MARGIN_PCT`
    (config, 기본 20%p) 이내면 경계값으로 clip
  - `prodtype_mode="drop"|"mask_target"` — 비-내수 행을 삭제(현행) vs 유지하고 품질 타깃만 마스킹
- 3개 분기 빌드 스크립트 + 실행 스크립트 + 결과 노트북 스캐폴드:
  `02_physical_outlier_handling/{build_baseline,build_expA_damper_clip,build_expB_prodtype_keep}.py`,
  `train/run_phys_outlier_experiment.sh`, `02_physical_outlier_handling/results_phys_outlier_comparison.ipynb`.
  세 분기 모두 행 수/타임스탬프/타깃값/분할 지점 동일 → backtest 평가 지점 동일 → 공정 skill-score 비교.
- 편집 전 백업: `config.py.bak_20260902`, `data/build_dataset.py.bak_20260902`,
  `data/processed/quality_timeseries.csv.bak_20260902_before_phys_outlier_exp`.

### 실행 전 확인 / 판단 기준
1. **expB 사전 조사**: 비-내수 행 실제 비율 + 그 시각 품질 실측 유무. < 5%면 expB 스킵 가능.
   (`build_baseline.py` vs `build_expB_prodtype_keep.py` 실행 로그의 행 수/타깃 null rate 비교)
2. 3개 분기 학습·평가(blaine ctx1024, residue ctx1536, full/predlen4/1000steps/lr1e-6/batch64) 후
   normalized skill-score 비교. 개선폭이 재학습 노이즈(blaine ~0.004)보다 작으면 "차이 없음" → baseline 유지.
3. 유의미하면 seed 2~3개 재확인 → 가장 좋은 분기를 정식 파이프라인 기본값으로 채택, 이 로그에 결과 기록.

### 그대로 남은 이전 TODO
- 80% 구간 커버리지(0.45~0.53) — conformal / quantile-loss 가중
- 512 미만 context_length 비단조 패턴(128 근처 dip)
- full vs LoRA 격차 — LoRA rank/alpha 튜닝
- 보류된 애매한 물리 꼬리(gas/mater temp, blaine min/max) — 현장 엔지니어 확인 과제

---

## 2026-09-02 — `_prev` covariate 밀도 A/B 실험 (데이터 빌드·누출 감사 완료, ⚠️ 학습만 미실행)

`blaine_prev`/`residue_prev`를 실측 행에만 넣을지(**sparse**, 현행) vs 직전 실측값을 다음 측정
시점까지 forward-fill할지(**ffill**, dense)를 2-way 비교. sparse는 2026-08-12에 "매 시간 반복
노출되면 naive persistence로 수렴 위험"이라는 추론으로 정했을 뿐 직접 A/B로 확인한 적 없음 —
이번에 확인. 참고 노트북(`chronos-2-cement_share.ipynb`)의 dense는 backward-fill이었는데, bfill은
horizon이 측정 경계를 넘을 때 미관측 미래 실측값을 known-future covariate로 노출(누출)하므로
dense 분기는 **ffill**로 함.

- `data/build_dataset.py` `finalize_and_save()`에 `prev_density="sparse"|"ffill"` 인자 추가(기본값
  `"sparse"` = 현행 동작이라 정식 파이프라인 무변화). `"ffill"`은 시간 그리드 재색인 후
  `df.groupby("item_id")[PREV_QUALITY_COLS].ffill()` 한 줄.
- 파일: `03_prev_density_handling/run_and_compare_prev_density.ipynb`(**메인** — 빌드→감사→학습→비교
  올인원, 학습/평가는 정식 스크립트를 subprocess 호출), `{build_sparse,build_dense_ffill}.py`,
  `smoketest_prev_density.py`, `train/run_prev_density_experiment.sh`,
  `results_prev_density_comparison.ipynb`, `README.md`.

### 누출 감사 결과 (2026-09-02, 실데이터 51,209행 × 3호기 × 2타깃 — 노트북 2절)
- **D**: 두 arm 은 `_prev` 2컬럼 말고 모든 값(feature/타깃/타임스탬프) 동일.
- **B**: ffill 컬럼 == sparse 컬럼을 item_id별 forward-fill 한 것. sparse 의 non-null 값은 보존.
- **A (핵심)**: merge_asof 로 각 ffill `_prev[t]` 의 출처 실측 timestamp 를 역추적 → 모든 행에서
  `t − 출처 ≥ 4h` (min 4.0h, median 7.0h). scheduled `_prev`가 이미 4h 전 값이라 forward-fill
  해도 어떤 행이든 ≥4h 전 값만 노출. **predlen ≤ 4 에서 누출 0건.** (predlen 늘리면 재검증 필요.)
- **C**: `make_eval_tasks`로 실제 평가 태스크 2,071개 생성 → 평가 지점(timestamp/actual)·컨텍스트
  타깃·`naive_pred`(skill-score 분모)·제어 covariate 가 두 arm 완전 동일. known-future `_prev`는
  505개 (태스크×컬럼)에서 sparse=NaN vs ffill=값 으로 다름 = **측정 대상 treatment**, 누출은 A로 배제.
- 빌드된 CSV: `data/processed/prev_sparse/`, `data/processed/prev_dense_ffill/` (`_prev` null rate 0.82 vs ~0.00).

### 결과 (2026-09-02 실행, 이 머신 RTX 5060, full/predlen4/1000steps/lr1e-6/batch64)

| target | arm | MAE | R² | Spec 정확도 | 80% 커버리지 | skill-score |
|---|---|---|---|---|---|---|
| blaine  | sparse(현행) | 69.08 | 0.490 | 70.14% | 44.44% | 0.9965 |
| blaine  | **ffill**    | **67.51** | 0.492 | 70.72% | 41.06% | **0.9733** |
| residue | sparse(현행) | 0.4089 | 0.866 | 83.30% | 50.00% | 1.0353 |
| residue | **ffill**    | **0.3987** | 0.869 | 83.11% | 47.30% | **1.0089** |

- **dense ffill 이 두 타깃 모두에서 유의미하게 나음.** Δskill(ffill−sparse) = blaine −0.023 / residue −0.026
  → 재학습 노이즈(~0.004)의 약 6배, 방향·크기 일관. MAE −2.3% / −2.5%.
- blaine 은 그동안 skill 0.9965 로 naive 를 겨우 이기던 상태였는데 ffill 로 0.9733 — 확실히 벗어남.
  residue 는 여전히 naive 미달(>1.0)이나 1.0353→1.0089 로 크게 좁힘.
- 80% 커버리지만 소폭 하락(blaine 44→41, residue 50→47). 절대 수준이 목표(0.80)에서 멀어 노이즈 취급.
- 해석: 2026-08-12 에 sparse 를 고른 근거("매 시간 반복 노출 → naive persistence 로 붕괴 위험")는
  기우였음. `fresh_reading_only` 평가라 "복사" 이득이 걸러진 상태에서 나온 개선.
- 산출물: `eval/backtest_{blaine,residue}_prevdensity_{sparse,ffill}.csv`,
  `checkpoints/{target}_full_predlen4_ctx{1024,1536}_prevdensity_{sparse,ffill}/`,
  실행된 노트북 `03_prev_density_handling/run_and_compare_prev_density.ipynb`(표·그래프·6절 결론 포함).

### 남은 단계
1. seed 2~3개로 재확인 (blaine ~18분/run, residue ~90분/run — 이 머신 기준).
2. 재현되면 정식 파이프라인 기본값을 `prev_density="ffill"` 로 전환: `data/build_dataset.py` `main()`,
   `data/dataset_utils.py`·`config.py` 주석, 01/02 실험 build 스크립트도 함께.

---

## 2026-09-05 — 신규 소스 `운전&품질데이터_실측치.xlsx`로 ctx1024(blaine)/ctx1536(residue) 재실행

Cement 폴더에 새 원본 xlsx가 추가됨. 기존 `SOURCE_XLSX`("운전 & 품질 데이터_2026.xlsx")와 시트 구성·
행 수는 완전히 동일(2CM/3CM/4CM 각 39854/40071/39421행)하지만, 셀 단위 diff 결과 2025-01-01~
2025-11-17 구간에서 품질 실측 시각(4시간 간격: 00/04/08/12/16/20시)이 **아닌** 시간대의 `grind_aid`/
`blaine`/`residue`/`ProdType`/`Remark` 값이 기존 파일엔 채워져 있던 반면(원인 불명 — carry-forward성
스며듦으로 추정) 신규 파일엔 비어 있음(2CM/3CM/4CM 각 4771/4715/4601행 차이, 그중 85%+가 비측정
시각 값을 NaN으로 되돌린 것). 실측 시각 자체도 소수 정정됨(2CM 기준 507건 중 486건 값 삭제, 21건
값 자체 수정). 즉 신규 파일은 이름 그대로 "진짜 실측치만" 남긴 버전.

`data/build_dataset.py`의 전처리 로직은 전혀 건드리지 않고 `config.SOURCE_XLSX`만 이 신규 파일로
바꿔(`04_measured_actual_data/build_measured_actual.py`) processed CSV를 재생성
(`data/processed/measured_actual/quality_timeseries.csv`, 153,627행 — 행 수/타임스탬프/분할 지점은
기존과 동일, target null rate만 73.5%→82.6%로 정직하게 증가). 그 위에서 2026-08-28 확정 최적
context_length(blaine=1024, residue=1536)를 zero-shot/LoRA/Full 3가지 방식 전부, 기존과 동일
하이퍼파라미터(predlen=4, num_steps=1000, batch=64; LoRA lr=1e-5 기본값, Full lr=1e-6)로 재실행
(`train/run_measured_actual_experiment.sh`, `eval/run_measured_actual_backtest.sh`). 체크포인트/결과
파일은 전부 `_measured` 태그로 분리 저장 — 기존 파일은 하나도 덮어쓰지 않음.

**결과 (신규 실측치 데이터, n=852 blaine / 856 residue)**:

| target (ctx) | 방식 | MAE | R² | spec 정확도 | 80% 커버리지 | skill-score |
|---|---|---|---|---|---|---|
| blaine (1024) | Naive | 83.76 | 0.163 | 67.02% | — | — |
| blaine (1024) | zero-shot | 75.51 | 0.390 | 67.84% | 61.97% | 0.901 |
| blaine (1024) | LoRA | 70.39 | 0.465 | 69.48% | 66.67% | 0.840 |
| blaine (1024) | Full | 69.47 | 0.479 | 71.01% | 66.08% | 0.829 |
| residue (1536) | Naive | 0.568 | 0.578 | 78.50% | — | — |
| residue (1536) | zero-shot | 0.554 | 0.712 | 78.27% | 49.65% | 0.974 |
| residue (1536) | LoRA | 0.501 | 0.741 | 81.07% | 55.61% | 0.881 |
| residue (1536) | Full | 0.486 | 0.754 | 81.78% | 67.52% | 0.855 |

**핵심 발견**: 이 데이터에서는 **zero-shot조차 두 타깃 모두 naive baseline을 이김**(skill-score 전부
< 1.0). 기존 xlsx로는 같은 ctx1024/1536 설정에서 6개 조합 전부 skill 1.005~1.03(naive에 근소하게
뒤지거나 겨우 따라잡는 수준)이었던 것과 뚜렷이 대비됨. naive MAE 자체가 신규 데이터에서 크게
나빠짐(blaine 63.21→83.76, residue 0.415→0.568) — 비실측 시간대에 섞여 있던 값들이 naive에게
유리한 "쉬운" 전환 구간을 인위적으로 늘리고 있었던 것으로 추정. 80% 구간 커버리지도 6개 조합 전부
크게 개선(Full 기준 blaine 50%→66%, residue 51%→68%, 여전히 목표 80% 미달이지만 지금까지 최고치).

**주의(미통제 변수)**: "기존" 비교값은 2026-08-19/21 결과 파일로, 이후(08-31) 물리적 정제 규칙
확장·`prev_density` 기본값 등 파이프라인도 소폭 바뀐 뒤라 이 비교는 "xlsx 소스 교체 효과"만 순수
분리한 게 아니라 그 사이 파이프라인 개선분도 섞여 있음. 완전 통제 비교는 미실행(현재 canonical
파이프라인 + 기존 xlsx로 ctx1024/1536 재실행 필요, 추가 GPU ~2-3시간).

**결론**: 기존 실험들의 "Chronos-2가 naive를 잘 못 이긴다"는 결론 상당수가 데이터 오염(비실측
시간대 값 스밈) 자체의 영향이었을 가능성이 있음 — 정식 파이프라인의 `SOURCE_XLSX`를 이 실측치
파일로 교체하는 것을 다음 우선순위로 검토할 가치가 있음.

산출물: `04_measured_actual_data/build_measured_actual.py`, `data/processed/measured_actual/
quality_timeseries.csv`, `train/run_measured_actual_experiment.sh`,
`eval/run_measured_actual_backtest.sh`, `eval/backtest_{blaine,residue}_predlen4_ctx{1024,1536}_
{zeroshot,lora,full}_measured.csv`, 체크포인트 `checkpoints/{blaine,residue}_predlen4_ctx{1024,1536}_
{lora,full}_measured/`, 결과 노트북 `results_measured_actual.ipynb`.

### 다음에 시도해볼 것 (완료 항목은 아래 통제 비교 참고)
1. ~~통제 비교~~ → 아래 항목에서 완료.
2. 재현되면 정식 파이프라인의 `SOURCE_XLSX`를 이 실측치 파일로 전환.
3. `03_prev_density_handling`의 ffill A/B를 이 실측치 데이터로도 재확인(값 밀도 자체가 바뀌었으므로
   결론이 달라질 가능성).

---

## 2026-09-05 (이어서) — 통제 비교: 현재 파이프라인 + 기존 xlsx로 ctx1024/1536 재실행 → **xlsx 교체 효과 확정**

위 항목의 "미통제 변수" 캐비엇을 해소하기 위해, **현재 canonical 파이프라인**(`data/processed/
quality_timeseries.csv`, 08-31 물리 정제 규칙 확장판 그대로, 코드 변경 없음)** + 기존 xlsx**로
ctx1024(blaine)/ctx1536(residue) zero-shot/LoRA/Full을 처음부터 다시 재현
(`train/run_oldxlsx_ctrl_experiment.sh`, `eval/run_oldxlsx_ctrl_backtest.sh`, `_oldxlsx` 태그).
이제 "통제-기존xlsx"(이 결과) vs "신규-실측치"(위 결과)는 **파이프라인 코드가 100% 동일하고
`config.SOURCE_XLSX` 하나만 다르므로**, 둘의 차이가 xlsx 소스 교체만의 순수한 효과다.

**결과 (통제-기존xlsx, n=1035 blaine / 1036 residue — 위 실측치 결과의 852/856과 다름에 주의,
같은 canonical 파이프라인이라도 xlsx가 다르면 "fresh-reading" 이벤트 자체가 달라짐)**:

| target (ctx) | 방식 | MAE | R² | spec 정확도 | 80% 커버리지 | skill-score |
|---|---|---|---|---|---|---|
| blaine (1024) | Naive | 69.30 | 0.400 | 71.11% | — | — |
| blaine (1024) | zero-shot | 71.18 | 0.449 | 70.43% | 37.00% | 1.027 |
| blaine (1024) | LoRA | 71.04 | 0.452 | 70.24% | 36.91% | 1.025 |
| blaine (1024) | Full | 69.08 | 0.490 | 70.14% | 44.44% | 0.997 |
| residue (1536) | Naive | 0.397 | 0.848 | 83.40% | — | — |
| residue (1536) | zero-shot | 0.416 | 0.858 | 81.18% | 40.64% | 1.048 |
| residue (1536) | LoRA | 0.413 | 0.862 | 81.95% | 42.86% | 1.041 |
| residue (1536) | Full | 0.409 | 0.866 | 83.30% | 50.00% | 1.029 |

(참고: blaine Full MAE=69.08/skill=0.997은 2026-08-28 IQR-off 실험의 69.14/0.9973과 거의 동일 —
같은 canonical CSV를 쓰는 서로 다른 run이 잘 재현됨을 확인.)

**xlsx 교체만의 순수한 효과 (통제-기존xlsx → 신규-실측치, skill-score)**:

| target | 방식 | skill 통제 | skill 신규 | 변화 |
|---|---|---|---|---|
| blaine | zero-shot | 1.027 | 0.901 | -0.126 |
| blaine | LoRA | 1.025 | 0.840 | -0.185 |
| blaine | Full | 0.997 | 0.829 | -0.168 |
| residue | zero-shot | 1.048 | 0.974 | -0.074 |
| residue | LoRA | 1.041 | 0.881 | -0.160 |
| residue | Full | 1.029 | 0.855 | -0.174 |

**결론(확정)**: 6개 조합 전부, 파이프라인을 고정한 채 xlsx만 실측치로 바꾸면 skill-score가 뚜렷하게
낮아진다(0.07~0.19 하락, 2026-08-28 IQR 실험의 재학습 노이즈 ~0.004~0.006보다 한 자릿수 이상 큼 —
노이즈로 설명 안 되는 확실한 효과). **통제-기존xlsx 조건에서는 6개 조합 중 blaine Full 하나만
naive를 근소하게 이기고(skill 0.997) 나머지 5개는 전부 naive에 졌던 반면, 신규-실측치에서는 6개
전부 확실하게 이긴다(skill 0.83~0.97).** 즉 이 프로젝트의 여러 실험(2026-08-04 첫 파인튜닝부터
2026-09-02 prev-density까지)에서 반복적으로 나온 "Chronos-2가 naive baseline을 잘 못 이긴다"는
결론의 상당 부분이, 비실측 시간대에 값이 스며 있던 **데이터 오염 자체의 인공물이었다**는 게 이번
통제 비교로 확인됨(정황이 아니라 확정).

**다음 우선순위로 격상**: 정식 파이프라인의 `SOURCE_XLSX`를 이 실측치 파일로 전환하고, 그 위에서
지금까지의 주요 A/B 실험들(IQR on/off, prev-density sparse/ffill, 물리적 이상치 처리, context_length
스윕 전체)을 재검증해야 함 — 이번 결과로 볼 때 결론이 바뀔 실험이 있을 가능성이 있음.

**한계**: 각 조합 단일 run(시드 고정 안 함).

산출물: `train/run_oldxlsx_ctrl_experiment.sh`, `eval/run_oldxlsx_ctrl_backtest.sh`,
`eval/backtest_{blaine,residue}_predlen4_ctx{1024,1536}_{zeroshot,lora,full}_oldxlsx.csv`,
체크포인트 `checkpoints/{blaine,residue}_predlen4_ctx{1024,1536}_{lora,full}_oldxlsx/`. 3묶음
비교 노트북: `results_measured_actual.ipynb`(갱신 — 구버전/통제-기존xlsx/신규-실측치 3묶음 전부 포함).

### 다음에 시도해볼 것
1. 정식 파이프라인의 `SOURCE_XLSX`를 실측치 파일로 전환(`config.py`).
2. 전환 후 기존 A/B 실험(IQR, prev-density, 물리적 이상치 처리, context_length 스윕)을 실측치
   데이터로 재검증 — 결론이 바뀌는지 확인.
3. `03_prev_density_handling`의 ffill A/B를 이 실측치 데이터로도 재확인(값 밀도 자체가 바뀌었으므로
   결론이 달라질 가능성).
