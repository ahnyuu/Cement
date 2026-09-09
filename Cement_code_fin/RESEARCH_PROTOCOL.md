# 검증·평가 수정본 사용 안내 — 2026-09-09

이번 수정은 모델 선택과 평가의 신뢰성을 개선합니다. 성능 향상을 입증한 결과가 아니며, 재학습은 아직 수행하지 않았습니다. 기존 실험은 탐색 기록으로 보존합니다.

## 변경한 내용

1. 학습 중 검증: 호기별 검증 시계열 끝의 한 구간 대신, 여러 날짜에 분산한 최대 128개 구간을 사용합니다. 구간마다 실제 품질 정답이 최소 1개 있어야 합니다. 세 호기에서 최대 384개 구간입니다. 128은 계산량을 제한하기 위한 기본값이며 최적값이라는 뜻이 아닙니다. `--validation-windows-per-item 0`이면 조건을 충족하는 구간을 모두 사용합니다.
2. 검증 구간은 예측 시작부터 학습 `prediction_length`만큼의 시간입니다. 그 이전 학습 데이터와 이미 지나간 검증 데이터는 과거 입력으로 사용할 수 있습니다. 시험 데이터는 포함하지 않습니다. 결측치를 보간하거나 시간축을 압축하지 않습니다.
3. 정답이 없는 검증을 차단합니다. 실제 Chronos 입력으로 변환한 후에도 마지막 예측 구간의 정답 존재 여부를 확인합니다. `validation_windows.csv`에 구간별 시각과 정답 수를 기록합니다.
4. 학습 중 체크포인트 선택은 공식 `eval_loss`를 사용합니다. 이는 MAE가 아닙니다. 학습 길이 1과 4의 loss 절댓값을 직접 비교해 우열을 정하지 말고, 별도의 동일한 1시간 예측 MAE로 비교합니다.
5. 모든 파라미터 실험 실행기는 `validation`을 평가합니다. `test`는 `--split test`로 명시할 때만 평가합니다. 이미 살펴본 과거 시험 구간은 코드 변경 후에도 미사용 데이터가 되지 않습니다.
6. 실제 측정값이 직전 값과 같아도 평가합니다. 기본 `stride=4`는 측정 후보 네 개마다 하나를 고르며, 4시간마다라는 뜻이 아닙니다. 전체 측정 후보를 평가하려면 `--stride 1`을 사용하되 비교하는 모든 모델에 동일하게 적용합니다.
7. 평가 시점 선택에 공통으로 최근 24시간의 품질 관측 존재 여부를 사용합니다. 따라서 context 24~1536 비교에서 같은 시점과 같은 naive 기준을 사용합니다. 24는 기존 context 연구의 최솟값입니다. 더 짧은 context 연구에서는 `--eligibility-context-length`를 모든 비교 조건에서 같은 값으로 낮춰야 합니다. 최근 24시간에 관측이 없는 사례는 평가에서 제외하고 `.coverage.csv`에 남깁니다. 이 성능을 장기 결측 상황까지 포함한 전체 공정 성능으로 해석하면 안 됩니다.
8. 새 체크포인트·평가 파일·로그는 `review_v2` 이름으로 분리합니다. 기존 모델과 시험 결과를 새 결과로 자동 재사용하지 않습니다. 새 결과를 덮어쓰지 않으며 중단된 학습은 완료된 모델로 취급하지 않습니다.
9. 학습 seed 기본값은 42이며 데이터 파일 해시, 실행 인수, 패키지 버전을 기록합니다. 같은 seed가 모든 장치에서 비트 단위 동일성을 보장하지는 않습니다. 반복 실험 시 새 output-tag와 seed를 사용합니다.
10. IQR 실험을 재생성하면 실제 학습/검증 분할과 같은 **시간 격자상의 학습 구간에서만** 경계를 계산합니다. 입력 변수에는 고정 경계를 적용하되 검증·시험의 품질 정답은 삭제하지 않습니다. IQR 실험 실행기는 두 모델 모두 같은 IQR 미적용 검증 입력·정답으로 평가합니다. 이는 학습 전처리의 효과를 같은 입력 조건에서 비교하는 설계입니다. 과거의 별도 `evalon_clean` 결과는 새 실험에 포함하지 않습니다.

## 유지한 내용과 아직 필요한 확인

- 학습 길이 4, 최종 예측 길이 1, batch size 64, 원래의 학습 구간 추출 방식, 결측치 유지, 기본 모델의 IQR 미적용은 유지했습니다.
- 원본 엑셀, 기본 processed CSV, ANN 코드, 예전 모델과 예전 결과 파일은 변경하지 않았습니다.
- 이전 품질값은 기존의 **정규 4시간 측정 시각 사이에서 shift한 sparse 변수**입니다. 비정규 측정까지 포함한 직전 이용 가능 측정값으로 바꾼 것은 아닙니다. ffill 실험도 이 기존 변수의 밀도 비교이며, 생성 규칙의 정합성을 검증하는 실험은 아닙니다.
- 실제 예측 시각에 제어값과 품질 분석 결과를 알고 있는지, 엑셀 시각이 채취 시각인지 분석 완료 시각인지 확인해야 합니다. 이 정보 없이 입력의 현장 가용성을 확정할 수 없습니다.
- 매시간 출력은 가능하지만, 현재 데이터로 직접 성능을 검증할 수 있는 것은 품질 실측 시점입니다.
- 기존 test로 골랐던 context 값은 탐색 결과입니다. 수정본의 검증 결과로 다시 선택한 뒤 가능하면 새 미사용 기간에서 최종 확인해야 합니다. 미사용 기간이 없으면 평가 설계의 한계를 논문에 명시해야 합니다.
- 기존 보고서 생성용 노트북과 `build_results_notebook.py` 계열은 과거 결과를 읽습니다. 새 결과는 아래의 `summarize_validation.py`로 정리합니다. 과거와 새 결과를 한 표에 합칠 때는 검증 방식과 평가 구간 차이를 표시해야 합니다.

## 실행 방법

아래 명령은 `Cement_code_fin` 폴더에서 기존 Chronos-2 학습 환경을 활성화한 뒤 실행합니다. 학습 패키지를 설치하지 않아도 pandas/numpy가 있으면 사전 점검을 실행할 수 있습니다.

### 1. 학습 없이 데이터와 검증 구성 점검

```text
python train/finetune_chronos2.py --target blaine --prediction-length 4 --dry-run
python train/finetune_chronos2.py --target residue --prediction-length 1 --dry-run
python eval/backtest.py --target blaine --split validation --context-length 24 --dry-run
python -m unittest discover -s tests -v
```

### 2. 대표 설정 하나 재학습 후 검증 MAE 확인

아래 context 512는 실행 예시이며 최적값을 확정한 것이 아닙니다.

```text
python train/finetune_chronos2.py --target blaine --context-length 512 --output-tag ctx512_check
python eval/backtest.py --target blaine --checkpoint checkpoints/review_v2/blaine_ctx512_check/final --context-length 512 --split validation --tag ctx512_check
```

### 3. 파라미터 실험

기존 실행기 이름을 그대로 사용할 수 있습니다. 실행하면 많은 모델을 학습하므로 먼저 대표 설정에서 실행 환경을 확인합니다.

```text
python experiments/run_ctx_matrix.py
python experiments/predlen/run_predlen_experiment.py
python experiments/prev/run_prev_experiment.py
python experiments/prev_density/run_prev_density_experiment.py
```

이전 품질값 밀도 실험에는 기존 `prev_sparse`, `prev_ffill` CSV를 사용할 수 있습니다. IQR 실험에는 과거 IQR CSV를 재사용하지 않고 새 변형 파일을 만들어야 합니다. 원본 엑셀이 프로젝트 상위 폴더에 없다면 `CHRONOS_SOURCE_XLSX` 환경 변수에 실제 파일 경로를 지정합니다.

```text
python experiments/iqr/build_variants.py
python experiments/iqr/run_iqr_experiment.py
```

### 4. 검증 결과 비교

```text
python experiments/summarize_validation.py --pattern "backtest_*_ctx*_full.csv" --metric MAE
```

선택 기준 기본값은 사용자가 확인 중인 전체 MAE입니다. 호기별 동등 가중 MAE 또는 naive 대비 비율도 선택할 수 있으나, 논문에서 기준을 먼저 정하고 결과에 따라 바꾸지 않습니다. 비교하는 결과의 시점·정답·naive가 다르면 이 도구는 순위를 만들지 않습니다. 패턴에 해당하는 **완료된 결과들 사이에서만** 선택하므로, 실험 행렬이 모두 끝났는지도 확인해야 합니다.

### 5. 선택을 마친 후 과거 test 재평가

```text
python eval/backtest.py --target blaine --checkpoint checkpoints/review_v2/blaine_ctx512_check/final --context-length 512 --split test --tag ctx512_check
```

위 명령은 기존 기간의 재평가입니다. 논문용 미사용 최종 평가 기간을 새로 마련하는 작업을 대신하지 않습니다. 새 기간의 데이터 수집·분할 계획은 별도로 정해야 합니다.

## 저장 위치와 재실행

- 모델 및 검증 구간 목록: `checkpoints/review_v2/{target}_{tag}/`
- 검증 결과: `eval/review_v2/validation/`
- 기존 시험 기간 재평가: `eval/review_v2/test/`
- 새 IQR 변형 데이터: `data/processed/review_v2/`
- 새 실험 로그: 각 실험 폴더의 `*_review_v2.log`

실험 실행기는 완료된 수정본 결과를 재사용합니다. 데이터·파라미터·검증 규칙을 다시 바꾼 경우 반드시 새로운 tag/출력 위치를 사용해야 합니다. 중단된 실행이 남아 있으면 보존 후 다른 tag로 재실행합니다.

## 확인 범위

데이터 처리·검증 구간 구성·평가 시점 생성과 기본 데이터 재생성은 실제 제공 데이터로 점검했습니다. 현재 점검 환경에는 torch/Chronos가 없으므로 실제 모델 로딩·GPU 재학습·MAE 개선 여부는 아직 검증하지 않았습니다. 공식 Chronos-2 v2.3.1의 `from_data_frame`, 검증 마지막 구간 분리, `fit` 인수를 기준으로 작성했으며, 실제 실행 패키지 버전은 학습 시 기록합니다.

- [공식 검증 구간 처리](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.1/src/chronos/chronos2/dataset.py)
- [공식 학습·검증 API](https://github.com/amazon-science/chronos-forecasting/blob/v2.3.1/src/chronos/chronos2/pipeline.py)
