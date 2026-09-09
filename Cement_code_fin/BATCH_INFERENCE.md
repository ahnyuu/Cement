# 시점별 예측을 묶어 실행하기

현재 backtest.py에 공유 노트북의 독립 윈도우 배치 추론 방식을 추가했습니다.
기본값은 기존처럼 한 시점씩 처리합니다. 아래 옵션을 넣으면 8개 시점을 묶습니다.
이는 예측 길이, 학습 설정, 평가 날짜, 정답, 이전 품질값 생성 규칙을 바꾸지 않습니다.

## 원리

각 예측 요청은 해당 시점 이전의 과거 데이터와 예측 시점의 허용된 입력으로 구성됩니다.
각 요청에 임시 ID를 붙여 묶고, 모델이 반환한 ID와 시각을 확인해 원래 호기·날짜로 복원합니다.
여러 요청을 같이 계산해도 각 요청은 여전히 다음 1시간만 예측합니다.
cross_learning=False를 명시해 요청 사이의 정보 공유를 차단합니다. 서로 다른 날짜를 묶은
backtest에서 이 옵션을 켜면 나중 시점의 기록이 앞선 예측에 노출될 수 있습니다.

공유 코드가 전체 평가 구간을 한 번에 합치는 것과 달리 이번 코드는 정해진 개수씩 합칩니다.
임시 병합 데이터의 크기를 제한하기 위함입니다. 원래 backtest의 전체 task 목록 생성은 유지합니다.
공유 코드의 cuDF/CuPy 후처리는 도입하지 않았습니다. 핵심 변경은 predict_df 호출을 묶는 것입니다.

## 실행

Cement_code_fin 폴더에서 아래 형태로 실행합니다. YOUR_CHECKPOINT는 실제 학습 가중치 폴더로
바꿔야 합니다. 현재 다운로드한 자료에는 그 가중치가 확인되지 않아 실행 예시는 아직 수행하지 않았습니다.

```powershell
python eval/backtest.py --target blaine --checkpoint "YOUR_BLAINE_CHECKPOINT" --context-length 512 --split validation --inference-window-batch-size 8 --inference-batch-size 64 --tag batch8
python eval/backtest.py --target residue --checkpoint "YOUR_RESIDUE_CHECKPOINT" --context-length 256 --split validation --inference-window-batch-size 8 --inference-batch-size 64 --tag batch8
```

- inference-window-batch-size: predict_df 한 번에 넘기는 독립 예측 요청 수. 기본 1, 위 예시는 8.
- inference-batch-size: Chronos 내부 추론 배치 크기. 기본 64. 학습 batch_size와 별개이며,
  타깃과 보조 입력 채널을 함께 셉니다. 64가 예측 시점 64개를 뜻하지 않습니다.
  현재 이전 품질 포함 입력은 요청당 42채널이므로 내부 GPU 배치는 훨씬 작은 요청 수로 구성됩니다.
  메모리가 충분하면 128 또는 256을 비교할 수 있지만, 속도 개선은 실제로 측정해야 합니다.
- verify-batch-windows: 배치 추론 전에 개별 실행과 비교할 표본 수. 기본 8, 0은 비활성화.
  점 예측과 q10/q90을 rtol=1e-5, atol=1e-5로 비교하며 초과하면 저장 전에 중단합니다.
  이는 표본 검사입니다. 논문 결과에 사용할 때 전체 평가 시점에서도 비교하는 것이 좋습니다.

기존 실험 실행 스크립트는 추가 옵션을 전달하지 않으므로 계속 개별 실행합니다.
배치 처리를 자동 실험의 기본으로 전환하기 전에 같은 체크포인트로 전체 결과를 비교하세요.
단순 변경으로 학습을 다시 할 필요는 없습니다.

## 시간과 수치 비교

같은 체크포인트, 데이터, target, context, stride, split, inference-batch-size를 유지하고
inference-window-batch-size만 1과 8로 바꿔 서로 다른 tag로 저장합니다.
두 CSV의 호기·시각·실측·naive가 같고 pred/q10/q90이 허용 오차 안에서 같은지 확인합니다.
배치 형상에 따른 부동소수점 오차로 완전한 비트 일치가 보장되지는 않습니다.

결과 JSON에 inference_seconds, inference_windows_per_second, batch_verification이 저장됩니다.
시간은 입력 병합, predict_df와 결과 복원을 포함하며 데이터 읽기·모델 로딩·저장·사전 비교는 제외합니다.
GPU 모델 출력을 CPU 데이터프레임으로 반환한 뒤 측정합니다.
공정한 시간 비교를 위해 충분히 예열하고 같은 조건에서 반복 실행해야 합니다.
기본 개별 모드는 사전 비교가 없으므로 첫 실행 시간을 그대로 배치 모드와 비교하면
예열 효과가 섞일 수 있습니다.

이 변경은 다수 날짜를 평가하는 backtest의 처리량 개선에 적합합니다.
실시간으로 한 시간마다 한 건만 들어온다면 미래 요청을 모으려고 기다려서는 안 됩니다.
같은 시각에 준비된 세 호기의 입력을 품질별 모델에 묶는 방식은 적용할 수 있습니다.
Blaine과 Residue는 가중치가 다른 모델이므로 각각 호출합니다.

## 확인한 범위

- 기존 11개 + 새 배치 관련 4개, 총 15개 단위 테스트 통과.
- 순서가 뒤집힌 모의 출력, 마지막 불완전 묶음, 원본 입력 보존, 누락·중복·잘못된 시각·NaN 결과 차단 검사.
- 미래 target/미래 context 차단과 개별·배치 비교 검사의 실패 조건 확인.
- 실제 데이터 dry-run: Blaine context512 970개, Residue context256 967개 검증 시점 입력 준비 통과.
- 실제 Chronos 가중치 추론 및 GPU 속도 측정은 미실행. 모의 테스트는 실제 모델의 수치 동등성을 입증하지 않음.

공식 API 확인 기준: Chronos v2.3.1의 predict_df, batch_size, cross_learning.
https://github.com/amazon-science/chronos-forecasting/blob/v2.3.1/src/chronos/chronos2/pipeline.py
