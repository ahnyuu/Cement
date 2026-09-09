# Chronos-2 Quality Net

## 이 폴더를 GPU 머신으로 옮기기

`chronos_quality_net/` 폴더 전체를 그대로 복사(또는 git으로 push/pull)하면 됩니다.
`data/processed/quality_timeseries.csv`가 이미 빌드되어 폴더 안에 포함되어 있으므로,
**원본 xlsx 없이도 바로 fine-tuning을 실행할 수 있습니다.** (`data/build_dataset.py`는
원본 xlsx를 다시 처리해서 이 csv를 재생성할 때만 필요하고, 평소 학습/평가에는 필요 없습니다.)

## GPU 머신 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 1) PyTorch를 먼저, 해당 머신의 CUDA 버전에 맞게 설치
#    https://pytorch.org/get-started/locally/ 에서 본인 CUDA 버전에 맞는 명령 확인
#    예시(CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) 나머지 의존성 설치
pip install -r requirements.txt
```

GPU/CPU는 코드가 `torch.cuda.is_available()`로 자동 감지합니다(`config.default_device_map()`).
`--device-map` 인자를 따로 안 줘도 GPU 머신에서는 자동으로 `cuda`를 씁니다.

## 실행 순서

```bash
# 1) fine-tuning (target별로 따로)
python train/finetune_chronos2.py --target blaine --finetune-mode lora --num-steps 2000
python train/finetune_chronos2.py --target residue --finetune-mode lora --num-steps 2000

# 2) 평가 (naive baseline과 자동 비교 출력)
python eval/backtest.py --target blaine --checkpoint checkpoints/blaine/final --stride 3
python eval/backtest.py --target residue --checkpoint checkpoints/residue/final --stride 3
```

- 모델은 최초 실행 시 HuggingFace에서 `amazon/chronos-2` (120M) 가중치를 자동 다운로드합니다.
  인터넷 연결이 필요하고, `HF_TOKEN` 환경변수를 설정하면 다운로드 속도 제한이 풀립니다(선택 사항).
- `checkpoints/`는 fine-tuning 실행 시 자동 생성됩니다.
- 원본 xlsx를 다시 처리해야 할 경우에만 `data/build_dataset.py`를 실행하세요.
  원본 파일 경로는 `config.py`의 `SOURCE_XLSX`(환경변수 `CHRONOS_SOURCE_XLSX`로 재정의 가능)에
  있으며, 이 컴퓨터의 경로라 GPU 머신에서는 보통 존재하지 않습니다 — 필요 시 경로를 바꿔주세요.

## 다른 전처리 버전(예: IQR 유무)으로 학습/평가하기

기본적으로 학습/평가는 `data/processed/quality_timeseries.csv`(IQR 이상치 제거 적용된 정식 버전)를
읽습니다. 다른 processed csv로 바꾸려면 `CHRONOS_PROCESSED_CSV` 환경변수를 쓰고, 체크포인트/결과
파일이 서로 안 겹치도록 `--output-tag`(학습)와 `--tag`(평가)를 같이 지정하세요:

```bash
CHRONOS_PROCESSED_CSV=data/processed/with_iqr/quality_timeseries.csv \
    python train/finetune_chronos2.py --target blaine --output-tag with_iqr
CHRONOS_PROCESSED_CSV=data/processed/with_iqr/quality_timeseries.csv \
    python eval/backtest.py --target blaine --checkpoint checkpoints/blaine_with_iqr/final --tag with_iqr

CHRONOS_PROCESSED_CSV=data/processed/without_iqr/quality_timeseries.csv \
    python train/finetune_chronos2.py --target blaine --output-tag without_iqr
CHRONOS_PROCESSED_CSV=data/processed/without_iqr/quality_timeseries.csv \
    python eval/backtest.py --target blaine --checkpoint checkpoints/blaine_without_iqr/final --tag without_iqr
```

두 버전 모두 `checkpoints/blaine_with_iqr`/`checkpoints/blaine_without_iqr`, `eval/backtest_blaine_with_iqr.csv`/`eval/backtest_blaine_without_iqr.csv`로 따로 남아서 언제든 비교할 수 있습니다.
