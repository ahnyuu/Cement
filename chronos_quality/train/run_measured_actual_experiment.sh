#!/usr/bin/env bash
# 2026-09-05: "운전&품질데이터_실측치.xlsx"(신규 데이터, 04_measured_actual_data/build_measured_actual.py
# 로 빌드)로, 지금까지 가장 성능이 좋았던 context_length 설정(2026-08-28 사용자 확정: blaine=1024,
# residue=1536)을 zero-shot/LoRA/Full 3가지 방식 전부 재현. results_context1024.ipynb /
# results_context1536.ipynb와 같은 하이퍼파라미터(prediction_length=4, num_steps=1000, batch_size=64;
# LoRA는 기본 lr=1e-5, Full은 lr=1e-6) -- 바뀌는 건 데이터 소스(CHRONOS_PROCESSED_CSV)와 체크포인트/
# 결과 파일 이름의 "_measured" 태그뿐, 기존 체크포인트/backtest 결과는 전혀 건드리지 않음.
#
# zero-shot은 학습이 필요 없어 이 스크립트에는 없음 -- eval/backtest.py를 checkpoint=amazon/chronos-2
# 로 바로 호출(run_measured_actual_backtest.sh 참고).
#
# Sequential on purpose (run_ctx_sweep_v2.sh와 동일한 이유: 이 GPU의 VRAM으로 동시 2개 실행은 미검증).
set -uo pipefail
cd "$(dirname "$0")/.."

export CHRONOS_PROCESSED_CSV="data/processed/measured_actual/quality_timeseries.csv"

LOG_DIR="checkpoints/_measured_actual_logs"
mkdir -p "$LOG_DIR"

run() {
  local target=$1 mode=$2 ctx=$3 tag=$4
  shift 4
  local log="$LOG_DIR/${target}_${mode}_ctx${ctx}.log"
  echo "=== $(date) : starting ${target} ${mode} ctx=${ctx} -> checkpoints/${target}_${tag} ==="
  ./.venv/Scripts/python.exe train/finetune_chronos2.py \
    --target "$target" \
    --finetune-mode "$mode" \
    --prediction-length 4 \
    --context-length "$ctx" \
    --num-steps 1000 \
    --batch-size 64 \
    --output-tag "$tag" \
    "$@" \
    > "$log" 2>&1
  status=$?
  echo "=== $(date) : finished ${target} ${mode} ctx=${ctx} (exit $status) ==="
}

# blaine: ctx=1024 (2026-08-28 확정 최적값)
run blaine lora 1024 "predlen4_ctx1024_lora_measured"
run blaine full 1024 "predlen4_ctx1024_full_measured" --learning-rate 1e-6

# residue: ctx=1536 (2026-08-28 확정 최적값)
run residue lora 1536 "predlen4_ctx1536_lora_measured"
run residue full 1536 "predlen4_ctx1536_full_measured" --learning-rate 1e-6

echo "=== ALL DONE ==="
