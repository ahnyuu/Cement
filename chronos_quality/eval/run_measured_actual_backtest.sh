#!/usr/bin/env bash
# 2026-09-05: measured_actual 데이터(운전&품질데이터_실측치.xlsx)로 학습한 6개 체크포인트(blaine/residue
# x zero-shot/LoRA/Full, ctx는 각 타깃 확정 최적값 1024/1536) 전부를 같은 데이터(CHRONOS_PROCESSED_CSV)
# 위에서 backtest. --tag에 "measured" 표시를 붙여 기존 backtest_*_predlen4_ctx{1024,1536}_*.csv를
# 전혀 건드리지 않음. stride/기타 옵션은 스크립트 기본값(backtest.py 기본 --stride 4)을 그대로 사용해
# results_context1024.ipynb/results_context1536.ipynb와 같은 조건으로 비교 가능하게 함.
set -uo pipefail
cd "$(dirname "$0")/.."

export CHRONOS_PROCESSED_CSV="data/processed/measured_actual/quality_timeseries.csv"

LOG_DIR="eval/_measured_actual_backtest_logs"
mkdir -p "$LOG_DIR"

run_bt() {
  local target=$1 ckpt=$2 ctx=$3 tag=$4
  local log="$LOG_DIR/${tag}.log"
  echo "=== $(date) : backtest ${target} ctx=${ctx} ckpt=${ckpt} tag=${tag} ==="
  ./.venv/Scripts/python.exe eval/backtest.py \
    --target "$target" \
    --checkpoint "$ckpt" \
    --context-length "$ctx" \
    --tag "$tag" \
    > "$log" 2>&1
  echo "=== $(date) : done ${tag} (exit $?) ==="
}

# blaine, ctx=1024
run_bt blaine amazon/chronos-2                                              1024 predlen4_ctx1024_zeroshot_measured
run_bt blaine checkpoints/blaine_predlen4_ctx1024_lora_measured/final       1024 predlen4_ctx1024_lora_measured
run_bt blaine checkpoints/blaine_predlen4_ctx1024_full_measured/final       1024 predlen4_ctx1024_full_measured

# residue, ctx=1536
run_bt residue amazon/chronos-2                                             1536 predlen4_ctx1536_zeroshot_measured
run_bt residue checkpoints/residue_predlen4_ctx1536_lora_measured/final     1536 predlen4_ctx1536_lora_measured
run_bt residue checkpoints/residue_predlen4_ctx1536_full_measured/final    1536 predlen4_ctx1536_full_measured

echo "=== ALL BACKTESTS DONE ==="
