#!/usr/bin/env bash
# 2026-09-05: run_oldxlsx_ctrl_experiment.sh 체크포인트 6개(blaine/residue x zero-shot/LoRA/Full)를
# 같은 canonical CSV(현재 파이프라인 + 기존 xlsx)에서 backtest. run_measured_actual_backtest.sh와
# 완전히 동일한 조건(옵션/stride 기본값)으로, --tag만 "_oldxlsx"로 구분.
set -uo pipefail
cd "$(dirname "$0")/.."

export CHRONOS_PROCESSED_CSV="data/processed/quality_timeseries.csv"

LOG_DIR="eval/_oldxlsx_ctrl_backtest_logs"
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
run_bt blaine amazon/chronos-2                                          1024 predlen4_ctx1024_zeroshot_oldxlsx
run_bt blaine checkpoints/blaine_predlen4_ctx1024_lora_oldxlsx/final    1024 predlen4_ctx1024_lora_oldxlsx
run_bt blaine checkpoints/blaine_predlen4_ctx1024_full_oldxlsx/final    1024 predlen4_ctx1024_full_oldxlsx

# residue, ctx=1536
run_bt residue amazon/chronos-2                                         1536 predlen4_ctx1536_zeroshot_oldxlsx
run_bt residue checkpoints/residue_predlen4_ctx1536_lora_oldxlsx/final  1536 predlen4_ctx1536_lora_oldxlsx
run_bt residue checkpoints/residue_predlen4_ctx1536_full_oldxlsx/final  1536 predlen4_ctx1536_full_oldxlsx

echo "=== ALL BACKTESTS DONE ==="
