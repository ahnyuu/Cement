#!/usr/bin/env bash
# 2026-09-05: 통제 비교용 -- "실측치 vs 기존 xlsx"를 순수하게 분리하기 위해, 현재 canonical 파이프라인
# (data/processed/quality_timeseries.csv, 2026-08-31 물리 정제 규칙 확장판, IQR-off, prev_density=
# sparse -- 08-28/08-31 확정 그대로) 위에서 blaine=ctx1024 / residue=ctx1536, zero-shot/LoRA/Full 3가지
# 를 다시 재현. run_measured_actual_experiment.sh와 하이퍼파라미터 100% 동일, CHRONOS_PROCESSED_CSV
# 만 canonical(기존 xlsx 소스)로 -- 이게 기본값이라 사실 env var 없이도 같지만 명시적으로 고정.
#
# 이 결과가 나오면: "oldxlsx_ctrl"(현재 파이프라인+기존 xlsx) vs "measured"(현재 파이프라인+실측치)
# 비교가 xlsx 소스 교체 효과만 순수하게 분리한 비교가 됨. 기존 Aug19/21 결과 파일은 그대로 두고
# (파이프라인 자체가 다른 시점 것이라 참고용으로만 남김), 이 run이 새로운 "before" 기준이 됨.
set -uo pipefail
cd "$(dirname "$0")/.."

export CHRONOS_PROCESSED_CSV="data/processed/quality_timeseries.csv"

LOG_DIR="checkpoints/_oldxlsx_ctrl_logs"
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
run blaine lora 1024 "predlen4_ctx1024_lora_oldxlsx"
run blaine full 1024 "predlen4_ctx1024_full_oldxlsx" --learning-rate 1e-6

# residue: ctx=1536 (2026-08-28 확정 최적값)
run residue lora 1536 "predlen4_ctx1536_lora_oldxlsx"
run residue full 1536 "predlen4_ctx1536_full_oldxlsx" --learning-rate 1e-6

echo "=== ALL DONE ==="
