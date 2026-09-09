#!/usr/bin/env bash
# Re-run the full predlen4 context_length sweep (2026-08-19) with the new
# make_loss_history_callback()-based finetune_chronos2.py, so every run's loss_history.csv/.png is
# complete regardless of save_total_limit checkpoint pruning (see eval/loss_history.py docstring).
# Also folds ctx=24 into predlen=4 for the first time, so it becomes comparable to the rest.
#
# --num-steps 1000 (half the original 2000) per user's choice, trading completeness for wall-clock
# time -- prior full runs still showed val_loss decreasing at their last visible point, so 1000
# should still be enough to see the trend even if it doesn't reach convergence.
#
# Sequential on purpose: this GPU has 8.5GB VRAM, confirmed OK for one full-fine-tuning run at a
# time up to ctx=1024/batch_size=64, not verified safe for two concurrent runs.
set -uo pipefail
cd "$(dirname "$0")/.."

TARGETS="blaine residue"
CTX_VALUES="24 73 128 168 256 384 512 768 1024"
NUM_STEPS=1000
LOG_DIR="checkpoints/_sweep_v2_logs"
mkdir -p "$LOG_DIR"

for target in $TARGETS; do
  for ctx in $CTX_VALUES; do
    tag="full_predlen4_ctx${ctx}_v2"
    log="$LOG_DIR/${target}_ctx${ctx}.log"
    echo "=== $(date) : starting ${target} ctx=${ctx} -> checkpoints/${target}_${tag} ==="
    ./.venv/Scripts/python.exe train/finetune_chronos2.py \
      --target "$target" \
      --finetune-mode full \
      --prediction-length 4 \
      --context-length "$ctx" \
      --learning-rate 1e-6 \
      --num-steps "$NUM_STEPS" \
      --batch-size 64 \
      --output-tag "$tag" \
      > "$log" 2>&1
    status=$?
    echo "=== $(date) : finished ${target} ctx=${ctx} (exit $status) ==="
  done
done
echo "=== ALL DONE ==="
