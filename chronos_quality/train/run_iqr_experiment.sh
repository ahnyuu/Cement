#!/usr/bin/env bash
# ============================================================================
# IQR 이상치 제거 유무 A/B 실험 (2026-08-28)
# ============================================================================
# context_length는 스윕 결론대로 타깃별 고정 (사용자 확정 2026-08-28):
#     blaine -> 1024,  residue -> 1536
#
# 학습: with_iqr / without_iqr 두 전처리 CSV로 각각 full fine-tuning.
#       하이퍼파라미터는 run_ctx_sweep_v2.sh와 100% 동일 (predlen4 / 1000 steps /
#       lr 1e-6 / batch 64) -- 오직 입력 CSV만 다르게 해서 IQR 효과만 분리.
#
# 평가: 각 모델을 "두" 테스트셋 모두에서 backtest -> 총 4 모델 x 2 = 8 backtest
#   - evalon_raw   : without_iqr 테스트셋 (극단값 포함, 실측 그대로) = 배포 현실, 주 지표
#   - evalon_clean : with_iqr 테스트셋   (IQR로 극단값 제거된 정상운전 구간)
#   두 모델을 같은 테스트셋에서 평가해야 공정 비교가 됨 (naive baseline /
#   normalized skill-score 의 분모도 동일해짐).
#
# 순차 실행 (이 GPU는 8.5GB VRAM -- 동시 2개 학습은 미검증).
# 예상 소요: blaine 1024 x2 ~= 36분, residue 1536 x2 ~= 3.8시간, backtest 8개 ~= 25분
#            => 총 5시간 안팎.
# 산출물:
#   checkpoints/{target}_full_predlen4_ctx{ctx}_{withiqr,withoutiqr}/
#   eval/backtest_{target}_{withiqr,withoutiqr}_evalon_{raw,clean}.csv
#   checkpoints/_iqr_experiment_logs/*.log
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/Scripts/python.exe
LOG_DIR="checkpoints/_iqr_experiment_logs"
mkdir -p "$LOG_DIR"

WITH_IQR_CSV="data/processed/with_iqr/quality_timeseries.csv"
WITHOUT_IQR_CSV="data/processed/without_iqr/quality_timeseries.csv"

run_one () {
  local target="$1" ctx="$2" trainvar="$3" traincsv="$4"
  local tag="full_predlen4_ctx${ctx}_${trainvar}"
  local ckpt="checkpoints/${target}_${tag}/final"

  echo "=== $(date) : TRAIN ${target} / ${trainvar} / ctx=${ctx} -> checkpoints/${target}_${tag} ==="
  CHRONOS_PROCESSED_CSV="$traincsv" "$PY" train/finetune_chronos2.py \
    --target "$target" --finetune-mode full --prediction-length 4 \
    --context-length "$ctx" --learning-rate 1e-6 --num-steps 1000 --batch-size 64 \
    --output-tag "$tag" > "$LOG_DIR/${target}_${trainvar}_train.log" 2>&1
  echo "=== $(date) : train exit $? ==="

  for evalpair in "raw:${WITHOUT_IQR_CSV}" "clean:${WITH_IQR_CSV}"; do
    local evalvar="${evalpair%%:*}" evalcsv="${evalpair#*:}"
    echo "=== $(date) : BACKTEST ${target} / ${trainvar} model / evalon_${evalvar} ==="
    CHRONOS_PROCESSED_CSV="$evalcsv" "$PY" eval/backtest.py \
      --target "$target" --checkpoint "$ckpt" \
      --context-length "$ctx" --stride 4 --tag "${trainvar}_evalon_${evalvar}" \
      > "$LOG_DIR/${target}_${trainvar}_backtest_${evalvar}.log" 2>&1
    echo "=== $(date) : backtest exit $? ==="
  done
}

run_one blaine  1024 withiqr    "$WITH_IQR_CSV"
run_one blaine  1024 withoutiqr "$WITHOUT_IQR_CSV"
run_one residue 1536 withiqr    "$WITH_IQR_CSV"
run_one residue 1536 withoutiqr "$WITHOUT_IQR_CSV"

echo "=== ALL DONE $(date) ==="
