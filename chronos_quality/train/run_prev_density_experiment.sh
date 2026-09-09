#!/usr/bin/env bash
# ============================================================================
# _prev covariate 밀도 A/B 실험 (2026-09-02)  --  03_prev_density_handling/
# ============================================================================
# blaine_prev / residue_prev 를
#   sparse : 실측 scheduled 행에만 값, 그 사이는 NaN (현행 기본값)      = prev_density="sparse"
#   ffill  : 직전 실측값을 다음 측정 시점까지 forward-fill (모든 행에 앵커) = prev_density="ffill"
# 로 넣었을 때 어느 쪽이 나은지 2-way 비교.
#
# context_length 는 확정값 고정 (사용자 확정 2026-08-28): blaine -> 1024, residue -> 1536.
# 하이퍼파라미터는 run_iqr_experiment.sh / run_phys_outlier_experiment.sh 와 100% 동일
#   (full fine-tuning / predlen4 / 1000 steps / lr 1e-6 / batch 64) -- 오직 입력 CSV 만 다르게.
#
# 사전 준비: 원본 xlsx 가 있는 머신에서 2개 분기 CSV 를 먼저 생성해 둔다.
#   python 03_prev_density_handling/smoketest_prev_density.py     # 먼저 로직 확인
#   python 03_prev_density_handling/build_sparse.py
#   python 03_prev_density_handling/build_dense_ffill.py
#
# 평가: 각 분기 모델을 자기 분기 CSV 의 테스트셋에서 backtest. 두 분기는 타깃값 / 분할 지점이
#       동일하고 (build 스크립트 docstring 참고), backtest 평가 지점은 실측 갱신 시점이라 그 행의
#       _prev 값도 두 분기가 같다 -> naive baseline / normalized skill-score 분모 동일 -> 공정 비교.
#
# 순차 실행. 예상 소요: blaine 1024 x2 ~= 37분, residue 1536 x2 ~= 3.7시간, backtest 4개 ~= 13분
#            => 총 4.5시간 안팎.
# 산출물:
#   checkpoints/{target}_full_predlen4_ctx{ctx}_prevdensity_{arm}/
#   eval/backtest_{target}_prevdensity_{arm}.csv
#   checkpoints/_prev_density_experiment_logs/*.log
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/Scripts/python.exe
LOG_DIR="checkpoints/_prev_density_experiment_logs"
mkdir -p "$LOG_DIR"

SPARSE_CSV="data/processed/prev_sparse/quality_timeseries.csv"
FFILL_CSV="data/processed/prev_dense_ffill/quality_timeseries.csv"

run_one () {
  local target="$1" ctx="$2" arm="$3" csv="$4"
  local tag="full_predlen4_ctx${ctx}_prevdensity_${arm}"
  local ckpt="checkpoints/${target}_${tag}/final"

  echo "=== $(date) : TRAIN ${target} / ${arm} / ctx=${ctx} -> checkpoints/${target}_${tag} ==="
  CHRONOS_PROCESSED_CSV="$csv" "$PY" train/finetune_chronos2.py \
    --target "$target" --finetune-mode full --prediction-length 4 \
    --context-length "$ctx" --learning-rate 1e-6 --num-steps 1000 --batch-size 64 \
    --output-tag "$tag" > "$LOG_DIR/${target}_${arm}_train.log" 2>&1
  echo "=== $(date) : train exit $? ==="

  echo "=== $(date) : BACKTEST ${target} / ${arm} ==="
  CHRONOS_PROCESSED_CSV="$csv" "$PY" eval/backtest.py \
    --target "$target" --checkpoint "$ckpt" \
    --context-length "$ctx" --stride 4 --tag "prevdensity_${arm}" \
    > "$LOG_DIR/${target}_${arm}_backtest.log" 2>&1
  echo "=== $(date) : backtest exit $? ==="
}

for pair in "sparse:${SPARSE_CSV}" "ffill:${FFILL_CSV}"; do
  run_one blaine 1024 "${pair%%:*}" "${pair#*:}"
done
for pair in "sparse:${SPARSE_CSV}" "ffill:${FFILL_CSV}"; do
  run_one residue 1536 "${pair%%:*}" "${pair#*:}"
done

echo "=== ALL DONE $(date) ==="
