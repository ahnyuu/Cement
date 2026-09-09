#!/usr/bin/env bash
# ============================================================================
# 물리적 이상치 처리 방식 A/B 실험 (2026-09-02)  --  02_physical_outlier_handling/
# ============================================================================
# 01_iqr_outlier_removal에서 "통계적 IQR 제거는 안 한다"가 확정된 뒤, 남은 축인
# "물리적 이상치를 어떻게 처리하는가"를 3-way로 비교한다:
#   baseline : 현행 그대로 (전부 NaN 처리)            = 정식 파이프라인 main()
#   expA     : 댐퍼 개도율 경계 초과값을 조건부 clip     = prepare_base(damper_oob="clip")
#   expB     : 비-내수 행 유지 + 품질 타깃만 마스킹      = prepare_base(prodtype_mode="mask_target")
#
# context_length는 확정값 고정 (사용자 확정 2026-08-28): blaine -> 1024, residue -> 1536.
# 하이퍼파라미터는 run_iqr_experiment.sh / run_ctx_sweep_v2.sh와 100% 동일
#   (full fine-tuning / predlen4 / 1000 steps / lr 1e-6 / batch 64) -- 오직 입력 CSV만 다르게.
#
# 사전 준비: 원본 xlsx가 있는 머신에서 3개 분기 CSV를 먼저 생성해 둔다.
#   python 02_physical_outlier_handling/build_baseline.py
#   python 02_physical_outlier_handling/build_expA_damper_clip.py
#   python 02_physical_outlier_handling/build_expB_prodtype_keep.py
#
# 평가: 각 분기 모델을 자기 분기 CSV의 테스트셋에서 backtest. 세 분기 모두 타깃값 / 분할 지점이
#       동일하므로(build 스크립트 docstring 참고) naive baseline / normalized skill-score 의
#       분모가 같아 공정 비교가 된다 -- IQR 실험처럼 두 테스트셋을 교차 평가할 필요 없음.
#
# 순차 실행. 예상 소요: blaine 1024 x3 ~= 55분, residue 1536 x3 ~= 5.5시간, backtest 6개 ~= 20분
#            => 총 7시간 안팎.
# 산출물:
#   checkpoints/{target}_full_predlen4_ctx{ctx}_phys_{arm}/
#   eval/backtest_{target}_phys_{arm}.csv
#   checkpoints/_phys_outlier_experiment_logs/*.log
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/Scripts/python.exe
LOG_DIR="checkpoints/_phys_outlier_experiment_logs"
mkdir -p "$LOG_DIR"

BASELINE_CSV="data/processed/phys_baseline/quality_timeseries.csv"
EXPA_CSV="data/processed/phys_expA_damper_clip/quality_timeseries.csv"
EXPB_CSV="data/processed/phys_expB_prodtype_keep/quality_timeseries.csv"

run_one () {
  local target="$1" ctx="$2" arm="$3" csv="$4"
  local tag="full_predlen4_ctx${ctx}_phys_${arm}"
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
    --context-length "$ctx" --stride 4 --tag "phys_${arm}" \
    > "$LOG_DIR/${target}_${arm}_backtest.log" 2>&1
  echo "=== $(date) : backtest exit $? ==="
}

for pair in "baseline:${BASELINE_CSV}" "expA:${EXPA_CSV}" "expB:${EXPB_CSV}"; do
  run_one blaine 1024 "${pair%%:*}" "${pair#*:}"
done
for pair in "baseline:${BASELINE_CSV}" "expA:${EXPA_CSV}" "expB:${EXPB_CSV}"; do
  run_one residue 1536 "${pair%%:*}" "${pair#*:}"
done

echo "=== ALL DONE $(date) ==="