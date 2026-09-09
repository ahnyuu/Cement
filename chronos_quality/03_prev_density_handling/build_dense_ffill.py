"""실험: blaine_prev / residue_prev 를 dense 하게 -- 직전 실측값을 다음 실측 시점까지 forward-fill.

현행(baseline)은 `_prev` 를 실측 행(4시간 간격)에만 채우고 그 사이 hourly 행은 전부 NaN 으로 둔다
(add_prev_quality_covariates 의 native 출력). 그러면 모델은 측정 시점 사이 구간에서 "직전 lab 값"
앵커를 자기 hidden state 로만 이어가야 한다.

이 실험은 시간 그리드 재색인 후 `_prev` 를 item_id 별 forward-fill 하여 **모든 행**이 "가장 최근
실측값"을 명시적으로 들고 있게 한다. 참고 노트북(chronos-2-cement_share.ipynb "이전품질 적용 후
Chronos")은 backward-fill(copy_back)을 썼지만, bfill 은 예측 구간(horizon)이 측정 경계를 넘을 때
아직 관측되지 않은 미래 실측값을 known-future covariate 로 노출한다 -> 누출. ffill 은 그렇지 않다:
scheduled 행의 sparse `_prev` 는 이미 4시간 전 측정값이므로, 이를 앞으로 채워도 어떤 행에 놓이든
그 행보다 >=4h 전에 확보된 값만 노출된다 (prediction_length<=4 에서 누출 없음).
-> smoketest_prev_density.py 에서 검증.

baseline 과 다른 부분: finalize_and_save(prev_density="ffill") 한 곳뿐. 스키마 / 행 수 / 타임스탬프 /
타깃값 / 학습·평가 분할 지점은 baseline 과 완전히 동일하고, 오직 `_prev` 2개 컬럼의 비-실측 행 값만
다르다 (baseline: NaN, 실험: 직전 실측값) -> backtest 평가 지점(=실측 갱신 시점, `_prev` 는 원래
non-NaN) 동일 -> 공정한 skill-score 비교.

원본 xlsx 가 있는 컴퓨터에서 실행 (config.py 의 SOURCE_XLSX 참고).

실행:
    python 03_prev_density_handling/build_dense_ffill.py

결과: data/processed/prev_dense_ffill/quality_timeseries.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
from data.build_dataset import finalize_and_save, prepare_base

OUTPUT_PATH = ROOT / "data" / "processed" / "prev_dense_ffill" / "quality_timeseries.csv"


def main() -> None:
    df = prepare_base()  # baseline 과 동일 (damper_oob="nan", prodtype_mode="drop")
    # ==== baseline 과 다른 부분은 여기뿐: _prev 를 item_id 별 forward-fill ====
    finalize_and_save(df, OUTPUT_PATH, iqr_removed=None, prev_density="ffill")
    # =================================================================


if __name__ == "__main__":
    main()
