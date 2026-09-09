"""실험 A: 댐퍼 개도율의 물리 한계 초과값을 NaN 대신 조건부 clip.

댐퍼 개도율은 물리적으로 0~100%만 가능한데 센서 캘리브레이션 오차로 105%, -8% 같은 값이 종종 찍힌다.
현행(기준선)은 이런 값을 전부 NaN 처리한다 -- "거의 만개" / "거의 전폐"라는 실제 상태 정보까지 버린다.
실험 A는 한계를 DAMPER_CLIP_MARGIN_PCT(config.py, 기본 20%p) 이내로 벗어난 값만 경계값(0 또는 100)으로
clip해 그 신호를 살리고, 그 이상 벗어난 값은 기존대로 NaN(센서 고장으로 간주)으로 둔다.

기준선과 다른 부분: prepare_base(damper_oob="clip") 한 줄. 타깃(blaine/residue)은 건드리지 않으므로
행 수 / 타임스탬프 / 타깃값 / 학습·평가 분할 지점이 기준선과 완전히 동일하다
-> backtest 평가 지점 동일 -> 공정한 skill-score 비교. 분기 간 차이는 댐퍼 컬럼의 소수 셀뿐이다.

원본 xlsx가 있는 컴퓨터에서 실행 (config.py의 SOURCE_XLSX 참고).

실행:
    python 02_physical_outlier_handling/build_expA_damper_clip.py

결과: data/processed/phys_expA_damper_clip/quality_timeseries.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
from data.build_dataset import finalize_and_save, prepare_base

OUTPUT_PATH = ROOT / "data" / "processed" / "phys_expA_damper_clip" / "quality_timeseries.csv"


def main() -> None:
    # ==== 기준선과 다른 부분은 여기뿐: 댐퍼 초과값을 NaN 대신 경계값으로 clip ====
    df = prepare_base(damper_oob="clip")
    # =====================================================================
    finalize_and_save(df, OUTPUT_PATH, iqr_removed=None)


if __name__ == "__main__":
    main()