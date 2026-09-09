"""
분기 B: IQR 이상치 제거를 적용하지 않은 전처리 (build_with_iqr.py의 비교 대상).

공통 로직은 data/build_dataset.py의 prepare_base()/finalize_and_save()를 build_with_iqr.py와
똑같이 가져다 씀 -- 다른 점은 IQR 이상치 제거 단계가 아예 없다는 것, 그게 전부.

원본 xlsx가 있는 컴퓨터에서 실행해야 함 (config.py의 SOURCE_XLSX 참고).

실행:
    python notebooks/01_iqr_outlier_removal/build_without_iqr.py

결과: data/processed/without_iqr/quality_timeseries.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.build_dataset import finalize_and_save, prepare_base

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "without_iqr" / "quality_timeseries.csv"
)


def main() -> None:
    df = prepare_base()
    # (IQR 이상치 제거 없음 -- build_with_iqr.py와 비교할 대상이라 이 단계를 아예 건너뜀)
    finalize_and_save(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
