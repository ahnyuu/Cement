"""
분기 A: IQR 이상치 제거를 적용한 전처리.

공통 로직(원본 xlsx 읽기, clean(), prev covariate, warm-up feature, 저장/요약 출력)은
data/build_dataset.py의 prepare_base()/finalize_and_save()를 그대로 가져다 씀 -- 이 파일에서
build_without_iqr.py와 실제로 다른 부분은 "IQR 이상치 제거" 블록 하나뿐임 (아래 표시).

원본 xlsx가 있는 컴퓨터에서 실행해야 함 (config.py의 SOURCE_XLSX 참고, 이 저장소를 그대로
옮긴 GPU 머신에는 보통 없음 -- README.md 참고).

실행:
    python notebooks/01_iqr_outlier_removal/build_with_iqr.py

결과: data/processed/with_iqr/quality_timeseries.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config as cfg
from data.build_dataset import (
    finalize_and_save,
    prepare_base,
    raw_feature_cols,
    remove_iqr_outliers,
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "with_iqr" / "quality_timeseries.csv"
)


def main() -> None:
    df = prepare_base()

    # ==== build_without_iqr.py와 다른 부분은 여기뿐: IQR 이상치 제거 ====
    iqr_cols = [c for c in raw_feature_cols() if c != "RP_proc_time"] + cfg.TARGET_COLS
    df, iqr_removed = remove_iqr_outliers(df, iqr_cols)
    # ====================================================================

    finalize_and_save(df, OUTPUT_PATH, iqr_removed=iqr_removed)


if __name__ == "__main__":
    main()
