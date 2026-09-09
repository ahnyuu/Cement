"""분기 0 (기준선): 현행 물리 이상치 처리 그대로.

clean() 기본값(damper_oob="nan", prodtype_mode="drop"), IQR 미적용 -- 즉 정식 파이프라인
data/build_dataset.py의 main()과 완전히 동일한 산출물이다. 실험 A/B와 나란히 두고 3-way 비교하기
위한 명시적 사본 (01_iqr_outlier_removal이 without_iqr을 canonical과 별개 폴더에 둔 것과 같은 이유).

원본 xlsx가 있는 컴퓨터에서 실행 (config.py의 SOURCE_XLSX 참고).

실행:
    python 02_physical_outlier_handling/build_baseline.py

결과: data/processed/phys_baseline/quality_timeseries.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
from data.build_dataset import finalize_and_save, prepare_base

OUTPUT_PATH = ROOT / "data" / "processed" / "phys_baseline" / "quality_timeseries.csv"


def main() -> None:
    df = prepare_base()  # damper_oob="nan", prodtype_mode="drop" -- 현행 기본값
    finalize_and_save(df, OUTPUT_PATH, iqr_removed=None)


if __name__ == "__main__":
    main()