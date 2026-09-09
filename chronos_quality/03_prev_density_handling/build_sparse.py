"""분기 baseline: blaine_prev / residue_prev 를 현행처럼 sparse 하게 (실측 행에만) 넣는다.

finalize_and_save(prev_density="sparse") -- 즉 정식 파이프라인 data/build_dataset.py 의 main() 과
완전히 동일한 산출물이다. 실험(dense ffill)과 나란히 두고 2-way 비교하기 위한 명시적 사본
(02_physical_outlier_handling 이 baseline 을 별도 폴더에 둔 것과 같은 이유).

원본 xlsx 가 있는 컴퓨터에서 실행 (config.py 의 SOURCE_XLSX 참고).

실행:
    python 03_prev_density_handling/build_sparse.py

결과: data/processed/prev_sparse/quality_timeseries.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
from data.build_dataset import finalize_and_save, prepare_base

OUTPUT_PATH = ROOT / "data" / "processed" / "prev_sparse" / "quality_timeseries.csv"


def main() -> None:
    df = prepare_base()  # 현행 기본값 (damper_oob="nan", prodtype_mode="drop")
    finalize_and_save(df, OUTPUT_PATH, iqr_removed=None, prev_density="sparse")


if __name__ == "__main__":
    main()
