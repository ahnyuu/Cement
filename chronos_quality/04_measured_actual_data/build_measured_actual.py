"""
04_measured_actual_data: "운전&품질데이터_실측치.xlsx" (2026-09-05, Cement 폴더에 신규 추가)를
원본 소스로 삼아 canonical 전처리(build_dataset.py의 prepare_base()/finalize_and_save(), 기본
설정 그대로 -- IQR 미적용/damper_oob="nan"/prodtype_mode="drop"/prev_density="sparse")를 재실행.

이 신규 xlsx와 기존 SOURCE_XLSX("운전 & 품질 데이터_2026.xlsx")는 시트 구성/행수가 완전히 동일하지만
(2CM/3CM/4CM 각각 39854/40071/39421행, LIMS 시트 포함 8개 시트 모두 일치), 2025-01-01~2025-11-17
구간에서 품질 실측 시각(4시간 간격: 00/04/08/12/16/20시)이 아닌 시간대의 grind_aid/blaine/residue/
ProdType/Remark 값이 기존 파일에서는 채워져 있던 반면 신규 파일에서는 비어 있음 (2026-09-05 셀 단위
diff로 확인: 2CM/3CM/4CM에서 각각 4771/4715/4601행이 다르고, 그중 대다수(85%+)가 비측정 시각의
carry-forward성 값을 NaN으로 되돌린 것; 나머지 소수(2CM 기준 측정 시각 507건 중 486건은 값 삭제,
21건은 실측값 자체 정정)는 실제 측정치 보정). 즉 이 파일이 이름 그대로 "실측치만" 남긴 버전이고,
기존 파일은 실측 외 시간대에도 값이 스며 있던(원인 미상 -- 소스 시스템의 자체 carry-forward로 추정)
상태였던 것으로 보임.

build_dataset.py의 로직 자체는 전혀 건드리지 않음 -- 바뀌는 건 SOURCE_XLSX 하나뿐이라, 이 스크립트가
만드는 quality_timeseries.csv와 canonical data/processed/quality_timeseries.csv의 차이는 순수하게
입력 xlsx 차이만 반영한다.

실행:
    python 04_measured_actual_data/build_measured_actual.py

결과: data/processed/measured_actual/quality_timeseries.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # chronos_quality/
sys.path.insert(0, str(REPO_ROOT))

# config.py reads CHRONOS_SOURCE_XLSX at import time, so this must be set before the
# `from data.build_dataset import ...` below (which imports config.py transitively).
_NEW_SOURCE_XLSX = REPO_ROOT.parents[1] / "운전&품질데이터_실측치.xlsx"
if not _NEW_SOURCE_XLSX.exists():
    raise FileNotFoundError(f"신규 실측치 xlsx를 찾지 못했습니다: {_NEW_SOURCE_XLSX}")
os.environ["CHRONOS_SOURCE_XLSX"] = str(_NEW_SOURCE_XLSX)

from data.build_dataset import finalize_and_save, prepare_base  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "measured_actual" / "quality_timeseries.csv"


def main() -> None:
    print(f"SOURCE_XLSX -> {_NEW_SOURCE_XLSX}")
    df = prepare_base()
    finalize_and_save(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
