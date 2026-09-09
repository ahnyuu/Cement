"""실험 B: 비-내수(ProdType != 내수) 시각의 행을 삭제하지 않고 유지.

현행(기준선)은 clean()에서 비-내수 행을 통째로 제거한다 -> 이후 시간 그리드 재색인으로 그 시각들이
"모든 컬럼 NaN" 행으로 되살아난다. 결과적으로 비-내수 구간 동안 밀이 돌고 있었어도(댐퍼/피드/온도 등
실제 운전값이 존재) Chronos-2는 그 covariate 컨텍스트를 전혀 못 본다.

실험 B는 비-내수 행을 유지하되 품질 타깃(blaine/residue)만 NaN 처리한다 -- 운전 조건 covariate는
실제 값으로 남아 모델 컨텍스트가 끊기지 않는다.

기준선과 다른 부분: prepare_base(prodtype_mode="mask_target") 한 줄. 스키마 / 행 수 / 타임스탬프 /
타깃값 / 분할 지점은 기준선과 동일하고(주의 참고), 오직 비-내수 시각의 feature 컬럼 값만 다르다
(기준선: 전부 NaN, 실험 B: 실제 값) -> backtest 평가 지점 동일 -> 공정한 skill-score 비교.

주의 1: 어떤 호기의 맨 처음/맨 끝 행이 비-내수라면 실험 B의 시간 범위가 기준선보다 극소량 넓어질 수
        있다(행 몇 개). backtest는 실측 갱신 시점에 앵커되므로 평가 지점에 사실상 영향 없으나,
        결과 노트북에서 backtest CSV의 n이 세 분기 모두 같은지 한번 확인할 것.
주의 2: 이건 최소 변경 버전이다. 더 강한 버전 -- is_domestic 플래그를 known-future covariate로
        추가 -- 은 config.py / finetune_chronos2.py / backtest.py 수정이 필요하므로, 실험 B가
        유망하면 후속으로 검토한다.

원본 xlsx가 있는 컴퓨터에서 실행 (config.py의 SOURCE_XLSX 참고).

실행:
    python 02_physical_outlier_handling/build_expB_prodtype_keep.py

결과: data/processed/phys_expB_prodtype_keep/quality_timeseries.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
from data.build_dataset import finalize_and_save, prepare_base

OUTPUT_PATH = ROOT / "data" / "processed" / "phys_expB_prodtype_keep" / "quality_timeseries.csv"


def main() -> None:
    # ==== 기준선과 다른 부분은 여기뿐: 비-내수 행을 삭제하지 않고 타깃만 마스킹 ====
    df = prepare_base(prodtype_mode="mask_target")
    # ====================================================================
    finalize_and_save(df, OUTPUT_PATH, iqr_removed=None)


if __name__ == "__main__":
    main()