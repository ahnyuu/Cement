"""finalize_and_save 의 새 인자 prev_density ("sparse" | "ffill") 동작 확인 -- 합성 데이터로.

4.5시간짜리 run_prev_density_experiment.sh 를 돌리기 전에 한번 실행. GPU 불필요, pandas/numpy 만 있으면 됨.
검증 항목:
  1. sparse : _prev 는 실측 scheduled 행에만 값이 있고 (첫 측정 제외) 나머지는 NaN.
  2. ffill  : 첫 측정 이후 모든 행이 non-NaN 이고, 각 행 값 = 그 행 이하의 가장 최근 scheduled _prev 값.
  3. 누출 없음(prediction_length<=4): 임의의 origin i 에 대해 horizon i+1..i+4 각 행의 ffill 값은
     timestamp <= ts[i] 인 실측값에서 온다.
  4. backtest 평가 지점(=실측 scheduled 행)의 _prev 값은 두 방식이 완전히 동일 (ffill 이 non-NaN 을
     덮어쓰지 않음).

실행:
    python 03_prev_density_handling/smoketest_prev_density.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
import config as cfg
from data.build_dataset import PREV_QUALITY_COLS, add_prev_quality_covariates

SCHED = cfg.QUALITY_MEASUREMENT_HOURS  # [0, 4, 8, 12, 16, 20]
fails: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(label)


def make_synth(n_days: int = 4) -> pd.DataFrame:
    """2 호기, hourly. blaine/residue 는 scheduled 시각에만 측정값(행마다 고유), 나머지 NaN.
    일부 scheduled 시각은 결측(측정 누락)으로 비워 실제 데이터 흉내."""
    rows = []
    for item in ("2CM", "3CM"):
        idx = pd.date_range("2025-01-01", periods=n_days * 24, freq="h")
        blaine = np.full(len(idx), np.nan)
        residue = np.full(len(idx), np.nan)
        k = 0
        for i, ts in enumerate(idx):
            if ts.hour in SCHED:
                k += 1
                if k % 7 != 0:  # 7번째 측정마다 결측
                    blaine[i] = 3800.0 + k + (10 if item == "3CM" else 0)
                    residue[i] = 8.0 + 0.01 * k
        rows.append(pd.DataFrame({"item_id": item, "timestamp": idx,
                                  "blaine": blaine, "residue": residue}))
    return pd.concat(rows, ignore_index=True)


def apply_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """finalize_and_save 의 prev_density=='ffill' 분기와 동일한 연산."""
    df = df.copy()
    df[PREV_QUALITY_COLS] = df.groupby("item_id", sort=False)[PREV_QUALITY_COLS].ffill()
    return df


df = make_synth()
sparse = add_prev_quality_covariates(df).sort_values(["item_id", "timestamp"]).reset_index(drop=True)
dense = apply_ffill(sparse)

# --- 1. sparse: scheduled+측정 행에만 값 ---
sched_measured = sparse["timestamp"].dt.hour.isin(SCHED) & sparse["blaine"].notna()
off_sched = ~sparse["timestamp"].dt.hour.isin(SCHED)
check("sparse: 비-scheduled 행은 blaine_prev 전부 NaN",
      bool(sparse.loc[off_sched, "blaine_prev"].isna().all()))
# 각 호기 첫 측정의 prev 는 NaN, 그 이후 scheduled-measured 행은 non-NaN
first_meas_idx = sparse.loc[sched_measured].groupby("item_id").head(1).index
later = sparse.loc[sched_measured].drop(index=first_meas_idx)
check("sparse: 첫 측정 이후 scheduled-measured 행 blaine_prev 는 non-NaN",
      bool(later["blaine_prev"].notna().all()))

# --- 2. ffill: 첫 non-NaN 이후 모든 행 채워짐 + 계단식 유지 ---
for item, g in dense.groupby("item_id"):
    g = g.reset_index(drop=True)
    first = g["blaine_prev"].first_valid_index()
    check(f"ffill[{item}]: 첫 값 이후 blaine_prev 결측 없음",
          bool(g["blaine_prev"].iloc[first:].notna().all()))
    # 계단식: 값이 바뀌는 지점은 scheduled 시각뿐
    changed = g["blaine_prev"].iloc[first:].ne(g["blaine_prev"].iloc[first:].shift())
    change_hours = set(g.loc[changed.index[changed], "timestamp"].dt.hour) - {g["timestamp"].iloc[first].hour}
    check(f"ffill[{item}]: blaine_prev 는 scheduled 시각에만 변함",
          change_hours.issubset(set(SCHED)), f"변한 시각들={sorted(change_hours)}")

# --- 3. 누출 없음: horizon i+1..i+4 의 ffill 값 출처 timestamp <= ts[i] ---
# sparse blaine_prev@S == blaine@(S 직전 scheduled-measured) 이므로, 출처 timestamp 를 역매핑해 확인.
PRED_LEN = 4
leak = 0
for item, g in dense.groupby("item_id"):
    g = g.reset_index(drop=True)
    meas = g.loc[g["blaine"].notna(), ["timestamp", "blaine"]]
    val_to_ts = dict(zip(meas["blaine"].round(6), meas["timestamp"]))
    for i in range(len(g) - 1):
        origin_ts = g["timestamp"].iloc[i]
        for j in range(i + 1, min(i + 1 + PRED_LEN, len(g))):
            v = g["blaine_prev"].iloc[j]
            if pd.isna(v):
                continue
            src_ts = val_to_ts.get(round(float(v), 6))
            if src_ts is not None and src_ts > origin_ts:
                leak += 1
check("ffill: prediction_length=4 에서 known-future 누출 0건", leak == 0, f"누출 {leak}건")

# --- 4. 평가 지점(scheduled-measured 행)의 _prev 는 sparse 와 ffill 이 동일 ---
anchor = sparse["timestamp"].dt.hour.isin(SCHED) & sparse["blaine"].notna()
same = np.allclose(sparse.loc[anchor, "blaine_prev"].fillna(-1),
                   dense.loc[anchor, "blaine_prev"].fillna(-1))
check("평가 지점 blaine_prev: sparse == ffill (ffill 이 non-NaN 을 안 덮어씀)", bool(same))

# 참고 출력: 결측률
print()
print("blaine_prev 결측률:  sparse={:.3f}  ffill={:.3f}".format(
    sparse["blaine_prev"].isna().mean(), dense["blaine_prev"].isna().mean()))

print()
if fails:
    print(f"SMOKETEST FAILED ({len(fails)}): {fails}")
    sys.exit(1)
print("SMOKETEST PASSED")
