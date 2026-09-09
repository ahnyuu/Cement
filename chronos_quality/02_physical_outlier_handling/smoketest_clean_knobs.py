"""clean()의 새 인자(damper_oob / prodtype_mode) 동작 확인 -- 원본 xlsx 없이 합성 데이터로.

7시간짜리 run_phys_outlier_experiment.sh를 돌리기 전에 한번 실행해서 전처리 로직이 의도대로인지
확인하는 용도. GPU 불필요, pandas/numpy만 있으면 됨.

실행:
    python 02_physical_outlier_handling/smoketest_clean_knobs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))
import config as cfg
from data.build_dataset import clean

cols = (["item_id", "timestamp", "mill_feed_c", "blaine", "residue",
         "mill_out_gas_temp", "mill_out_mater_temp", "RP_roller_p2", "ProdType", "Remark"]
        + cfg.DAMPER_COLS + cfg.BAG_FILTER_PRESS_COLS)
cols = list(dict.fromkeys(cols))

n = 8
base = {c: [0.0] * n for c in cols}
base["item_id"] = ["2CM"] * n
base["timestamp"] = pd.date_range("2025-01-01", periods=n, freq="h")
base["blaine"] = [3800.0] * n
base["residue"] = [8.0] * n
base["ProdType"] = ["내수", "내수", "수출", None, "내수", "수출", "내수", "내수"]
dcol = cfg.DAMPER_COLS[0]
#            row:  0     1      2      3     4      5      6     7
base[dcol] = [50.0, 105.0, 130.0, -8.0, -50.0, 100.0, 0.0, 999.0]
df0 = pd.DataFrame(base)

hi, lo, margin = (cfg.CLEANING_RULES["damper_max_pct"], cfg.CLEANING_RULES["damper_min_pct"],
                  cfg.DAMPER_CLIP_MARGIN_PCT)
fails = []


def check(label, got, expect):
    ok = got == expect or (isinstance(got, list) and got == expect)
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: got={got} expect={expect}")
    if not ok:
        fails.append(label)


print(f"damper bounds [{lo}, {hi}], clip margin {margin}")

print("\n=== damper_oob='nan' (default) ===")
r = clean(df0.copy(), damper_oob="nan", prodtype_mode="mask_target")
d = r[dcol].tolist()
check("out-of-range -> NaN", [x for x in d if not pd.isna(x)], [50.0, 100.0, 0.0])

print("\n=== damper_oob='clip' ===")
r = clean(df0.copy(), damper_oob="clip", prodtype_mode="mask_target")
d = [None if pd.isna(x) else x for x in r[dcol].tolist()]
check("near-overshoot clipped, far -> NaN", d, [50.0, hi, None, lo, None, 100.0, 0.0, None])

print("\n=== prodtype_mode='drop' (default) ===")
r = clean(df0.copy(), prodtype_mode="drop")
check("only 내수 rows kept", len(r), 5)
check("kept ProdType all 내수", sorted(set(r["ProdType"])), ["내수"])

print("\n=== prodtype_mode='mask_target' ===")
r = clean(df0.copy(), prodtype_mode="mask_target").reset_index(drop=True)
check("all rows kept", len(r), 8)
check("targets NaN on out-of-scope rows (2,3,5)",
      sorted(r.index[r["blaine"].isna()].tolist()), [2, 3, 5])
# mask only touches the target columns -- an in-range feature value on an out-of-scope row survives
check("out-of-scope feature value NOT masked (row 5 damper=100)", r[dcol].iloc[5], 100.0)

print("\n=== bad args raise ValueError ===")
for kw in (dict(damper_oob="x"), dict(prodtype_mode="x")):
    try:
        clean(df0.copy(), **kw)
        check(f"raise for {kw}", "no raise", "ValueError")
    except ValueError as e:
        print(f"  [ok] raised for {kw}: {e}")

print()
if fails:
    print(f"SMOKETEST FAILED ({len(fails)}): {fails}")
    sys.exit(1)
print("SMOKETEST PASSED")
