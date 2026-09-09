"""
REVISED 2026-09-09: validation-only study; outputs under review_v2.
The original study description below is retained as historical context.

IQR outlier-removal A/B -- train 4 models, backtest each on BOTH test sets (8 backtests).

Reproduces Cement_code/chronos_quality/train/run_iqr_experiment.sh on the measured-actual data,
with the context_length values from the 2026-09-06 sweep (blaine 512 / residue 256) instead of
the old 1024 / 1536.

    train:  full fine-tune, predlen 4, num_steps 1000, lr 1e-6, batch 64
            {blaine ctx512, residue ctx256} x {with_iqr CSV, without_iqr CSV}  -> 4 checkpoints
    eval:   each checkpoint backtested on BOTH test sets (same eval points => fair skill-score)
            evalon_raw   = without_iqr test set (extremes kept)  = deployment reality, PRIMARY
            evalon_clean = with_iqr test set     (IQR-cleaned normal operation)

Outputs:
    checkpoints/{target}_ctx{N}_{withiqr,withoutiqr}/
    eval/backtest_{target}_ctx{N}_{withiqr,withoutiqr}_evalon_{raw,clean}.csv

Resumable. Logged to experiments/iqr/iqr_experiment.log.
Prereq: experiments/iqr/build_variants.py has been run.

    .venv/Scripts/python.exe experiments/iqr/run_iqr_experiment.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.research_protocol import checkpoint_complete, evaluation_complete

PY = sys.executable
LOG = ROOT / "experiments" / "iqr" / "iqr_experiment_review_v2.log"

CTX = {"blaine": 512, "residue": 256}
WITHOUT_CSV = "data/processed/review_v2/without_iqr/quality_timeseries.csv"
WITH_CSV = "data/processed/review_v2/with_iqr/quality_timeseries.csv"
TRAIN_CSV = {"withiqr": WITH_CSV, "withoutiqr": WITHOUT_CSV}
EVAL_CSV = {"raw": WITHOUT_CSV, "clean": WITH_CSV}


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], env_csv: str, phase: str) -> None:
    log(f"START {phase}  (CHRONOS_PROCESSED_CSV={env_csv})")
    env = {**os.environ, "CHRONOS_PROCESSED_CSV": env_csv, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = (time.time() - t0) / 60
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"  cmd: {' '.join(cmd)}\n----- stdout tail -----\n"
                + "\n".join((p.stdout or "").splitlines()[-20:]) + "\n")
        if p.returncode != 0:
            f.write(f"----- stderr -----\n{p.stderr}\n")
    if p.returncode != 0:
        log(f"FAIL  {phase} (exit {p.returncode}, {dt:.1f} min)")
        raise SystemExit(f"{phase} failed -- see {LOG}")
    log(f"DONE  {phase} ({dt:.1f} min)")


def main() -> None:
    print("review_v2: parameter selection uses VALIDATION; historical outputs are not reused.")
    for name, rel in [("with_iqr", WITH_CSV), ("without_iqr", WITHOUT_CSV)]:
        if not (ROOT / rel).exists():
            raise SystemExit(f"missing {rel} -- run experiments/iqr/build_variants.py first")

    log(f"=== IQR A/B start (ctx={CTX}, full/predlen4/1000steps/lr1e-6/batch64) ===")
    for target, ctx in CTX.items():
        for trainvar in ("withiqr", "withoutiqr"):
            tag = f"ctx{ctx}_{trainvar}"
            ckpt = ROOT / "checkpoints" / "review_v2" / f"{target}_{tag}" / "final"
            if checkpoint_complete(ckpt):
                log(f"SKIP  finetune {target} {trainvar} (checkpoint exists)")
            else:
                run([PY, "train/finetune_chronos2.py", "--target", target,
                     "--finetune-mode", "full", "--prediction-length", "4",
                     "--context-length", str(ctx), "--learning-rate", "1e-6",
                     "--num-steps", "1000", "--batch-size", "64", "--output-tag", tag],
                    TRAIN_CSV[trainvar], f"finetune {target} {trainvar} ctx{ctx}")

            for evalvar in ("raw",):  # same unfiltered validation inputs and labels for both models
                csv = ROOT / "eval" / "review_v2" / "validation" / f"backtest_{target}_ctx{ctx}_{trainvar}_evalon_{evalvar}.csv"
                if evaluation_complete(csv):
                    log(f"SKIP  backtest {target} {trainvar} evalon_{evalvar} (csv exists)")
                    continue
                run([PY, "eval/backtest.py", "--split", "validation", "--target", target, "--checkpoint", str(ckpt),
                     "--context-length", str(ctx), "--stride", "4",
                     "--tag", f"ctx{ctx}_{trainvar}_evalon_{evalvar}"],
                    EVAL_CSV[evalvar], f"backtest {target} {trainvar} evalon_{evalvar}")
    log("=== IQR A/B complete ===")


if __name__ == "__main__":
    main()
