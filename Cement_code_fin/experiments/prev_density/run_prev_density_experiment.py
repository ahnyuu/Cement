"""
REVISED 2026-09-09: validation-only study; outputs under review_v2.
The original study description below is retained as historical context.

prev-density A/B: sparse (current canonical) vs ffill prev covariates.

Reproduces the old folder's 2026-09-02 prev-density experiment on the measured-actual data, at
the 2026-09-06 sweep-optimal context_length (blaine 512 / residue 256).

  arm "sparse" : blaine_prev/residue_prev only at scheduled measurement rows (current canonical)
  arm "ffill"  : those two columns forward-filled per station after the hourly reindex

4 models {blaine, residue} x {sparse, ffill}, full fine-tune, predlen 4, 1000 steps, lr 1e-6,
batch 64. Each arm trained AND backtested on its own CSV (targets / split points / eval points /
naive baseline identical between arms -- only the two *_prev covariate columns differ, which is
the treatment; leak-safe per build_variants.py audit).

Outputs:
  checkpoints/{target}_ctx{N}_prevdensity_{sparse,ffill}/
  eval/backtest_{target}_ctx{N}_prevdensity_{sparse,ffill}.csv

Resumable. Logged to experiments/prev_density/prev_density_experiment.log.
Prereq: experiments/prev_density/build_variants.py has been run.

    .venv/Scripts/python.exe experiments/prev_density/run_prev_density_experiment.py
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
LOG = ROOT / "experiments" / "prev_density" / "prev_density_experiment_review_v2.log"

CTX = {"blaine": 512, "residue": 256}
CSV = {
    "sparse": "data/processed/prev_sparse/quality_timeseries.csv",
    "ffill": "data/processed/prev_ffill/quality_timeseries.csv",
}


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
    for name, rel in CSV.items():
        if not (ROOT / rel).exists():
            raise SystemExit(f"missing {rel} -- run experiments/prev_density/build_variants.py first")

    log(f"=== prev-density A/B start (ctx={CTX}, full/predlen4/1000steps/lr1e-6/batch64) ===")
    for target, ctx in CTX.items():
        for arm in ("sparse", "ffill"):
            tag = f"ctx{ctx}_prevdensity_{arm}"
            ckpt = ROOT / "checkpoints" / "review_v2" / f"{target}_{tag}" / "final"
            if checkpoint_complete(ckpt):
                log(f"SKIP  finetune {target} {arm} (checkpoint exists)")
            else:
                run([PY, "train/finetune_chronos2.py", "--target", target,
                     "--finetune-mode", "full", "--prediction-length", "4",
                     "--context-length", str(ctx), "--learning-rate", "1e-6",
                     "--num-steps", "1000", "--batch-size", "64", "--output-tag", tag],
                    CSV[arm], f"finetune {target} {arm} ctx{ctx}")

            csv_out = ROOT / "eval" / "review_v2" / "validation" / f"backtest_{target}_{tag}.csv"
            if evaluation_complete(csv_out):
                log(f"SKIP  backtest {target} {arm} (csv exists)")
            else:
                run([PY, "eval/backtest.py", "--split", "validation", "--target", target, "--checkpoint", str(ckpt),
                     "--context-length", str(ctx), "--stride", "4", "--tag", tag],
                    CSV[arm], f"backtest {target} {arm}")
    log("=== prev-density A/B complete ===")


if __name__ == "__main__":
    main()
