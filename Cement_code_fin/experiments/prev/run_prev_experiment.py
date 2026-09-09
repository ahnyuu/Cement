"""
prev-quality covariate A/B: with prev (sparse, current canonical) vs without prev.

Reproduces the old folder's 2026-08-12 no-prev vs prev comparison on the measured-actual data,
at the 2026-09-06 sweep-optimal context_length (blaine 512 / residue 256).

  arm "prev"   : blaine_prev/residue_prev as known-future covariates (default)
  arm "noprev" : --no-prev  -> those two columns dropped entirely

Everything else identical: same canonical CSV, full fine-tune, predlen 4, 1000 steps, lr 1e-6,
batch 64, stride 4. Same eval points / naive baseline in both arms (make_eval_tasks keys off the
target series, not the covariates) -> fair skill-score comparison.

Outputs:
  checkpoints/{target}_ctx{N}_{prev,noprev}/
  eval/backtest_{target}_ctx{N}_{prev,noprev}.csv

Resumable. Logged to experiments/prev/prev_experiment.log.

    .venv/Scripts/python.exe experiments/prev/run_prev_experiment.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LOG = ROOT / "experiments" / "prev" / "prev_experiment.log"

CTX = {"blaine": 512, "residue": 256}
ARMS = {"prev": [], "noprev": ["--no-prev"]}


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], phase: str) -> None:
    log(f"START {phase}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    log(f"=== prev A/B start (ctx={CTX}, full/predlen4/1000steps/lr1e-6/batch64) ===")
    for target, ctx in CTX.items():
        for arm, extra in ARMS.items():
            tag = f"ctx{ctx}_{arm}"
            ckpt = ROOT / "checkpoints" / f"{target}_{tag}" / "final"
            if (ckpt / "config.json").exists():
                log(f"SKIP  finetune {target} {arm} (checkpoint exists)")
            else:
                run([PY, "train/finetune_chronos2.py", "--target", target,
                     "--finetune-mode", "full", "--prediction-length", "4",
                     "--context-length", str(ctx), "--learning-rate", "1e-6",
                     "--num-steps", "1000", "--batch-size", "64",
                     "--output-tag", tag, *extra],
                    f"finetune {target} {arm} ctx{ctx}")

            csv = ROOT / "eval" / f"backtest_{target}_{tag}.csv"
            if csv.exists():
                log(f"SKIP  backtest {target} {arm} (csv exists)")
            else:
                run([PY, "eval/backtest.py", "--target", target, "--checkpoint", str(ckpt),
                     "--context-length", str(ctx), "--stride", "4", "--tag", tag, *extra],
                    f"backtest {target} {arm}")
    log("=== prev A/B complete ===")


if __name__ == "__main__":
    main()
