"""
REVISED 2026-09-09: validation-only study; outputs under review_v2.
The original study description below is retained as historical context.

prediction_length (train) 1 vs 4 A/B.

Reproduces the old folder's 2026-08-18 predlen sweep on the measured-actual data, at the
2026-09-06 sweep-optimal context_length (blaine 512 / residue 256).

  arm "predlen4" : --prediction-length 4  (current canonical)
  arm "predlen1" : --prediction-length 1

Eval is always 1-step (backtest.py hardcodes predict_df prediction_length=1 -- chronos-2 handles
inference at a shorter horizon than training fine), so "1-step forecast accuracy" stays the same
yardstick for both arms. Everything else identical: canonical CSV, full fine-tune, 1000 steps,
lr 1e-6, batch 64, stride 4. Same eval points / naive baseline in both arms.

Outputs:
  checkpoints/{target}_ctx{N}_predlen{1,4}/
  eval/backtest_{target}_ctx{N}_predlen{1,4}.csv

Resumable. Logged to experiments/predlen/predlen_experiment.log.

    .venv/Scripts/python.exe experiments/predlen/run_predlen_experiment.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.research_protocol import checkpoint_complete, evaluation_complete

PY = sys.executable
LOG = ROOT / "experiments" / "predlen" / "predlen_experiment_review_v2.log"

CTX = {"blaine": 512, "residue": 256}
PREDLENS = [4, 1]


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
    print("review_v2: parameter selection uses VALIDATION; historical outputs are not reused.")
    log(f"=== predlen A/B start (ctx={CTX}, predlens={PREDLENS}, full/1000steps/lr1e-6/batch64) ===")
    for target, ctx in CTX.items():
        for pl in PREDLENS:
            tag = f"ctx{ctx}_predlen{pl}"
            ckpt = ROOT / "checkpoints" / "review_v2" / f"{target}_{tag}" / "final"
            if checkpoint_complete(ckpt):
                log(f"SKIP  finetune {target} predlen{pl} (checkpoint exists)")
            else:
                run([PY, "train/finetune_chronos2.py", "--target", target,
                     "--finetune-mode", "full", "--prediction-length", str(pl),
                     "--context-length", str(ctx), "--learning-rate", "1e-6",
                     "--num-steps", "1000", "--batch-size", "64", "--output-tag", tag],
                    f"finetune {target} predlen{pl} ctx{ctx}")

            csv = ROOT / "eval" / "review_v2" / "validation" / f"backtest_{target}_{tag}.csv"
            if evaluation_complete(csv):
                log(f"SKIP  backtest {target} predlen{pl} (csv exists)")
            else:
                run([PY, "eval/backtest.py", "--split", "validation", "--target", target, "--checkpoint", str(ckpt),
                     "--context-length", str(ctx), "--stride", "4", "--tag", tag],
                    f"backtest {target} predlen{pl}")
    log("=== predlen A/B complete ===")


if __name__ == "__main__":
    main()
