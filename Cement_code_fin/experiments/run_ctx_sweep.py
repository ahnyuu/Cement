"""
context_length sweep: {512, 768, 1024, 1280, 1536} x {blaine, residue}.

For each (target, ctx):
  1. full fine-tune  -> checkpoints/{target}_ctx{ctx}/final
  2. zero-shot backtest at that ctx -> eval/backtest_{target}_ctx{ctx}_zeroshot.csv
  3. fine-tuned backtest at that ctx -> eval/backtest_{target}_ctx{ctx}_full.csv

Resumable: any step whose output already exists is skipped. All hyperparameters other than
context_length are the EXPERIMENT_LOG-confirmed values (predlen 4 train, full, 1000 steps,
lr 1e-6, batch 64). Everything is logged to experiments/ctx_sweep.log.

Run (from the project root):
    .venv/Scripts/python.exe experiments/run_ctx_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "experiments" / "ctx_sweep.log"

TARGETS = ["blaine", "residue"]
CONTEXT_LENGTHS = [512, 768, 1024, 1280, 1536]
NUM_STEPS = 1000
STRIDE = 4  # backtest: evaluate every Nth fresh-reading row


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], phase: str) -> None:
    log(f"START {phase}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = "\n".join((proc.stdout or "").splitlines()[-25:])
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"----- stdout tail ({phase}) -----\n{tail}\n")
        if proc.returncode != 0:
            f.write(f"----- stderr ({phase}) -----\n{proc.stderr}\n")
    if proc.returncode != 0:
        log(f"FAIL  {phase} (exit {proc.returncode}, {dt/60:.1f} min) -- see log stderr")
        raise SystemExit(f"{phase} failed")
    log(f"DONE  {phase} ({dt/60:.1f} min)")


def main() -> None:
    log(f"=== ctx sweep start (targets={TARGETS}, ctx={CONTEXT_LENGTHS}, steps={NUM_STEPS}) ===")
    for target in TARGETS:
        for ctx in CONTEXT_LENGTHS:
            tag = f"ctx{ctx}"
            ckpt = ROOT / "checkpoints" / f"{target}_{tag}" / "final"
            zs_csv = ROOT / "eval" / f"backtest_{target}_{tag}_zeroshot.csv"
            ft_csv = ROOT / "eval" / f"backtest_{target}_{tag}_full.csv"

            if (ckpt / "config.json").exists():
                log(f"SKIP  finetune {target} {tag} (checkpoint exists)")
            else:
                run(
                    [PY, "train/finetune_chronos2.py", "--target", target,
                     "--context-length", str(ctx), "--num-steps", str(NUM_STEPS),
                     "--output-tag", tag],
                    f"finetune {target} {tag}",
                )

            if zs_csv.exists():
                log(f"SKIP  backtest {target} {tag} zeroshot (csv exists)")
            else:
                run(
                    [PY, "eval/backtest.py", "--target", target,
                     "--checkpoint", "amazon/chronos-2", "--context-length", str(ctx),
                     "--stride", str(STRIDE), "--tag", f"{tag}_zeroshot"],
                    f"backtest {target} {tag} zeroshot",
                )

            if ft_csv.exists():
                log(f"SKIP  backtest {target} {tag} full (csv exists)")
            else:
                run(
                    [PY, "eval/backtest.py", "--target", target,
                     "--checkpoint", str(ckpt), "--context-length", str(ctx),
                     "--stride", str(STRIDE), "--tag", f"{tag}_full"],
                    f"backtest {target} {tag} full",
                )
    log("=== ctx sweep complete ===")


if __name__ == "__main__":
    main()
