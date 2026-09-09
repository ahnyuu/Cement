"""
REVISED 2026-09-09: validation-only study; outputs under review_v2.
The original study description below is retained as historical context.

Full context_length parameter study, reproducing Cement_code/chronos_quality's
results_context_comparison.ipynb matrix on the 운전&품질데이터_실측치.xlsx data:

    context_length in {24, 73, 128, 168, 256, 384, 512, 768, 1024, 1280, 1536}
    target          in {blaine, residue}
    method          in {zero-shot, LoRA, Full fine-tune}

Identical hyperparameters to the old folder (run_ctx_sweep_v2.sh / run_measured_actual_experiment.sh):
    prediction_length 4 (train) / 1 (eval)   num_steps 1000   batch_size 64   stride 4
    Full: lr 1e-6      LoRA: lr 1e-5      zero-shot: no training, backtest amazon/chronos-2

Outputs (tags chosen so the 5 ctx already run by run_ctx_sweep.py are reused as-is):
    checkpoints/{target}_ctx{N}/          Full
    checkpoints/{target}_ctx{N}_lora/     LoRA
    eval/backtest_{target}_ctx{N}_{full,lora,zeroshot}.csv

Resumable: any step whose output exists is skipped. Logged to experiments/ctx_matrix.log.

Run:  .venv/Scripts/python.exe experiments/run_ctx_matrix.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.research_protocol import checkpoint_complete, evaluation_complete

PY = sys.executable
LOG = ROOT / "experiments" / "ctx_matrix_review_v2.log"

TARGETS = ["blaine", "residue"]
CONTEXT_LENGTHS = [24, 73, 128, 168, 256, 384, 512, 768, 1024, 1280, 1536]
NUM_STEPS = 1000
BATCH_SIZE = 64
PRED_LEN = 4
STRIDE = 4
LR = {"full": "1e-6", "lora": "1e-5"}


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], phase: str) -> None:
    log(f"START {phase}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = (time.time() - t0) / 60
    tail = "\n".join((proc.stdout or "").splitlines()[-20:])
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"  cmd: {' '.join(cmd)}\n----- stdout tail ({phase}) -----\n{tail}\n")
        if proc.returncode != 0:
            f.write(f"----- stderr ({phase}) -----\n{proc.stderr}\n")
    if proc.returncode != 0:
        log(f"FAIL  {phase} (exit {proc.returncode}, {dt:.1f} min)")
        raise SystemExit(f"{phase} failed -- see {LOG}")
    log(f"DONE  {phase} ({dt:.1f} min)")


def finetune(target: str, ctx: int, mode: str) -> Path:
    tag = f"ctx{ctx}" if mode == "full" else f"ctx{ctx}_lora"
    ckpt_dir = ROOT / "checkpoints" / "review_v2" / f"{target}_{tag}"
    if checkpoint_complete(ckpt_dir / "final"):
        log(f"SKIP  finetune {target} {mode} ctx{ctx} (checkpoint exists)")
        return ckpt_dir / "final"
    run(
        [PY, "train/finetune_chronos2.py", "--target", target,
         "--finetune-mode", mode, "--prediction-length", str(PRED_LEN),
         "--context-length", str(ctx), "--learning-rate", LR[mode],
         "--num-steps", str(NUM_STEPS), "--batch-size", str(BATCH_SIZE),
         "--output-tag", tag],
        f"finetune {target} {mode} ctx{ctx}",
    )
    return ckpt_dir / "final"


def backtest(target: str, ctx: int, method: str, checkpoint: str) -> None:
    csv = ROOT / "eval" / "review_v2" / "validation" / f"backtest_{target}_ctx{ctx}_{method}.csv"
    if evaluation_complete(csv):
        log(f"SKIP  backtest {target} {method} ctx{ctx} (csv exists)")
        return
    run(
        [PY, "eval/backtest.py", "--split", "validation", "--target", target, "--checkpoint", checkpoint,
         "--context-length", str(ctx), "--stride", str(STRIDE), "--tag", f"ctx{ctx}_{method}"],
        f"backtest {target} {method} ctx{ctx}",
    )


def main() -> None:
    print("review_v2: parameter selection uses VALIDATION; historical outputs are not reused.")
    log(f"=== ctx matrix start (targets={TARGETS}, ctx={CONTEXT_LENGTHS}, "
        f"methods=zeroshot/lora/full, steps={NUM_STEPS}) ===")
    for ctx in CONTEXT_LENGTHS:
        for target in TARGETS:
            # zero-shot: no training
            backtest(target, ctx, "zeroshot", "amazon/chronos-2")
            # LoRA
            lora_ckpt = finetune(target, ctx, "lora")
            backtest(target, ctx, "lora", str(lora_ckpt))
            # Full
            full_ckpt = finetune(target, ctx, "full")
            backtest(target, ctx, "full", str(full_ckpt))
    log("=== ctx matrix complete ===")


if __name__ == "__main__":
    main()
