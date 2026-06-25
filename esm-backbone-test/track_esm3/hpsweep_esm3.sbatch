#!/bin/bash
#SBATCH --job-name=hpsweep_esm3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=20:00:00
# NOTE: SLURM does NOT expand shell/env vars (e.g. ${RUN}) inside #SBATCH
# directives — only its own patterns (%j jobid, %x job-name). The output/error
# files therefore use %x_%j; to fold RUN into the filename, override BOTH the
# job-name and output/error on the sbatch command line (the login shell expands
# them before SLURM sees them), e.g.:
#   sbatch --job-name=hpsweep_lr3e5 \
#          --output=/ibex/.../results/hpsweep_lr3e5_%j.out \
#          --error=/ibex/.../results/hpsweep_lr3e5_%j.err \
#          --export=ALL,LR=3e-5,RUN=lr3e5 track_esm3/hpsweep_esm3.sbatch
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/hpsweep_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/hpsweep_%x_%j.err

# ESM3 Stage-2 (SKEMPI binding) hyperparameter-sweep job.
#
# Goal: determine whether ESM3's poor two-stage result (per-interface Spearman
# ~0.104 vs ProteinMPNN 0.445) is under-adapted training hyperparameters vs a
# real limitation. The smoking gun is that the Stage-2 train loss plateaued at
# ~3.4 (~the ddG variance) — it isn't fitting even the TRAIN set. This job
# finetunes Stage-2 from a fixed Stage-1 checkpoint with one HP config, then
# evals BOTH the train and test splits and appends a single summary row so the
# train-fit vs test-generalization can be compared across configs.
#
# Each parameter is an env var with a default, overridable per launch via
#   sbatch --export=ALL,LR=...,RUN=...   track_esm3/hpsweep_esm3.sbatch
# so one job can be launched per LR (or per recipe variant).
#
# Tunable params (env vars; defaults shown):
#   LR=1e-5            Stage-2 learning rate
#   WARMUP=0           linear LR-warmup fraction of total steps
#   WD=0               weight decay (only meaningful with OPT=adamw)
#   OPT=adam           adam | adamw
#   SCHED=constant     constant | cosine
#   EPOCHS=15          Stage-2 epochs
#   STAGE1_CKPT=track_esm3/ckpts/esm3_stage1_14.pt   fixed Stage-1 init
#   RUN=hp             run tag (names ckpts / CSVs / summary row)
set -euo pipefail

# --------------------------- tunable params -------------------------------- #
LR=${LR:-1e-5}
WARMUP=${WARMUP:-0}
WD=${WD:-0}
OPT=${OPT:-adam}
SCHED=${SCHED:-constant}
EPOCHS=${EPOCHS:-15}
STAGE1_CKPT=${STAGE1_CKPT:-track_esm3/ckpts/esm3_stage1_14.pt}
RUN=${RUN:-hp}
CKPT_DIR=track_esm3/ckpts
# --------------------------------------------------------------------------- #

# Use the env's bin directly (robust; `conda activate` is unreliable in non-login shells).
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
echo "python -> $(which python)"
export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test
mkdir -p "${CKPT_DIR}" results

echo "[hpsweep] RUN=${RUN} LR=${LR} WARMUP=${WARMUP} WD=${WD} OPT=${OPT} SCHED=${SCHED} EPOCHS=${EPOCHS} STAGE1_CKPT=${STAGE1_CKPT}"

FT_CKPT="${CKPT_DIR}/esm3_${RUN}_${EPOCHS}.pt"
TRAIN_LOG="${CKPT_DIR}/esm3_${RUN}_train_log.csv"
TEST_CSV="results/eval_esm3_${RUN}_test.csv"
TRAIN_CSV="results/eval_esm3_${RUN}_train.csv"
SWEEP_CSV="results/esm3_hp_sweep.csv"

# ---------------- 1) Stage-2 SKEMPI finetune with this HP config ----------- #
python -m run.finetune \
  --backbone esm3 --stage skempi \
  --checkpoint "${STAGE1_CKPT}" \
  --lr "${LR}" --epochs "${EPOCHS}" --batch_size 1500 \
  --optimizer "${OPT}" --weight_decay "${WD}" \
  --warmup_frac "${WARMUP}" --lr_schedule "${SCHED}" \
  --out "${CKPT_DIR}" --run_name "esm3_${RUN}"

# ----------------------- 2) Eval the TEST split ---------------------------- #
python -m run.eval \
  --backbone esm3 --checkpoint "${FT_CKPT}" \
  --split test --out "${TEST_CSV}" --ensemble 1

# ---------------------- 3) Eval the TRAIN split ---------------------------- #
python -m run.eval \
  --backbone esm3 --checkpoint "${FT_CKPT}" \
  --split train --out "${TRAIN_CSV}" --ensemble 1

# ---- 4) Per-structure Spearman on both CSVs -> append a sweep summary row -- #
# Reads the per-structure Spearman from baselines.eval_utils.compute_metrics on
# BOTH eval CSVs and the final train loss from the last row of the finetune
# train-log CSV, then appends one row to results/esm3_hp_sweep.csv (header is
# written once if the file does not yet exist). Params are passed via env vars.
RUN="${RUN}" LR="${LR}" WARMUP="${WARMUP}" WD="${WD}" OPT="${OPT}" \
SCHED="${SCHED}" EPOCHS="${EPOCHS}" TRAIN_LOG="${TRAIN_LOG}" \
TRAIN_CSV="${TRAIN_CSV}" TEST_CSV="${TEST_CSV}" SWEEP_CSV="${SWEEP_CSV}" \
python - <<'PY'
import csv
import os
import sys

# `baselines` lives at the worktree root (one level above esm-backbone-test, the
# cwd of this job). run/eval.py and run/finetune.py do the same bootstrap; mirror
# it here so `from baselines.eval_utils import compute_metrics` resolves.
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

import pandas as pd

from baselines.eval_utils import compute_metrics


def per_structure_spearman(csv_path):
    """Per-structure Spearman from a baseline-format eval CSV (NaN if missing/empty)."""
    if not csv_path or not os.path.exists(csv_path):
        return float("nan")
    df = pd.read_csv(csv_path)
    if df.empty:
        return float("nan")
    try:
        return float(compute_metrics(df, bootstrap=False)["Per Structure Spearman"])
    except Exception as e:  # too few per-structure groups, etc.
        print(f"[summary] compute_metrics failed on {csv_path}: {e!r}", flush=True)
        return float("nan")


def final_train_loss(train_log):
    """mean_loss from the LAST row of the finetune train-log CSV (NaN if missing)."""
    if not train_log or not os.path.exists(train_log):
        return float("nan")
    df = pd.read_csv(train_log)
    if df.empty or "mean_loss" not in df.columns:
        return float("nan")
    return float(df["mean_loss"].iloc[-1])


train_sp = per_structure_spearman(os.environ["TRAIN_CSV"])
test_sp = per_structure_spearman(os.environ["TEST_CSV"])
ftl = final_train_loss(os.environ["TRAIN_LOG"])

sweep_csv = os.environ["SWEEP_CSV"]
os.makedirs(os.path.dirname(sweep_csv) or ".", exist_ok=True)
header = ["run", "lr", "warmup", "wd", "opt", "sched", "epochs",
          "final_train_loss", "train_spearman", "test_spearman"]
row = [os.environ["RUN"], os.environ["LR"], os.environ["WARMUP"],
       os.environ["WD"], os.environ["OPT"], os.environ["SCHED"],
       os.environ["EPOCHS"], f"{ftl:.6f}", f"{train_sp:.6f}", f"{test_sp:.6f}"]

write_header = not os.path.exists(sweep_csv)
with open(sweep_csv, "a", newline="") as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(header)
    w.writerow(row)

print(f"[summary] appended to {sweep_csv}: "
      f"run={row[0]} final_train_loss={ftl:.4f} "
      f"train_spearman={train_sp:.4f} test_spearman={test_sp:.4f}", flush=True)
PY

echo "ESM3_HPSWEEP_DONE RUN=${RUN}"
