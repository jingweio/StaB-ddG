#!/bin/bash
#SBATCH --job-name=twostage_esmc6b
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=70:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/twostage_esmc6b_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/twostage_esmc6b_%j.err

# Full two-stage StaB-ddG pipeline for the ESMC-6B backbone via FSDP/ZeRO-3 on
# 4xa100 (8-GPU full nodes don't schedule on this account; 4xa100 validated by
# the pilot):
#   stage1 = Megascale folding-stability finetune (FSDP)
#   stage2 = SKEMPI binding finetune (FSDP), initialised from the stage-1 ckpt
#   eval   = single-GPU SKEMPI test-split eval of the consolidated stage-2 ckpt
# The FSDP harness gathers a consolidated FULL_STATE_DICT per epoch named
# `{run_name}_{epoch+1}.pt` (fsdp_train_esmc6b.py:472), loadable into a plain
# ESMCScorer — so `--checkpoint` chains stages and the final eval runs on 1 GPU.
# Per-stage per-epoch checkpoints are saved (resumable). Outputs -> results/.

set -euo pipefail

# ----------------------------- tunable params ----------------------------- #
LR=1e-5                 # full-FT of pretrained ESM-C (1e-6 is for tiny ProteinMPNN)
S1=15                   # stage-1 (Megascale) epochs (converges ~15; ESM3 evidence)
S2=25                   # stage-2 (SKEMPI) epochs
BS=8000                 # token budget per forward (validated by the 4xa100 pilot)
RUN1=esmc6b_stage1
RUN2=esmc6b_stage2
CKPT_DIR=track_esmc/ckpts
NPROC=4
# --------------------------------------------------------------------------- #

# Use the env's bin directly (robust; `conda activate` is unreliable in non-login shells).
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
echo "python -> $(which python); torchrun -> $(which torchrun)"

export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test
mkdir -p "${CKPT_DIR}" results

# ----------------------- Stage 1: Megascale folding ----------------------- #
torchrun --standalone --nproc_per_node="${NPROC}" -m track_esmc.fsdp_train_esmc6b \
  --backbone esmc_6b --stage stability \
  --lr "${LR}" --epochs "${S1}" --batch_size "${BS}" \
  --out "${CKPT_DIR}" --run_name "${RUN1}" \
  --use_flash_attn 1 --mixed_precision bf16

# ------------------- Stage 2: SKEMPI binding (chained) -------------------- #
torchrun --standalone --nproc_per_node="${NPROC}" -m track_esmc.fsdp_train_esmc6b \
  --backbone esmc_6b --stage skempi \
  --checkpoint "${CKPT_DIR}/${RUN1}_${S1}.pt" \
  --lr "${LR}" --epochs "${S2}" --batch_size "${BS}" \
  --out "${CKPT_DIR}" --run_name "${RUN2}" \
  --use_flash_attn 1 --mixed_precision bf16

# --------------- Eval: SKEMPI test split (single-GPU, plain ESMCScorer) --- #
# The consolidated stage-2 ckpt loads into a plain ESMCScorer.backbone_module.
python -m run.eval \
  --backbone esmc_6b \
  --checkpoint "${CKPT_DIR}/${RUN2}_${S2}.pt" \
  --out results/eval_esmc6b_twostage.csv \
  --ensemble 1
