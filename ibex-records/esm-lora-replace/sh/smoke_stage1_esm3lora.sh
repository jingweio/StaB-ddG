#!/bin/bash
#SBATCH --job-name=smoke_s1_esm3lora
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/smoke_s1_esm3lora_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/smoke_s1_esm3lora_%j.err
set -euo pipefail

# Use the env's python directly (conda activate is unreliable on non-login shells).
ENVBIN=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin
# Override HF cache to the Ibex location (the scorer's module-level setdefault
# points at a LOCAL path that does not exist here; an explicit export wins over setdefault).
export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test

# Stage-1 (Megascale folding) GPU smoke: 3 domains, 1 epoch. Validates the bf16
# autocast + out.embeddings path on a100 (the residual the CPU smoke could not cover)
# and that a small LoRA+head adapter ckpt is produced.
srun $ENVBIN/python -m run.finetune --backbone esm3_stab --stage stability \
    --epochs 1 --limit 3 --batch_size 2000 \
    --out cache/s1_smoke --run_name esm3stab_s1_smoke
