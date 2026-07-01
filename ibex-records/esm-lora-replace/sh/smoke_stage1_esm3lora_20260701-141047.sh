#!/bin/bash
#SBATCH --job-name=smoke_s1_esm3lora
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/smoke_s1_esm3lora_20260701-141047_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/smoke_s1_esm3lora_20260701-141047_%j.err
set -euo pipefail

ENVBIN=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin   # py3.12 faithful env
export HF_HOME=/ibex/user/guoj0f/share/hf_cache            # shared store (biohub esm3 weights)
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test

# GPU smoke: Stage-1 (Megascale folding) on 3 domains, 1 epoch. Validates the full Ibex
# stack on a100 — new py3.12 env + esm 3.3.0 + biohub weights + bf16 out.embeddings + LoRA+head.
srun $ENVBIN/python -m run.finetune --backbone esm3_stab --stage stability \
    --epochs 1 --limit 3 --batch_size 2000 \
    --out cache/s1_smoke --run_name esm3stab_s1_smoke
