#!/bin/bash
#SBATCH --job-name=megascale_stage1_esm3lora
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/megascale_stage1_esm3lora_20260701-141047_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/megascale_stage1_esm3lora_20260701-141047_%j.err
set -euo pipefail

ENVBIN=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin   # py3.12 faithful env
export HF_HOME=/ibex/user/guoj0f/share/hf_cache            # shared store (biohub esm3 weights)
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test

# Stage-1: Megascale folding-stability LoRA+head finetune (239 train domains).
# HP (user-confirmed, reference config_esm3.py): lr 1e-3 / AdamW / warmup 0.05 / cosine / 15 epochs
# / batch 2000 tokens. LoRA r=4 + dropout 0.15 are scorer defaults. Per-epoch adapter ckpts (~13MB).
srun $ENVBIN/python -m run.finetune --backbone esm3_stab --stage stability \
    --epochs 15 --batch_size 2000 \
    --optimizer adamw --weight_decay 0.0 --lr 1e-3 --warmup_frac 0.05 --lr_schedule cosine \
    --out /ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s1_esm3lora --run_name esm3lora_s1
