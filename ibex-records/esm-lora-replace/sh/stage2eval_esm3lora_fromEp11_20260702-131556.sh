#!/bin/bash
#SBATCH --job-name=s2eval_esm3lora_ep11
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/stage2eval_esm3lora_fromEp11_20260702-131556_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/stage2eval_esm3lora_fromEp11_20260702-131556_%j.err
set -euo pipefail

ENVBIN=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin   # py3.12 faithful env
export HF_HOME=/ibex/user/guoj0f/share/hf_cache            # shared store (biohub esm3 weights)
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test

S1CKPT=/ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s1_esm3lora_lr5e4/esm3lora_s1_lr5e4_11.pt   # Stage-1 train-loss min
OUT=/ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s2_esm3lora_fromEp11

# Stage-2: SKEMPI binding-ddG finetune (StaBddG complex-binder1-binder2 decomposition + ddG MSE),
# chained from Stage-1 ep11. HP: lr 5e-4 / AdamW / warmup 0.05 / cosine / 15 ep / batch 1000 tokens
# (lr 5e-4 proven stable in Stage-1; 1e-3 diverged). Per-epoch adapter ckpts.
srun $ENVBIN/python -m run.finetune --backbone esm3_stab --stage skempi \
    --checkpoint $S1CKPT \
    --epochs 15 --batch_size 1000 \
    --optimizer adamw --weight_decay 0.0 --lr 5e-4 --warmup_frac 0.05 --lr_schedule cosine \
    --out $OUT --run_name esm3lora_s2_ep11

# Eval final Stage-2 ckpt on SKEMPI test (homology-OOD): per-interface Spearman vs ProteinMPNN 0.445.
srun $ENVBIN/python -m run.eval --backbone esm3_stab \
    --checkpoint $OUT/esm3lora_s2_ep11_15.pt \
    --split test --ensemble 1 \
    --out $OUT/eval_test_ep11.csv
