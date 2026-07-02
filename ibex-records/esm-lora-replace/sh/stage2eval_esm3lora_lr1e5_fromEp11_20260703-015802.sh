#!/bin/bash
#SBATCH --job-name=s2_esm3lr1e5_ep11
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/stage2eval_esm3lora_lr1e5_fromEp11_20260703-015802_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/stage2eval_esm3lora_lr1e5_fromEp11_20260703-015802_%j.err
set -euo pipefail

ENVBIN=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin   # py3.12 faithful env
export HF_HOME=/ibex/user/guoj0f/share/hf_cache            # shared store (biohub esm3 weights)
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test

S1CKPT=/ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s1_esm3lora_lr5e4/esm3lora_s1_lr5e4_11.pt   # Stage-1 train-loss min
OUT=/ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s2_esm3lora_lr1e5_fromEp11

# Stage-2 RETRY at lr=1e-5 (USER decision 2026-07-03): the lr=5e-4 Stage-2 FAILED — binding-ddG
# predictions collapsed to ~0 (pred std 0.005 vs true 1.9), per-interface Spearman -0.05, train loss
# flat ~6.35 all 15 ep. Retry with lr=1e-5 (= StaB finetune.py default; 50x lower). All else identical:
# AdamW / warmup 0.05 / cosine / 15 ep / batch 1000 tokens. NAMING isolated (lr1e5) from prior runs.
srun $ENVBIN/python -m run.finetune --backbone esm3_stab --stage skempi \
    --checkpoint $S1CKPT \
    --epochs 15 --batch_size 1000 \
    --optimizer adamw --weight_decay 0.0 --lr 1e-5 --warmup_frac 0.05 --lr_schedule cosine \
    --out $OUT --run_name esm3lora_s2lr1e5_ep11

# Eval final Stage-2 ckpt on SKEMPI test (homology-OOD): per-interface Spearman vs ProteinMPNN 0.445.
srun $ENVBIN/python -m run.eval --backbone esm3_stab \
    --checkpoint $OUT/esm3lora_s2lr1e5_ep11_15.pt \
    --split test --ensemble 1 \
    --out $OUT/eval_test_lr1e5_ep11.csv
