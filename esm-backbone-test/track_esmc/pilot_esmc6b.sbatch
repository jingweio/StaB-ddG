#!/bin/bash
#SBATCH --job-name=pilot_esmc6b
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/pilot_esmc6b_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/pilot_esmc6b_%j.err

# Short 8xa100 pilot to validate ESMC-6B FSDP/ZeRO-3 memory + harness mechanics
# (multi-shard wrap, 6-graph step, consolidated checkpoint save) on a tiny SKEMPI
# subset BEFORE the full two-stage run. Do NOT submit from the worker session —
# this is the human-gated pilot. Peak mem -> results/esmc6b_pilot.md.

set -euo pipefail

# Use the env's bin directly (robust; `conda activate` is unreliable in non-login shells).
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
echo "python -> $(which python); torchrun -> $(which torchrun)"

export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test

# flash-attn IS installed on Ibex -> use_flash_attn 1; bf16 mixed precision on a100.
# -m module form: fsdp_train_esmc6b.py prepends the worktree root + esm-backbone-test
# onto sys.path at import, so `track_esmc` / `common` / `stabddg` all resolve.
torchrun --standalone --nproc_per_node=8 -m track_esmc.fsdp_train_esmc6b \
  --backbone esmc_6b --stage skempi --lr 1e-5 --epochs 1 --batch_size 8000 \
  --limit 5 --use_flash_attn 1 --mixed_precision bf16 \
  --out track_esmc/ckpts --run_name pilot_esmc6b
