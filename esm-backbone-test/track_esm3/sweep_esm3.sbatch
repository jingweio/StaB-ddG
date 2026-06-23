#!/bin/bash
#SBATCH --job-name=sweep_esm3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/sweep_esm3_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/sweep_esm3_%j.err

set -euo pipefail

# Fill in the conda base path before submitting:
#   CONDA_BASE=$(conda info --base)  # typically /ibex/user/guoj0f/anaconda3
# Replace <CONDA_BASE> below with the output of that command.
CONDA_BASE=<CONDA_BASE>

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate esm-backbone

export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test

mkdir -p results/sweep_esm3

python -m run.lr_sweep \
    --backbone esm3 \
    --lrs "1e-6,5e-6,1e-5,3e-5" \
    --epochs 5 \
    --batch_size 5000 \
    --device cuda \
    --out_dir results/sweep_esm3
