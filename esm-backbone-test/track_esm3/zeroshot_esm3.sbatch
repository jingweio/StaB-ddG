#!/bin/bash
#SBATCH --job-name=zs_esm3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/zeroshot_esm3_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/zeroshot_esm3_%j.err

# ESM3 (structure-conditioned) zero-shot ΔΔG eval on the SKEMPI OOD test split.
# ensemble=1: ESM3 is deterministic given (sequence, structure) tokens, so MC>1 is
# unnecessary (the StaB MC ensembling exists only for ProteinMPNN's stochastic decoding).
set -euo pipefail
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
echo "python -> $(which python)"
export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test
python -m run.zeroshot_eval --backbone esm3 --out results/zeroshot_esm3.csv --ensemble 1 --device cuda
echo "ZS_ESM3_DONE"
