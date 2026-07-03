#!/bin/bash
#SBATCH --job-name=esm3dg_repro_mgnify
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-reproduce-mgnify/repro_mgnify_base_ens_20260703-024719_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-reproduce-mgnify/repro_mgnify_base_ens_20260703-024719_%j.err
# Task2: reproduce paper ESM3ΔG absolute-dG on MGnify test (3283 seqs, base 3-member ensemble)
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export HF_HOME=/ibex/user/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg-reproduce
W=../data/esm3dg_weights
echo "===== Task2: ESM3ΔG (base 3-member ensemble) on MGnify test ====="
srun python -u eval_mgnify.py \
  --members $W/ESM3dG_weights_1_lora.ckpt,$W/ESM3dG_weights_2_lora.ckpt,$W/ESM3dG_weights_3_lora.ckpt \
  --index_csv ../data/mgnify/mgnify_training_index.csv \
  --struct_dir ../data/mgnify/structures_test --split test \
  --out ../ibex-records/ESM3ΔG-reproduce-mgnify/results/eval_mgnify_test_base_ens.csv
echo "===== DONE ====="
