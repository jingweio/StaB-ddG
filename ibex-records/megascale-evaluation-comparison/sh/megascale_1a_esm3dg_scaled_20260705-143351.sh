#!/bin/bash
#SBATCH --job-name=megascale_1a_scaled
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/megascale-evaluation-comparison/megascale_1a_esm3dg_scaled_20260705-143351_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/megascale-evaluation-comparison/megascale_1a_esm3dg_scaled_20260705-143351_%j.err
# Task1a RE-RUN: ESM3dG (base 3-ens) on Megascale test, now reporting scaled(primary)+raw.
# (Megascale is cDNA -> paper rule sigmoid on = scaled.) 1b/ProteinMPNN result stands, not re-run.
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export HF_HOME=/ibex/user/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg_stab
W=../data/esm3dg_weights
R=../ibex-records/megascale-evaluation-comparison/results
echo "===== Task1a (re-run, scaled+raw): ESM3dG base 3-ens on Megascale test ====="
srun python -u eval_megascale.py \
  --members $W/ESM3dG_weights_1_lora.ckpt,$W/ESM3dG_weights_2_lora.ckpt,$W/ESM3dG_weights_3_lora.ckpt \
  --split test --max_batch 32 --out $R/eval_megascale_esm3dg_base_ens_scaledraw.csv
echo "===== DONE ====="
