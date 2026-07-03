#!/bin/bash
#SBATCH --job-name=megascale_eval_cmp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/megascale-evaluation-comparison/megascale_eval_esm3dg_vs_pmpnn_20260703-024719_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/megascale-evaluation-comparison/megascale_eval_esm3dg_vs_pmpnn_20260703-024719_%j.err
# Task1: Megascale test folding-ddG — (1a) ESM3dG base 3-ens  vs  (1b) ProteinMPNN stage1 (20x MC)
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export HF_HOME=/ibex/user/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg_stab
W=../data/esm3dg_weights
R=../ibex-records/megascale-evaluation-comparison/results

echo "===== Task1a: ESM3dG (base 3-member ensemble) on Megascale test ====="
srun python -u eval_megascale.py \
  --members $W/ESM3dG_weights_1_lora.ckpt,$W/ESM3dG_weights_2_lora.ckpt,$W/ESM3dG_weights_3_lora.ckpt \
  --split test --max_batch 32 --out $R/eval_megascale_esm3dg_base_ens.csv

echo "===== Task1b: ProteinMPNN stage1 (stability_finetuned.pt, 20x MC) on Megascale test ====="
srun python -u eval_megascale_pmpnn.py --split test --mc 20 \
  --out $R/eval_megascale_pmpnn_stage1_mc20.csv
echo "===== DONE ====="
