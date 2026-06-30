#!/bin/bash
#SBATCH --job-name=esm3dg_skempi_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/smoke_esm3dg_skempi_20260630-115200_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/smoke_esm3dg_skempi_20260630-115200_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm-backbone
export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg_stab
LORA=../data/esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===== SMOKE: 1 epoch, 12 train complexes ====="
srun python -u finetune.py --stage skempi --lora_ckpt $LORA --split train --limit 12 \
  --lr 6e-5 --optimizer adamw --weight_decay 0.05 --epochs 1 --batch_tokens 4000 --max_batch 4 \
  --out ../cache/smoke_skempi_a100.pt --run_name smoke_skempi_a100
echo "===== SMOKE EVAL: 12 test complexes ====="
srun python -u eval.py --lora_ckpt $LORA --checkpoint ../cache/smoke_skempi_a100.pt --split test --limit 12 \
  --batch_tokens 4000 --max_batch 4
echo "===== SMOKE DONE ====="
