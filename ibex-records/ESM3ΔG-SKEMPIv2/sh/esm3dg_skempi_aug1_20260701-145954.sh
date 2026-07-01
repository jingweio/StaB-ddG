#!/bin/bash
#SBATCH --job-name=esm3dg_skempi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/esm3dg_skempi_aug1_20260701-145954_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/esm3dg_skempi_aug1_20260701-145954_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm-backbone   # py3.12 + torch2.6 + esm(from share/esm) — rebuilt via share/ibex_build_env_offline.sh
export HF_HOME=/ibex/user/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg_stab
LORA=../data/esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt
RUN=esm3dg_skempi_aug1
echo "===== TASK1: ESM3ΔG -> SKEMPI (clean share/esm) ====="
srun python -u finetune.py --stage skempi --lora_ckpt $LORA --split train \
  --lr 6e-5 --optimizer adamw --weight_decay 0.05 --epochs 15 \
  --batch_tokens 4000 --max_batch 4 --seed 0 --save_freq 5 \
  --out ../cache/$RUN.pt --run_name $RUN
echo "===== EVAL on SKEMPI test ====="
srun python -u eval.py --lora_ckpt $LORA --checkpoint ../cache/$RUN.pt --split test \
  --batch_tokens 4000 --max_batch 4 --out ../ibex-records/ESM3ΔG-SKEMPIv2/results/eval_$RUN.csv
echo "===== DONE ====="
