#!/bin/bash
#SBATCH --job-name=esm3dg_mega_skempi
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=23:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-Megascale-SKEMPIv2/esm3dg_mega_skempi_aug1_20260701-145954_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-Megascale-SKEMPIv2/esm3dg_mega_skempi_aug1_20260701-145954_%j.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg   # this task's DEDICATED env (py3.12+torch2.6+esm from share/esm; built offline from per-branch wheelhouse)
export HF_HOME=/ibex/user/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg_stab
LORA=../data/esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt
RUN=esm3dg_mega_skempi_aug1
echo "===== TASK2 stage-1: ESM3ΔG -> Megascale folding ddG (single_batch) ====="
srun python -u finetune.py --stage megascale --lora_ckpt $LORA --split train --single_batch \
  --lr 6e-5 --optimizer adamw --weight_decay 0.05 --epochs 15 --batch_tokens 4000 --max_batch 8 --seed 0 \
  --out ../cache/${RUN}_stage1.pt --run_name ${RUN}_stage1
echo "===== TASK2 stage-2: -> SKEMPI binding ddG (resume stage-1) ====="
srun python -u finetune.py --stage skempi --lora_ckpt $LORA --resume ../cache/${RUN}_stage1.pt --split train \
  --lr 6e-5 --optimizer adamw --weight_decay 0.05 --epochs 15 --batch_tokens 4000 --max_batch 4 --seed 0 --save_freq 5 \
  --out ../cache/${RUN}.pt --run_name ${RUN}
echo "===== EVAL on SKEMPI test ====="
srun python -u eval.py --lora_ckpt $LORA --checkpoint ../cache/${RUN}.pt --split test \
  --batch_tokens 4000 --max_batch 4 --out ../ibex-records/ESM3ΔG-Megascale-SKEMPIv2/results/eval_${RUN}.csv
echo "===== DONE ====="
