#!/bin/bash
#SBATCH --job-name=esm3_abl_fp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/abl_fp_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/abl_fp_%j.err
set -uo pipefail
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test
# SKEMPI-only at the BEST config, NO Stage-1 checkpoint (from pretrained ESM3) -> does Megascale Stage-1 help?
python -m run.finetune --backbone esm3 --stage skempi --lr 6e-5 --optimizer adamw --weight_decay 0.05 \
  --epochs 15 --batch_size 1500 --out track_esm3/ckpts --run_name esm3_abl_fp
python -m run.eval --backbone esm3 --checkpoint track_esm3/ckpts/esm3_abl_fp_15.pt --split test \
  --out results/eval_esm3_abl_fp_test.csv --ensemble 1
echo "ABL_FP_DONE"
