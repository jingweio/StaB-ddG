#!/bin/bash
#SBATCH --job-name=esm3_stage2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/esm3_stage2_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/esm3_stage2_%j.err
set -euo pipefail
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test
# Stage-2 (SKEMPI binding) resuming from the CONVERGED Stage-1 ckpt (matches ESMC S2=25)
python -m run.finetune --backbone esm3 --stage skempi \
  --checkpoint track_esm3/ckpts/esm3_stage1_14.pt \
  --lr 1e-5 --epochs 25 --batch_size 1500 \
  --out track_esm3/ckpts --run_name esm3_stage2
python -m run.eval --backbone esm3 --checkpoint track_esm3/ckpts/esm3_stage2_25.pt \
  --out results/eval_esm3_twostage.csv --ensemble 1
echo "ESM3_TWOSTAGE_DONE"
