#!/bin/bash
#SBATCH --job-name=ft_timing_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=2:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/ft_timing_smoke_a100_20260706-132142.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/ft_timing_smoke_a100_20260706-132142.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export HF_HOME=/ibex/user/guoj0f/share/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 2-epoch full-train run: measures per-epoch wall time (basis for max-epochs) + validates the
# grad-accum / OOM-drop path on a100 over the REAL train set (expect ~3 giant complexes skipped).
CK=data/esm3dg_weights/ESM3dG_weights_1_lora.ckpt
# finetune.py now prints per-epoch wall time -> read "time=Ns" from the 2 epoch lines
python esm3dg_stab/finetune.py \
  --stage skempi --lora_ckpt $CK --split train \
  --epochs 2 --batch_tokens 10000 --max_batch 4 --lr 1e-5 \
  --out cache/ft_timing_smoke.pt --run_name ft_timing_smoke
echo "ALL DONE"
