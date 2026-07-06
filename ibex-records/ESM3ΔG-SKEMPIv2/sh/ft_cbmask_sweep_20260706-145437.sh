#!/bin/bash
#SBATCH --job-name=esm3dg_ft_cbmask
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=30:00:00
#SBATCH --array=0-4
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/ft_cbmask_lr%a_20260706-145437.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/ft_cbmask_lr%a_20260706-145437.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export HF_HOME=/ibex/user/guoj0f/share/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace
nvidia-smi --query-gpu=name --format=csv,noheader

# exp4 sweep: 5 lr x construction {chainbreak + mask_pipe}, single member weights_1.
# max-epochs=50, save every 10ep (ep10/20/30/40 + final ep50). No val (StaB has none).
LRS=(1e-6 5e-6 1e-5 5e-5 1e-4)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
echo "=== array task $SLURM_ARRAY_TASK_ID  lr=$LR ==="
CK=data/esm3dg_weights/ESM3dG_weights_1_lora.ckpt
python esm3dg_stab/finetune.py \
  --stage skempi --lora_ckpt $CK --split train \
  --chainbreak --mask_pipe \
  --epochs 50 --save_freq 10 --batch_tokens 10000 --max_batch 4 \
  --lr $LR --seed 0 \
  --out cache/esm3dg_skempi_cbmask_lr${LR}.pt \
  --run_name ft_cbmask_lr${LR}
echo "DONE lr=$LR"
