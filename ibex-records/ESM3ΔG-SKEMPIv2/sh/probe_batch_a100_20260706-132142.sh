#!/bin/bash
#SBATCH --job-name=probe_batch_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=1:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/probe_batch_a100_20260706-132142.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/probe_batch_a100_20260706-132142.err
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export HF_HOME=/ibex/user/guoj0f/share/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CK=data/esm3dg_weights/ESM3dG_weights_1_lora.ckpt
# Probe the 3 largest complexes of BOTH the fine-tuning (train) set and the eval (test) set,
# at batch 4/6/8/10/12/16 with a REAL training step (binding_ddG + backward). train cache built
# here is reused by the actual fine-tuning later.
echo "############ TRAIN (fine-tuning set) ############"
python esm3dg_stab/probe_batch.py --ckpt $CK --split train --order largest --n_probe 3 --batches 4,6,8,10,12,16
echo "############ TEST (eval set) ############"
python esm3dg_stab/probe_batch.py --ckpt $CK --split test  --order largest --n_probe 3 --batches 4,6,8,10,12,16
echo "ALL DONE"
