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
# Probe complexes SPREAD across the L(complex) spectrum (min..max) of the fine-tuning (train) set,
# from B=1 up, with a REAL training step (binding_ddG + backward) => the full max-batch-vs-L OOM curve.
# Also probe the largest-3 explicitly (to nail the low end where even B=4 OOMs).
echo "############ TRAIN spread (8 complexes across L) ############"
python esm3dg_stab/probe_batch.py --ckpt $CK --split train --order spread   --n_probe 8 --batches 1,2,3,4,6,8,12,16,24
echo "############ TRAIN largest-3 (from B=1) ############"
python esm3dg_stab/probe_batch.py --ckpt $CK --split train --order largest  --n_probe 3 --batches 1,2,3,4
echo "############ TEST spread (6 complexes across L) ############"
python esm3dg_stab/probe_batch.py --ckpt $CK --split test  --order spread   --n_probe 6 --batches 1,2,3,4,6,8,12,16,24
echo "ALL DONE"
