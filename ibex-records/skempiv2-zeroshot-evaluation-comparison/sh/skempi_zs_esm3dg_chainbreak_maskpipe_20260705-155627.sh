#!/bin/bash
#SBATCH --job-name=skempi_zs_cb_maskpipe
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/skempiv2-zeroshot-evaluation-comparison/skempi_zs_esm3dg_chainbreak_maskpipe_20260705-155627_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/skempiv2-zeroshot-evaluation-comparison/skempi_zs_esm3dg_chainbreak_maskpipe_20260705-155627_%j.err
# Exp3 variant a: ESM3dG base 3-ens ZERO-SHOT on SKEMPI test binding ddG, multi-chain CHAINBREAK + mask-out pipe, raw
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export HF_HOME=/ibex/user/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg_stab
W=../data/esm3dg_weights; R=../ibex-records/skempiv2-zeroshot-evaluation-comparison/results
srun python -u eval_skempi_ens.py --chainbreak --mask_pipe --members $W/ESM3dG_weights_1_lora.ckpt,$W/ESM3dG_weights_2_lora.ckpt,$W/ESM3dG_weights_3_lora.ckpt \
  --split test --batch_tokens 4000 --max_batch 4 --out $R/eval_skempi_esm3dg_base_ens_chainbreak_maskpipe.csv
srun python -u skempi_metrics.py $R/eval_skempi_esm3dg_base_ens_chainbreak_maskpipe.csv
echo "===== DONE ====="
