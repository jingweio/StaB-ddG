#!/bin/bash
#SBATCH --job-name=esm3dg_earlystop_eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/earlystop_eval_20260702-130921_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/ESM3ΔG-SKEMPIv2/earlystop_eval_20260702-130921_%j.err
# ZERO-COST DIAGNOSTIC: eval the ep5/ep10 intermediate adapters (save_freq=5) of BOTH
# task1 (esm3dg_skempi_aug1) and task2 (esm3dg_mega_skempi_aug1) on SKEMPI test, to check
# whether early-stopping beats the ep15-final (task1 per-struct 0.158 / task2 0.109).
# No retraining, no code change — just re-runs the already-validated eval.py with different --checkpoint.
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg   # this task's DEDICATED env
export HF_HOME=/ibex/user/guoj0f/share/hf_cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace/esm3dg_stab
LORA=../data/esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt
R1=../ibex-records/ESM3ΔG-SKEMPIv2/results
R2=../ibex-records/ESM3ΔG-Megascale-SKEMPIv2/results
for CKPT in esm3dg_skempi_aug1_ep5 esm3dg_skempi_aug1_ep10; do
  echo "===== EVAL task1 $CKPT ====="
  srun python -u eval.py --lora_ckpt $LORA --checkpoint ../cache/$CKPT.pt --split test \
    --batch_tokens 4000 --max_batch 4 --out $R1/eval_$CKPT.csv
done
for CKPT in esm3dg_mega_skempi_aug1_ep5 esm3dg_mega_skempi_aug1_ep10; do
  echo "===== EVAL task2 $CKPT ====="
  srun python -u eval.py --lora_ckpt $LORA --checkpoint ../cache/$CKPT.pt --split test \
    --batch_tokens 4000 --max_batch 4 --out $R2/eval_$CKPT.csv
done
echo "===== ALL EARLYSTOP EVALS DONE ====="
