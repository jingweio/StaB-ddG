#!/bin/bash
#SBATCH --job-name=twostage_esm3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/twostage_esm3_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/twostage_esm3_%j.err

# Full two-stage StaB-ddG pipeline for the ESM3 (structure-conditioned) backbone
# on a single a100:
#   stage1 = Megascale folding-stability finetune (scorer.folding_ddG)
#   stage2 = SKEMPI binding finetune, initialised from the stage-1 backbone
#   eval   = SKEMPI test-split eval of the stage-2 backbone -> baseline-format CSV
# Per-stage per-epoch checkpoints are saved (resumable). Outputs -> results/.

set -euo pipefail

# ----------------------------- tunable params ----------------------------- #
LR=1e-5                 # full-FT of pretrained ESM3 (1e-6 is for tiny ProteinMPNN)
S1=30                   # stage-1 (Megascale) epochs
S2=40                   # stage-2 (SKEMPI) epochs
# ESM3 is the 1.4B structure-conditioned model; training retains six scorer
# graphs per binding step, so the SKEMPI (stage-2) token budget must stay modest.
# Stage-1 folding does ONE forward per step, so it tolerates a larger budget.
BS_STAGE1=3000          # Megascale folding-stability token budget
BS_STAGE2=1500          # SKEMPI: conservative to avoid OOM-skipping long complexes on a100
RUN1=esm3_stage1
RUN2=esm3_stage2
CKPT_DIR=track_esm3/ckpts
# --------------------------------------------------------------------------- #

# Use the env's bin directly (robust; `conda activate` is unreliable in non-login shells).
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
echo "python -> $(which python)"

export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test
mkdir -p "${CKPT_DIR}" results

# ----------------------- Stage 1: Megascale folding ----------------------- #
python -m run.finetune \
  --backbone esm3 --stage stability \
  --lr "${LR}" --epochs "${S1}" --batch_size "${BS_STAGE1}" \
  --out "${CKPT_DIR}" --run_name "${RUN1}"

# ------------------- Stage 2: SKEMPI binding (chained) -------------------- #
python -m run.finetune \
  --backbone esm3 --stage skempi \
  --checkpoint "${CKPT_DIR}/${RUN1}_${S1}.pt" \
  --lr "${LR}" --epochs "${S2}" --batch_size "${BS_STAGE2}" \
  --out "${CKPT_DIR}" --run_name "${RUN2}"

# ------------------------- Eval: SKEMPI test split ------------------------ #
python -m run.eval \
  --backbone esm3 \
  --checkpoint "${CKPT_DIR}/${RUN2}_${S2}.pt" \
  --out results/eval_esm3_twostage.csv \
  --ensemble 1
