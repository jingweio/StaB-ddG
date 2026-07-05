#!/bin/bash
#SBATCH --job-name=skempi_zs_pmpnn
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/skempiv2-zeroshot-evaluation-comparison/skempi_zs_pmpnn_stage1_20260705-155627_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/ibex-records/skempiv2-zeroshot-evaluation-comparison/skempi_zs_pmpnn_stage1_20260705-155627_%j.err
# Exp3 model C: StaB stage-1 ProteinMPNN (stability_finetuned.pt) ZERO-SHOT on SKEMPI test binding ddG, 20x MC
set -euo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm3dg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
cd /ibex/user/guoj0f/StaB-ddG/MGnify-replace
R=ibex-records/skempiv2-zeroshot-evaluation-comparison/results
srun python -u skempi_eval.py --checkpoint model_ckpts/stability_finetuned.pt \
  --skempi_pdb_dir data/SKEMPI2_PDBs --run_name eval_skempi_pmpnn_stage1 \
  --ensemble 20 --seed 0 --noise_level 0.1 --output_dir $R
srun python -u esm3dg_stab/skempi_metrics.py $R/eval_skempi_pmpnn_stage1.csv
echo "===== DONE ====="
