#!/bin/bash
#SBATCH --job-name=esm3_evalep
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/evalep_%x_%j.out
#SBATCH --error=/ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/results/evalep_%x_%j.err
set -euo pipefail
export PATH=/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin:$PATH
export HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test
RUN=${RUN:-lr6e5}
EPOCHS=${EPOCHS:-"2 4 6 8 10 12 14"}
OUT=results/esm3_earlystop_${RUN}.csv
echo "run,epoch,test_per_iface_spearman" > $OUT
for ep in $EPOCHS; do
  CK=track_esm3/ckpts/esm3_${RUN}_${ep}.pt
  [ -f "$CK" ] || { echo "skip $CK (missing)"; continue; }
  python -m run.eval --backbone esm3 --checkpoint $CK --split test --out results/_tmp_ep.csv --ensemble 1
  python - "$RUN" "$ep" >> $OUT <<PY
import sys, pandas as pd, warnings; warnings.filterwarnings('ignore')
from baselines.eval_utils import compute_metrics
run, ep = sys.argv[1], sys.argv[2]
d = pd.read_csv('results/_tmp_ep.csv', index_col=0)
print(f"{run},{ep},{compute_metrics(d, bootstrap=False)['Per Structure Spearman']:.4f}")
PY
done
echo "=== EARLYSTOP DONE ==="; cat $OUT
