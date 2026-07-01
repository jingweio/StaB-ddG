#!/bin/bash
# REPAIR: install the faithful stack INTO the existing (empty) esm-backbone py3.12 env on Ibex.
# NO conda env remove/create — safest for the concurrently-running Stage-1 (its files are detached).
# Offline from the rsynced wheelhouse (Ibex external downloads are flaky).
set -eo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
conda activate esm-backbone
WH=/ibex/user/guoj0f/StaB-ddG/esm-replace/env-build/wheelhouse-py312
echo "python (into which we install): $(python --version)"

echo "=== torch (offline) ==="
pip install -q --no-index --find-links "$WH" torch==2.6.0
echo "=== full deps (offline) ==="
pip install -q --no-index --find-links "$WH" "transformers>=4.40,<5.0" peft gemmi "biotite>=1.0" "numpy<2" scipy pytest \
  einops msgpack-numpy biopython scikit-learn brotli attrs pandas cloudpathlib httpx tenacity zstd \
  ipywidgets py3dmol pydssp boto3 pygtrie dna_features_viewer accelerate ipython rdkit
echo "=== esm editable from share/esm (offline, --no-build-isolation) ==="
pip install -q --no-index -e /ibex/user/guoj0f/share/esm --no-deps --no-build-isolation

echo "=== verify ==="
python -c "
import sys, torch, transformers, esm, peft, biotite, numpy
print('esm-backbone RESTORED — py', sys.version.split()[0], '| torch', torch.__version__, '| tfm', transformers.__version__, '| esm', esm.__version__, '| biotite', biotite.__version__)
from esm.pretrained import ESM3_sm_open_v0
from peft import inject_adapter_in_model
print('scorer-side imports OK')
"
echo "REPAIR_DONE_OK"
