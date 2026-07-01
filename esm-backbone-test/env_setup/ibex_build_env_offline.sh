#!/bin/bash
# Rebuild Ibex esm-backbone py3.12 env OFFLINE from a local wheelhouse (rsynced),
# to avoid the flaky Ibex login-node external downloads (torch from pytorch.org kept breaking).
set -eo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
WH=/ibex/user/guoj0f/StaB-ddG/esm-replace/env-build/wheelhouse-py312

echo "=== [1/5] remove old + create py3.12 ==="
conda env remove -n esm-backbone -y >/dev/null 2>&1 || true
conda create -n esm-backbone python=3.12 -y >/dev/null
conda activate esm-backbone
echo "python: $(python --version)"

echo "=== [2/5] torch (offline from wheelhouse) ==="
pip install -q --no-index --find-links "$WH" torch==2.6.0

echo "=== [3/5] full dep set (offline from wheelhouse) ==="
pip install -q --no-index --find-links "$WH" "transformers>=4.40,<5.0" peft gemmi "biotite>=1.0" "numpy<2" scipy pytest \
  einops msgpack-numpy biopython scikit-learn brotli attrs pandas cloudpathlib httpx tenacity zstd \
  ipywidgets py3dmol pydssp boto3 pygtrie dna_features_viewer accelerate ipython rdkit

echo "=== [4/5] esm official main (editable from Ibex share/esm), --no-deps ==="
pip install -q --no-index -e /ibex/user/guoj0f/share/esm --no-deps

echo "=== [5/5] verify ==="
python -c "
import sys, torch, transformers, esm, peft, biotite, numpy
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('esm         ', esm.__version__)
print('peft        ', peft.__version__)
print('biotite     ', biotite.__version__)
print('numpy       ', numpy.__version__)
"
echo "IBEX_ENV_BUILD_DONE_OK"
