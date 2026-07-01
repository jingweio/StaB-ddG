#!/bin/bash
#SBATCH --job-name=build_esm3dg
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
# build this task's DEDICATED esm3dg env on Ibex, OFFLINE from the per-branch wheelhouse.
set -eo pipefail
source /ibex/user/guoj0f/anaconda3/etc/profile.d/conda.sh
WD=/ibex/user/guoj0f/StaB-ddG/MGnify-replace/env-build
conda env remove -n esm3dg -y >/dev/null 2>&1 || true
conda create -n esm3dg python=3.12 -y >/dev/null
conda activate esm3dg
pip install --no-index --find-links "$WD" -r "$WD/requirements-lock.txt"
pip install --no-index -e /ibex/user/guoj0f/share/esm --no-deps --no-build-isolation  # --no-build-isolation: use env's setuptools offline (no PEP517 backend fetch)
python -c "import torch,esm,peft,gemmi,transformers,zstd; print('torch',torch.__version__,'esm',esm.__version__)"
echo IBEX_ESM3DG_ENV_OK
