#!/bin/bash
# Stage-1 (Megascale folding-stability) data for the two-stage finetune.
# Per StaB-ddG README: from Zenodo record 7992926 we need
#   - Tsuboyama2023_Dataset2_Dataset3_20230416.csv  (inside Processed_K50_dG_datasets.zip)
#   - AlphaFold_model_PDBs.zip                       (per-domain AF backbones)
# These are large (tens of GB) and gitignored. ESM3 (structure-conditioned) needs the
# AF backbones; ESMC (sequence-only) needs only the sequences (from the CSV).
set -euo pipefail
DATA_DIR="/home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace/data"
cd "$DATA_DIR"

dl() {  # url, outfile
  if [ -f "$2" ]; then echo "[skip] $2 exists"; else
    echo "[wget] $1"; wget -q --show-progress -O "$2" "$1"; fi
}

dl "https://zenodo.org/records/7992926/files/AlphaFold_model_PDBs.zip"      AlphaFold_model_PDBs.zip
dl "https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip" Processed_K50_dG_datasets.zip

echo "[unzip] AlphaFold_model_PDBs.zip";      [ -d AlphaFold_model_PDBs ]      || unzip -q AlphaFold_model_PDBs.zip
echo "[unzip] Processed_K50_dG_datasets.zip"; [ -d Processed_K50_dG_datasets ] || unzip -q Processed_K50_dG_datasets.zip

echo "=== verify ==="
ls -d AlphaFold_model_PDBs && echo "  AF PDBs: $(ls AlphaFold_model_PDBs | wc -l) files"
CSV="Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
[ -f "$CSV" ] && echo "  CSV present: $CSV ($(wc -l < "$CSV") lines)" || echo "  CSV MISSING"
echo "ALL DONE"
