# ESM Backbone Test — Environment Record (py3.12, faithful to reference)

> **2026-07-01 REBUILD.** The previous env was a **py3.11 hack** (forced esm 3.3.0 onto py3.11
> via `--ignore-requires-python --no-deps`, with `transformers 5.12.1` + esm from a personal
> fork `jingweio/esm`). That did **NOT** match the reference implementation we follow
> (Cho et al., *absolute-stability-predictor* — ESM3ΔG) and was a likely source of error.
> The env was rebuilt from scratch to **faithfully match the reference**.

## Conda environment
- Name: `esm-backbone`
- Python: **3.12** (reference `README`: `conda create -n stability python=3.12`)
- esm: **official `evolutionaryscale/esm` main** (commit `cf002f1d2fb9220a0918fc8c5ea4e3b56d28fbaf`,
  pyproject version 3.3.0), **editable from the shared store** `/home/guoj0f/share/esm`
  (NOT the deleted `repos/esm`).
- transformers: **`4.57.6`** (standard PyPI, `>=4.40,<5.0` per reference pyproject — NOT the 5.x we had).

## Shared store (large common files; see ibex-usage §1d)
- esm code: `/home/guoj0f/share/esm`  ↔  ibex `/ibex/user/guoj0f/share/esm`
- HF weight cache (`HF_HOME`): `/home/guoj0f/share/hf_cache`  ↔  ibex `/ibex/user/guoj0f/share/hf_cache`
- ESM3 weights: esm 3.3.0 main loads **`biohub/esm3-sm-open-v1`** (the mirror it snapshots) into the cache.
- Code sets `HF_HOME` via `os.environ.setdefault(...)`; **sbatch overrides** `HF_HOME=/ibex/user/guoj0f/share/hf_cache` on Ibex.

## Install recipe (exact, in order) — local AND ibex
```bash
conda create -n esm-backbone python=3.12 -y && conda activate esm-backbone
# torch (cu124, per reference README)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# esm official main, editable, --no-deps (do NOT pull esm's declared Biohub transformers fork;
#   use standard transformers<5 like the reference instead)
pip install -e <share>/esm --no-deps           # <share> = /home/guoj0f/share (local) | /ibex/user/guoj0f/share (ibex)
# full dep set: reference deps (transformers<5, peft, gemmi) + biotite>=1.0 (esm needs CIFCategory)
#   + every esm runtime dep + StaB extras (scipy/pytest)
pip install "transformers>=4.40,<5.0" peft gemmi "biotite>=1.0" "numpy<2" scipy pytest \
  einops msgpack-numpy biopython scikit-learn brotli attrs pandas cloudpathlib httpx tenacity zstd \
  ipywidgets py3dmol pydssp boto3 pygtrie dna_features_viewer accelerate ipython rdkit
```
Full pinned set: see `scratchpad/esm-backbone-py312-freeze.txt` (107 pkgs) captured from the working local env.

## Key installed versions
| Package | Version |
|---|---|
| python | 3.12.13 |
| torch | 2.6.0+cu124 |
| transformers | 4.57.6 |
| esm | 3.3.0 (editable, commit cf002f1d) |
| peft | 0.19.1 |
| biotite | 1.7.1 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| accelerate | 1.14.0 |
| huggingface_hub | 0.36.2 |

## Gotchas resolved during the rebuild
- **biotite**: esm 3.3.0 main uses the new `CIFCategory` API → needs **`biotite>=1.0`**. The reference
  pyproject pins `biotite<0.40` which is INCOMPATIBLE with the esm it installs → that pyproject is not a
  reliable env spec; follow esm's own requirement.
- **esm `--no-deps`**: esm main declares `transformers @ git+Biohub/transformers` (a fork). We install
  esm `--no-deps` and use **standard transformers<5** (the reference's declared transformers), then add
  every other esm runtime dep manually (the long pip line above).
- **`ProteinComplex.num_chains`** (esm 3.3.0 has it; 3.2.1 didn't) → our `track_esm3/esm3_struct.py`
  uses `len(list(complex_.chain_iter()))` (works on both).
- **flash-attn**: not installed locally (no CUDA toolchain); ESM3 forwards run without it (fallback attention).
  Install on Ibex GPU nodes only if speed requires.

## Verification
`cd esm-backbone-test && HF_HOME=/home/guoj0f/share/hf_cache HF_HUB_OFFLINE=1 python -m pytest tests/test_esm3_stab_scorer.py tests/test_esm3_stab_finetune.py -q`
→ **8 passed (offline)** on the rebuilt py3.12 env.

## Import convention (unchanged)
`conftest.py` adds the worktree root (for `stabddg`) + `esm-backbone-test/` to `sys.path`, and
`setdefault`s `HF_HOME=/home/guoj0f/share/hf_cache` + `HF_HUB_OFFLINE=1`. Run tests from inside
`esm-backbone-test/`.
