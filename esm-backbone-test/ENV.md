# ESM Backbone Test — Environment Record

## Conda environment

Name: `esm-backbone`
Python: 3.11.15 (pre-existing; packages installed with `--ignore-requires-python` where needed)

> **Note on Python version:** ESM 3.3.0's `pyproject.toml` declares `requires-python = ">=3.12,<3.13"`.
> Several transitive deps (numpy, scipy latest) also require 3.12.
> Workaround: pin `numpy<2.0` (→ 1.26.4) and `scipy<1.14` (→ 1.13.1) to get pre-built wheels
> for 3.11; install `esm` itself with `--no-deps --ignore-requires-python`; then install each
> dep manually.  All tests pass on Python 3.11 with the pinned deps below.

---

## Install commands (exact, in order)

```bash
# Activate env
source ~/anaconda3/etc/profile.d/conda.sh && conda activate esm-backbone

# 1. Pin numpy/scipy to 3.11-compatible wheels
pip install "numpy<2.0"      # → 1.26.4
pip install "scipy<1.14"     # → 1.13.1

# 2. Install ESM source (editable, skip Python version check, no deps yet)
pip install -e /home/guoj0f/repos/esm --ignore-requires-python --no-deps

# 3. Install torch + core ML deps
pip install "torch>=2.2.0,<2.6" einops biotite biopython pandas pytest

# 4. Install remaining ESM runtime deps
pip install attrs accelerate msgpack-numpy zstd pygtrie tenacity scikit-learn

# 5. Install networking/format deps
pip install cloudpathlib brotli pydssp

# 6. Install tokenization deps (standard transformers, not the custom Biohub fork)
pip install tokenizers transformers

# 7. flash-attn (OPTIONAL — see below)
pip install flash-attn --no-build-isolation   # SKIPPED — see flash-attn section
```

---

## Installed package versions

| Package           | Version  | Notes                                      |
|-------------------|----------|--------------------------------------------|
| esm               | 3.3.0    | editable install from /home/guoj0f/repos/esm |
| torch             | 2.5.1    |                                            |
| numpy             | 1.26.4   | pinned <2.0 for Python 3.11 compat         |
| scipy             | 1.13.1   | pinned <1.14 for Python 3.11 compat        |
| pandas            | 3.0.3    |                                            |
| biopython         | 1.87     |                                            |
| biotite           | 1.6.0    |                                            |
| huggingface_hub   | 1.20.1   |                                            |
| transformers      | 5.12.1   | standard PyPI version (not Biohub fork)    |
| tokenizers        | 0.22.2   |                                            |
| accelerate        | 1.14.0   |                                            |
| attrs             | 26.1.0   |                                            |
| einops            | 0.8.2    |                                            |
| cloudpathlib      | 0.24.0   |                                            |
| tenacity          | 9.1.4    |                                            |
| scikit-learn      | 1.9.0    |                                            |
| msgpack-numpy     | 0.4.8    |                                            |
| pytest            | 9.1.1    |                                            |

---

## flash-attn status: SKIPPED

**Command attempted:** `pip install flash-attn --no-build-isolation`

**Failure reason:** `CUDA_HOME environment variable is not set. Please set it to your CUDA install root.`

flash-attn requires a matching CUDA toolchain at build time.  This workstation
has no CUDA compiler exposed (no `nvcc` / no `CUDA_HOME`).  The package is
**NOT required** for local tests — all ESM calls pass `use_flash_attn=False`.

flash-attn will be installed on the Ibex GPU nodes at the Ibex-mirror task stage.

---

## HF weight cache layout

```
/home/guoj0f/repos/esm/.hf_cache/hub/
  models--biohub--esmc-600m-2024-12/
    snapshots/e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c/
      config.json
      README.md
      data/weights/esmc_600m_2024_12_v0.pth   ← actual weights
  models--biohub--esmc-6b-2024-12/
  models--biohub--esm3-sm-open-v1/
```

**Important:** `huggingface_hub >= 0.29.0`'s `load_torch_model()` looks for
`.safetensors` / `.bin` files at the snapshot root and does not find the nested
`.pth` file.  `conftest.py` patches `huggingface_hub.load_torch_model` (and its
module-level alias) to fall back to `rglob("*.pth")` when no standard checkpoint
files are present.  The patch also passes `assign=True` to `load_state_dict` so
that meta-tensor parameters (created by `accelerate.init_empty_weights`) are
properly populated.

---

## Import convention

All tests live in `esm-backbone-test/` (hyphens prevent this being a Python
package).  `conftest.py` manipulates `sys.path` so that:

- `import stabddg` → resolves against the git worktree root
  (`/home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace`)
- `from common.scorer import ...` → top-level package in `esm-backbone-test/common/`
- `from track_esmc.esmc_scorer import ...` → `esm-backbone-test/track_esmc/`
- `from track_esm3.esm3_scorer import ...` → `esm-backbone-test/track_esm3/`
- `from run.evaluate import ...` → `esm-backbone-test/run/`

**Always run tests from inside `esm-backbone-test/`:**

```bash
cd esm-backbone-test && python -m pytest tests/ -v
```

---

## Smoke test result

```
tests/test_env.py::test_esmc_600m_loads_offline PASSED   [4.67s]
```

Model: ESMC-600M, 575,036,992 parameters (~575M), loaded from offline cache on CPU.
```
