# esm3dg env (this task's dedicated conda env)

Per-task env for the ESM3ΔG×SKEMPI experiment (NEVER reuse another task's env, e.g. esm-backbone).

- **Python 3.12**, torch 2.6.0+cu124, esm 3.3.0 (editable from the shared store `share/esm`), + deps.
- Exact versions pinned in `requirements-lock.txt` (106 pkgs; excludes editable esm).
- ESM3 weights come from the shared HF cache: `HF_HOME=/home/guoj0f/share/hf_cache` (local) / `/ibex/user/guoj0f/share/hf_cache` (Ibex).

## Build (local)
```
conda create -n esm3dg python=3.12 -y && conda activate esm3dg
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-lock.txt      # or online: the dep set
pip install -e /home/guoj0f/share/esm --no-deps
```

## Build (Ibex, OFFLINE from a wheelhouse — pytorch.org downloads are flaky on Ibex)
1. Build wheelhouse locally: `pip download -r requirements-lock.txt -d <worktree>/env-build/ --extra-index-url https://download.pytorch.org/whl/cu124 --prefer-binary` (gitignored, ~3.1G).
2. rsync `env-build/` → Ibex per-branch `env-build/`.
3. `sbatch build_ibex_esm3dg.sbatch` (conda create py3.12 + `pip install --no-index --find-links env-build -r requirements-lock.txt` + `pip install -e /ibex/user/guoj0f/share/esm --no-deps`).
