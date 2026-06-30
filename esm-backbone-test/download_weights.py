#!/usr/bin/env python
"""Download ESM model weights for the StaB-ddG ESM-backbone experiment.

Weights are cached inside the `esm` repo (reproduce branch) at
$HF_HOME=/home/guoj0f/share/hf_cache (gitignored), then symlinked into this
experiment repo under esm-backbone-test/weights/<name> so the experiment's
load-paths live in THIS repo (per requirement 1).

All four biohub models are ungated (verified). hf_transfer is enabled for speed.
"""
import os

os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from pathlib import Path
from huggingface_hub import snapshot_download

WEIGHTS_DIR = Path("/home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace/esm-backbone-test/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# repo_id -> stable symlink name in our repo
MODELS = {
    "biohub/esmc-6b-2024-12": "esmc_6b",        # Track B (ESMC-6B, multi-GPU full-FT)
    "biohub/esm3-sm-open-v1": "esm3_sm_open",   # Track A (ESM3-1.4B, single-GPU full-FT)
    "biohub/esmc-600m-2024-12": "esmc_600m",    # fallback / fast-iteration
}


def main():
    for repo_id, name in MODELS.items():
        print(f"\n===== downloading {repo_id} -> {name} =====", flush=True)
        path = snapshot_download(repo_id=repo_id)
        link = WEIGHTS_DIR / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path)
        print(f"  cached at: {path}")
        print(f"  symlinked: {link} -> {path}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
