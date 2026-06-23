"""
conftest.py — pytest configuration for esm-backbone-test.

Import convention
-----------------
This experiment directory is named "esm-backbone-test/" (with hyphens), so it
cannot be used as a Python package name.  Instead, we manipulate sys.path here
so that:

  1. `import stabddg` resolves against the worktree root
     (/home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace).

  2. Sub-packages inside esm-backbone-test/ are importable as top-level names:
     `from common.scorer import ...`
     `from track_esmc.esmc_scorer import ...`
     `from track_esm3.esm3_scorer import ...`
     `from run.evaluate import ...`

Tests must be run from *inside* the esm-backbone-test/ directory:
    cd esm-backbone-test && pytest

Environment variables needed for offline weight loading are also set here so
they are in place before any test module is imported.

HF load_torch_model patch
--------------------------
huggingface_hub >= 0.29.0's load_torch_model only looks for .safetensors / .bin
at the snapshot root. The ESM 3.3.0 HF snapshots store weights under
data/weights/*.pth — a layout that pre-dates the safetensors convention.
We monkey-patch load_torch_model so that when given a directory that contains
no .safetensors/.bin at the root, it falls back to finding a single .pth file
anywhere under the tree. The patch is applied once at import time so it is in
effect for all tests in the session.
"""

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Offline HuggingFace cache — must be set before any esm import               #
# --------------------------------------------------------------------------- #
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# --------------------------------------------------------------------------- #
# sys.path additions                                                            #
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent          # .../esm-backbone-test/
_WORKTREE_ROOT = _THIS_DIR.parent                    # .../esm-replace/

for _p in [str(_WORKTREE_ROOT), str(_THIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------- #
# Monkey-patch huggingface_hub.load_torch_model to handle .pth snapshots      #
# --------------------------------------------------------------------------- #
import huggingface_hub as _hfhub
import huggingface_hub.serialization._torch as _hf_torch


def _patched_load_torch_model(model, checkpoint_path, **kwargs):
    """
    Wraps hf_hub's load_torch_model to fall back to recursive .pth discovery
    when the snapshot directory contains no .safetensors / .bin at its root.
    This is needed for ESM 3.3.0 snapshots which store weights as:
        <snapshot>/data/weights/<name>.pth
    """
    import torch
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        safe = kwargs.get("safe", True)
        # Check if standard files exist at root
        if safe:
            has_std = any(checkpoint_path.glob("*.safetensors")) or (
                checkpoint_path / "model.safetensors.index.json"
            ).exists()
        else:
            has_std = any(checkpoint_path.glob("*.bin")) or any(
                checkpoint_path.glob("*.safetensors")
            )
        if not has_std:
            # Fall back: find a single .pth file anywhere in the tree
            pth_files = list(checkpoint_path.rglob("*.pth"))
            if len(pth_files) == 1:
                pth_path = pth_files[0]
                state_dict = torch.load(
                    pth_path,
                    map_location=kwargs.get("map_location", None),
                    weights_only=kwargs.get("weights_only", False),
                )
                return model.load_state_dict(
                    state_dict, strict=kwargs.get("strict", False), assign=True
                )
            elif len(pth_files) > 1:
                raise ValueError(
                    f"Multiple .pth files found under {checkpoint_path}: {pth_files}. "
                    "Cannot determine which to load."
                )
    return _original_load_torch_model(model, checkpoint_path, **kwargs)


_original_load_torch_model = _hf_torch.load_torch_model
_hf_torch.load_torch_model = _patched_load_torch_model
_hfhub.load_torch_model = _patched_load_torch_model
