"""
hf_compat.py — huggingface_hub compatibility shim for ESM 3.3.0 snapshots.

huggingface_hub.load_torch_model (>= 0.29.0 / observed v1.20.1) only looks for
.safetensors / .bin at the snapshot *root*. The ESM 3.3.0 HF snapshots store the
ESMC-600M weights under ``data/weights/*.pth`` — a layout that pre-dates the
safetensors convention. Without a fallback, loading ESMC-600M fails.

This module provides an idempotent ``apply()`` that monkey-patches
``huggingface_hub.load_torch_model`` so that, when handed a directory containing
no standard weight files at its root, it falls back to discovering a single
``.pth`` file anywhere under the tree and loading it directly.

ESMC-6B is native sharded safetensors and needs NO patch; the fallback simply
does not trigger for it.

The patch is guarded by a module-level ``_APPLIED`` flag so calling ``apply()``
multiple times (e.g. from both conftest and the scorer's import) is safe.
"""

from pathlib import Path

_APPLIED = False
_original_load_torch_model = None


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


def apply():
    """Install the .pth-fallback patch on huggingface_hub.load_torch_model.

    Idempotent: subsequent calls are no-ops once the patch is in place.
    """
    global _APPLIED, _original_load_torch_model
    if _APPLIED:
        return

    import huggingface_hub as _hfhub
    import huggingface_hub.serialization._torch as _hf_torch

    _original_load_torch_model = _hf_torch.load_torch_model
    _hf_torch.load_torch_model = _patched_load_torch_model
    _hfhub.load_torch_model = _patched_load_torch_model

    _APPLIED = True
