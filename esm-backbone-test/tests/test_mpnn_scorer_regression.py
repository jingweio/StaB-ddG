"""
Regression test: the refactored backbone-agnostic StaBddG + MPNNScorer must
reproduce the ORIGINAL stabddg.model.StaBddG predictions bit-for-bit.

This is the arbiter that the SequenceScorer / MPNNScorer extraction is purely
behaviour-preserving for the ProteinMPNN backbone.  Run with:

    cd esm-backbone-test && python -m pytest tests/test_mpnn_scorer_regression.py -v
"""

import os
import pickle

import pytest
import torch

from stabddg.mpnn_utils import ProteinMPNN
from stabddg.model import StaBddG as OrigStaBddG

from common.scorer import MPNNScorer
from common.stab_model import StaBddG as NewStaBddG


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FIXTURE_PATH = os.path.join(_THIS_DIR, "fixtures", "one_complex.pkl")

# Checkpoint lives at the worktree root: .../esm-replace/model_ckpts/stabddg.pt
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_CHECKPOINT = os.path.join(_WORKTREE_ROOT, "model_ckpts", "stabddg.pt")

# ProteinMPNN is tiny; CPU is fully deterministic and avoids GPU contention.
_DEVICE = torch.device("cpu")


def _load_pmpnn():
    """Construct ProteinMPNN + load the StaB-ddG checkpoint.

    Constructor block copied verbatim from run_stabddg.py:130-144, including the
    checkpoint-loading idiom that handles both a {'model_state_dict': ...} dict
    and a raw state_dict.
    """
    pmpnn = ProteinMPNN(node_features=128,
                        edge_features=128,
                        hidden_dim=128,
                        num_encoder_layers=3,
                        num_decoder_layers=3,
                        k_neighbors=48,
                        dropout=0.0,
                        augment_eps=0.0)

    mpnn_checkpoint = torch.load(_CHECKPOINT, map_location=_DEVICE)
    if 'model_state_dict' in mpnn_checkpoint.keys():
        pmpnn.load_state_dict(mpnn_checkpoint['model_state_dict'])
    else:
        pmpnn.load_state_dict(mpnn_checkpoint)
    return pmpnn


def test_mpnn_scorer_regression_equals_original():
    assert os.path.exists(_FIXTURE_PATH), (
        f"Fixture {_FIXTURE_PATH} missing — build it with "
        f"`python esm-backbone-test/tests/build_fixture.py` from the worktree root."
    )
    assert os.path.exists(_CHECKPOINT), f"Checkpoint {_CHECKPOINT} missing."

    with open(_FIXTURE_PATH, "rb") as f:
        sample = pickle.load(f)

    complex = sample["complex"]
    binder1 = sample["binder1"]
    binder2 = sample["binder2"]
    complex_mut_seqs = sample["complex_mut_seqs"].to(_DEVICE)
    binder1_mut_seqs = sample["binder1_mut_seqs"].to(_DEVICE)
    binder2_mut_seqs = sample["binder2_mut_seqs"].to(_DEVICE)

    # Shared backbone weights so the only difference is the wrapper code path.
    pmpnn = _load_pmpnn().to(_DEVICE)
    pmpnn.eval()

    orig_model = OrigStaBddG(pmpnn=pmpnn, use_antithetic_variates=False, device=_DEVICE)
    orig_model.to(_DEVICE)
    orig_model.eval()

    scorer = MPNNScorer(pmpnn, use_antithetic_variates=False, device=_DEVICE)
    new_model = NewStaBddG(scorer)
    new_model.to(_DEVICE)
    new_model.eval()

    # Seed identically immediately before each forward so both consume the same
    # RNG stream (with antithetic variates off, neither actually draws, but we
    # seed anyway to make the comparison airtight).
    with torch.no_grad():
        torch.manual_seed(0)
        orig_out = orig_model(complex, binder1, binder2,
                              complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs)

        torch.manual_seed(0)
        new_out = new_model(complex, binder1, binder2,
                            complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs)

    assert orig_out.shape == new_out.shape, (orig_out.shape, new_out.shape)
    max_abs_diff = (orig_out - new_out).abs().max().item()
    print(f"\nmax abs diff (orig vs new): {max_abs_diff:.3e}  "
          f"over {orig_out.numel()} predictions")
    assert torch.allclose(orig_out, new_out, atol=1e-5), (
        f"Refactored StaBddG diverges from original: max abs diff = {max_abs_diff:.3e}"
    )
