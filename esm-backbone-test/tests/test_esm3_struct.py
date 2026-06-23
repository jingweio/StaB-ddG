"""
test_esm3_struct.py — tests for ESM3StructTokenizer.

Run from inside esm-backbone-test/:
    cd esm-backbone-test && python -m pytest tests/test_esm3_struct.py -v

Both tests use a REAL structure: the SKEMPI binder ``1A4Y_B`` (123 residues,
single chain) loaded from ``data/SKEMPI2_PDBs/1A4Y_B.pdb`` via ``pdb_dir`` —
the domain dict is the ``binder2`` entry of the bundled ``one_complex.pkl``
fixture, whose ``name`` ('1A4Y_B') resolves to that PDB.

The acceptance test loads ESM3 1.4B on CPU (slow, ~1-3 min) and checks that a
structure-conditioned forward recovers the chain's own sequence well above a
loose 0.30 floor — proving the structure tokens carry real information.
"""

import os
import pickle

import pytest
import torch

from track_esm3.esm3_struct import (
    ESM3StructTokenizer,
    STRUCT_BOS,
    STRUCT_EOS,
    STRUCT_MASK,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_PDB_DIR = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI2_PDBs")
_FIXTURE = os.path.join(_THIS_DIR, "fixtures", "one_complex.pkl")


def _binder_domain():
    """Return the single-chain binder domain (name='1A4Y_B', 123 residues)."""
    with open(_FIXTURE, "rb") as f:
        d = pickle.load(f)
    dom = d["binder2"]
    # Sanity: its name must resolve to a real PDB under pdb_dir.
    assert os.path.exists(os.path.join(_PDB_DIR, f"{dom['name']}.pdb")), dom["name"]
    return dom


def test_shape_specials_and_cache():
    """Shapes, special tokens, interior id range, and per-domain caching."""
    dom = _binder_domain()
    L = len(dom["seq"])  # 123 real residues

    tok = ESM3StructTokenizer(device="cpu", pdb_dir=_PDB_DIR)
    st, coords, plddt = tok.tokens_for(dom)

    # Shapes: T == L + 2 (BOS + EOS), batch dim 1.
    assert st.shape == (1, L + 2)
    assert coords.shape == (1, L + 2, 37, 3)
    assert plddt.shape == (1, L + 2)

    # Integer token dtype.
    assert not torch.is_floating_point(st)

    # Special tokens at the ends (BOS > EOS — intentional).
    assert st[0, 0].item() == STRUCT_BOS == 4098
    assert st[0, -1].item() == STRUCT_EOS == 4097

    # Interior tokens are real structure ids in [0, 4096) — never specials.
    interior = st[0, 1:-1]
    assert interior.min().item() >= 0
    assert interior.max().item() < STRUCT_MASK == 4096

    # Caching: a second call with the same domain returns the SAME objects.
    st2, coords2, plddt2 = tok.tokens_for(dom)
    assert st2 is st
    assert coords2 is coords
    assert plddt2 is plddt


def test_structure_tokens_are_meaningful():
    """Acceptance: structure-conditioned ESM3 recovers the sequence > 0.30.

    Loads ESM3 1.4B on CPU and runs a forward pass conditioned on the
    tokenizer's structure tokens plus the chain's own sequence tokens; the
    argmax sequence-recovery on the interior positions must clear a loose 0.30
    floor (mirrors the standalone 81% / 57% validation runs).
    """
    from esm.pretrained import ESM3_sm_open_v0
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    from esm.utils.encoding import tokenize_sequence

    dom = _binder_domain()
    seq = dom["seq_chain_B"]  # this chain's own sequence

    tok = ESM3StructTokenizer(device="cpu", pdb_dir=_PDB_DIR)
    structure_tokens, coords, plddt = tok.tokens_for(dom)

    seq_tokenizer = EsmSequenceTokenizer()
    sequence_tokens = tokenize_sequence(
        seq, seq_tokenizer, add_special_tokens=True
    ).unsqueeze(0)
    assert sequence_tokens.shape == structure_tokens.shape

    model = ESM3_sm_open_v0("cpu").to(torch.float32).eval()
    with torch.no_grad():
        out = model.forward(
            sequence_tokens=sequence_tokens,
            structure_tokens=structure_tokens,
        )

    pred = out.sequence_logits.argmax(-1)[0]  # (T,)
    true = sequence_tokens[0]
    interior = slice(1, -1)  # skip BOS / EOS
    recovery = (pred[interior] == true[interior]).float().mean().item()

    assert recovery > 0.30, f"sequence recovery too low: {recovery:.4f}"
