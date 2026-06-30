"""test_esm3_stab_scorer.py — ESM3StabScorer (LoRA + stability head) on CPU.

Run from esm-backbone-test/:  python -m pytest tests/test_esm3_stab_scorer.py -v
Uses tests/fixtures/one_complex.pkl (complex 1A4Y; binders 1A4Y_A/_B) under
data/SKEMPI2_PDBs. ESM3 loads on CPU once (module-scoped fixture; ~minutes).
"""
import os
import pickle

import pytest
import torch

from track_esm3stab.esm3_stab_scorer import ESM3StabScorer, ESM3StabilityHead

_THIS = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
PDB_DIR = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI2_PDBs")
FIX = os.path.join(_THIS, "fixtures", "one_complex.pkl")


def _load():
    with open(FIX, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def scorer():
    return ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)


def test_folding_dG_shape_finite(scorer):
    dom = _load()["binder1"]
    dG = scorer.folding_dG(dom, scorer.get_wt_seq(dom))
    assert dG.shape == (1,) and torch.isfinite(dG).all()


def test_wt_ddG_zero(scorer):
    dom = _load()["binder1"]
    assert torch.allclose(
        scorer.folding_ddG(dom, scorer.get_wt_seq(dom)), torch.zeros(1), atol=1e-4
    )


def test_mutation_changes_dG(scorer):
    dom = _load()["binder1"]
    wt = scorer.get_wt_seq(dom)
    mut = wt.clone()
    mut[0, 5] = (mut[0, 5] + 1) % 20
    d = scorer.folding_ddG(dom, mut)
    assert d.shape == (1,) and torch.isfinite(d).all() and d.abs().item() > 1e-6


def test_batch_consistency(scorer):
    dom = _load()["binder1"]
    s1 = scorer.get_wt_seq(dom)
    s2 = s1.clone()
    s2[0, 3] = (s2[0, 3] + 1) % 20
    batch = torch.cat([s1, s2], 0)
    dG_b = scorer.folding_dG(dom, batch)
    dG_e = torch.cat([scorer.folding_dG(dom, s1), scorer.folding_dG(dom, s2)])
    assert torch.allclose(dG_b, dG_e, atol=1e-3), (dG_b, dG_e)


def test_factory_returns_scorer():
    from common.build_scorer import build_scorer
    sc = build_scorer("esm3_stab", device="cpu", pdb_dir=PDB_DIR)
    assert isinstance(sc, ESM3StabScorer)
