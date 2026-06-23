"""
test_esm3_scorer.py — tests for the structure-conditioned ESM3Scorer.

Run from inside esm-backbone-test/:
    cd esm-backbone-test && python -m pytest tests/test_esm3_scorer.py -v

Uses the REAL fixture ``tests/fixtures/one_complex.pkl`` (a dict with nested
``complex`` / ``binder1`` / ``binder2`` domains whose ``name``s resolve under
``data/SKEMPI2_PDBs``: complex ``1A4Y``, binders ``1A4Y_A`` / ``1A4Y_B``).

Device is ``cpu`` throughout.  Loading ESM3 1.4B on CPU plus a few forwards on a
~583-residue complex takes a few minutes — acceptable.  The critical correctness
gate is ``test_multichain_alignment_recovery``: it catches any sequence<->
structure chain-break length/ordering misalignment.
"""

import os
import pickle

import torch

from track_esm3.esm3_scorer import ESM3Scorer

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
PDB_DIR = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI2_PDBs")
FIX = os.path.join(_THIS_DIR, "fixtures", "one_complex.pkl")


def _load():
    with open(FIX, "rb") as f:
        return pickle.load(f)


def test_wt_ddG_zero_singlechain():
    sc = ESM3Scorer(device="cpu", pdb_dir=PDB_DIR)
    dom = _load()["binder1"]
    assert torch.allclose(
        sc.folding_ddG(dom, sc.get_wt_seq(dom)), torch.zeros(1), atol=1e-3
    )


def test_multichain_alignment_recovery():
    """THE gate: StaB-domain WT sequence scored against the tokenizer's structure
    on a MULTI-CHAIN complex must recover the native sequence well (>0.30).
    Catches any sequence<->structure chain-order/length misalignment."""
    sc = ESM3Scorer(device="cpu", pdb_dir=PDB_DIR)
    dom = _load()["complex"]  # 2-chain (A + B)
    wt = sc.get_wt_seq(dom)
    st, coords, plddt = sc.struct.tokens_for(dom)

    from esm.utils.encoding import tokenize_sequence
    from common.seq_codec import (
        chain_lengths,
        insert_chainbreaks,
        int_seq_to_str,
    )

    seq_str = insert_chainbreaks(int_seq_to_str(wt[0]), chain_lengths(dom))
    seqtoks = tokenize_sequence(
        seq_str, sc.seq_tok, add_special_tokens=True
    ).unsqueeze(0)
    assert seqtoks.shape[1] == st.shape[1], (seqtoks.shape, st.shape)  # alignment

    with torch.no_grad():
        out = sc.model.forward(
            sequence_tokens=seqtoks,
            structure_tokens=st,
            structure_coords=coords,
            per_res_plddt=plddt,
        )
    logp = torch.log_softmax(out.sequence_logits.float(), dim=-1)[0]

    # recovery over real residue positions (skip BOS and any '|'/EOS)
    tgt = seqtoks[0]
    valid = ~(
        (tgt == sc.seq_tok.cls_token_id)
        | (tgt == sc.seq_tok.eos_token_id)
        | (tgt == sc.cb_id)
    )
    pred = logp.argmax(-1)
    rec = (pred[valid] == tgt[valid]).float().mean().item()
    assert rec > 0.30, f"recovery too low ({rec:.2%}) -> seq/structure misaligned"


def test_mutation_changes_dG():
    sc = ESM3Scorer(device="cpu", pdb_dir=PDB_DIR)
    dom = _load()["binder1"]
    wt = sc.get_wt_seq(dom)
    mut = wt.clone()
    mut[0, 5] = (mut[0, 5] + 1) % 20
    d = sc.folding_ddG(dom, mut)
    assert d.shape == (1,) and torch.isfinite(d).all() and d.abs().item() > 1e-4
