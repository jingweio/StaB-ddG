import torch
from track_esmc.esmc_scorer import ESMCScorer


def _toy(seq="MKTLLILAVLA"):
    return {"seq": seq, "seq_chain_A": seq}


def test_esmc_loader_guards_against_silent_random_weights(monkeypatch):
    """The real bug: the checkpoint keys carry an `esmc.` prefix and name the head
    `lm_head`, so a naive load (old from_pretrained path) silently mismatched all 808
    keys -> RANDOM weights, while shape-only tests still passed. `load_esmc` now RAISES
    if any weight is missing after the remap. This test proves that guard fires: with
    the remap disabled, the loader must raise rather than return a random-weight model.
    (Device/flash-independent; builds 600m real-init on CPU.)"""
    import pytest
    import common.esmc_loader as L
    # Force every key to mismatch -> load_esmc must RAISE rather than return a random model.
    # (600m keys are plain; 6B keys carry esmc./lm_head and genuinely need the real remap.)
    monkeypatch.setattr(L, "_remap", lambda k: "BOGUS_" + k)
    with pytest.raises(RuntimeError, match="missing"):
        L.load_esmc("esmc_600m", device="cpu", use_flash_attn=False)


def test_folding_ddG_of_wt_is_zero():
    sc = ESMCScorer("esmc_600m", device="cpu", use_flash_attn=False)
    dom = _toy()
    wt = sc.get_wt_seq(dom)
    assert torch.allclose(sc.folding_ddG(dom, wt), torch.zeros(1), atol=1e-4)


def test_dG_negative():
    sc = ESMCScorer("esmc_600m", device="cpu", use_flash_attn=False)
    dom = _toy("ACDEF")
    assert sc.folding_dG(dom, sc.get_wt_seq(dom))[0].item() < 0


def test_batch_consistency():
    """dG of a 2-row batch must equal scoring each row alone (catches pad-attention bugs)."""
    sc = ESMCScorer("esmc_600m", device="cpu", use_flash_attn=False)
    dom = _toy("ACDEFGHIKL")
    s1 = sc.get_wt_seq(dom)  # [1,L]
    s2 = s1.clone()
    s2[0, 0] = (s2[0, 0] + 1) % 20  # one mutation
    batch = torch.cat([s1, s2], dim=0)  # [2,L]
    dG_batch = sc.folding_dG(dom, batch)  # [2]
    dG_each = torch.cat([sc.folding_dG(dom, s1), sc.folding_dG(dom, s2)])
    assert torch.allclose(dG_batch, dG_each, atol=1e-3), (dG_batch, dG_each)


def test_multichain_chainbreak():
    """A 2-chain domain inserts '|' and still scores (chain-break excluded from sum)."""
    sc = ESMCScorer("esmc_600m", device="cpu", use_flash_attn=False)
    dom = {"seq": "ACDEFGHI", "seq_chain_A": "ACDE", "seq_chain_B": "FGHI"}
    wt = sc.get_wt_seq(dom)
    dG = sc.folding_dG(dom, wt)
    assert dG.shape == (1,) and torch.isfinite(dG).all()
