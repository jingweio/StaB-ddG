"""test_esm3_stab_finetune.py — param-efficient training path on CPU.

One optimizer step over the fixture domain must update ONLY LoRA+head, leaving
the frozen ESM3 base unchanged; adapter checkpoints round-trip. ESM3 loads on
CPU (a few minutes per scorer) — these are slow integration gates.
"""
import os
import pickle

import torch

from track_esm3stab.esm3_stab_scorer import ESM3StabScorer

_THIS = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
PDB_DIR = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI2_PDBs")
FIX = os.path.join(_THIS, "fixtures", "one_complex.pkl")


def _load():
    with open(FIX, "rb") as f:
        return pickle.load(f)


def test_one_step_updates_only_adapters():
    sc = ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)
    sc.freeze_base()
    sc.train()
    dom = _load()["binder1"]
    wt = sc.get_wt_seq(dom)
    mut = wt.clone()
    mut[0, 7] = (mut[0, 7] + 1) % 20

    base_name = next(n for n, _ in sc.model.named_parameters() if "lora_" not in n)
    base_before = dict(sc.model.named_parameters())[base_name].detach().clone()
    head_before = [p.detach().clone() for p in sc.head.parameters()]

    opt = torch.optim.Adam(sc.trainable_parameters(), lr=1e-2)
    pred = sc.folding_ddG(dom, mut)
    loss = torch.nn.functional.mse_loss(pred, torch.tensor([1.0]))
    loss.backward()
    opt.step()

    base_after = dict(sc.model.named_parameters())[base_name]
    assert torch.equal(base_before, base_after), "frozen base param changed!"
    head_after = list(sc.head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(head_before, head_after)), \
        "head did not update"


def test_adapter_ckpt_roundtrip(tmp_path):
    sc = ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)
    dom = _load()["binder1"]
    wt = sc.get_wt_seq(dom)
    before = sc.folding_dG(dom, wt).item()
    path = str(tmp_path / "adapters.pt")
    sc.save_adapters(path)

    sc2 = ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)
    sc2.load_adapters(path)
    after = sc2.folding_dG(dom, wt).item()
    assert abs(before - after) < 1e-4, (before, after)
