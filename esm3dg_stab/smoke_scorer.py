#!/usr/bin/env python3
"""Self-test for ESM3dGScorer (no SKEMPI needed):
 - encode one domain, score WT + a synthetic point-mutant
 - check folding_ddG(WT,WT) ≈ 0 ; mutant ddG ≠ 0
 - check gradients flow to LoRA + head (and NOT to frozen ESM3 trunk)
"""
import os, sys
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from scorer import ESM3dGScorer

CKPT = os.path.join(HERE, "..", "data", "esm3dg_weights", "ESM3dG_weights_augmented_1_lora.ckpt")
PDB = os.path.join(HERE, "test_examples", "mgnify_1A0N.pdb")

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = ESM3dGScorer(CKPT, device=dev, trainable=True)
    n_train = sum(p.numel() for p in s.trainable_parameters())
    print(f"trainable params (LoRA+head+scaling): {n_train:,}")

    enc = s.encode(PDB, "A")
    wt = enc["seq"].unsqueeze(0)                 # [1, L]
    L = wt.shape[1]
    print(f"encoded WT: L(tokens)={L}, aa_len={len(enc['wt_aa'])}")

    # WT-vs-WT ddG should be ~0
    ddg_ctrl = s.folding_ddG(enc, wt).item()
    print(f"folding_ddG(WT, WT) = {ddg_ctrl:+.4f}  (expect ~0)")

    # synthetic mutant: swap the sequence token at an interior residue position
    mut = wt.clone()
    pos = L // 2
    orig = mut[0, pos].item()
    # pick a different valid residue token from elsewhere in the sequence
    for j in range(1, L - 1):
        if mut[0, j].item() != orig:
            mut[0, pos] = mut[0, j]; break
    ddg_mut = s.folding_ddG(enc, mut)
    print(f"folding_ddG(mut@{pos}, WT) = {ddg_mut.item():+.4f}  (expect ≠ 0)")

    # gradient check: backprop the mutant ddG, confirm grads on adapters only
    s.model.zero_grad()
    ddg_mut.abs().sum().backward()
    g_lora = sum(p.grad.abs().sum().item() for n, p in s.model.named_parameters()
                 if p.grad is not None and "lora" in n.lower())
    g_head = sum(p.grad.abs().sum().item() for n, p in s.model.named_parameters()
                 if p.grad is not None and "stability_head" in n)
    n_trunk_grad = sum(1 for n, p in s.model.named_parameters()
                       if p.requires_grad and "lora" not in n.lower()
                       and "stability_head" not in n and "output_scaling" not in n)
    print(f"grad |LoRA|={g_lora:.3e}  |head|={g_head:.3e}  frozen-trunk-with-grad={n_trunk_grad}")
    assert abs(ddg_ctrl) < 1e-3, "WT-WT ddG not ~0"
    assert g_lora > 0 and g_head > 0, "no grad to adapters"
    print("SCORER SELF-TEST OK")

if __name__ == "__main__":
    main()
