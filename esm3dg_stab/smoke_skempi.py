#!/usr/bin/env python3
"""Smoke test the SKEMPI→ESM3 path: encode a couple of test-split complexes,
build mutant tokens (WT-alignment asserts inside), run binding_ddG."""
import os, sys
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
from scorer import ESM3dGScorer
from skempi_data import build_skempi

CKPT = os.path.join(HERE, "..", "data", "esm3dg_weights", "ESM3dG_weights_augmented_1_lora.ckpt")
DATA = os.path.join(HERE, "..", "data")

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = ESM3dGScorer(CKPT, device=dev, trainable=False)
    items = build_skempi(
        scorer,
        csv_path=os.path.join(DATA, "SKEMPI", "filtered_skempi.csv"),
        split_path=os.path.join(DATA, "SKEMPI", "test_pdb.pkl"),
        pdb_dir=os.path.join(DATA, "SKEMPI2_PDBs"),
        pdb_dict_cache_path=os.path.join(HERE, "..", "cache", "skempi_esm3_test_pdb_dict.pkl"),
        limit=2,
    )
    assert len(items) > 0, "no complexes built"
    it = items[0]
    k = min(4, it["ddG"].shape[0])
    print(f"complex {it['name']}: n_mut={it['ddG'].shape[0]}  "
          f"L(complex tokens)={it['complex']['seq'].shape[0]}")
    with torch.no_grad():
        pred = scorer.binding_ddG(it["complex"], it["binder1"], it["binder2"],
                                  it["complex_mut"][:k], it["binder1_mut"][:k], it["binder2_mut"][:k])
    print("pred binding_ddG:", [round(x, 3) for x in pred.detach().cpu().tolist()])
    print("label ddG       :", [round(x, 3) for x in it["ddG"][:k].tolist()])
    assert torch.isfinite(pred).all(), "non-finite predictions"
    print("SKEMPI DATA SMOKE OK")

if __name__ == "__main__":
    main()
