#!/usr/bin/env python3
"""Phase-0 smoke test: load ESM3ΔG (LoRA + stability head) and predict absolute ΔG
on a couple of example domains. Validates env + gated ESM3 weights + ckpt loading
before we build the SKEMPI binding-ddG bridge.

Run (in esm3dg env):
  cd <worktree>/esm3dg_stab
  HF_HOME=/home/guoj0f/share/hf_cache HF_HUB_OFFLINE=1 python smoke_load_predict.py
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `from model_utils import ...` inside esm3dg_model resolves

import torch
from esm3dg_model import ESM3dG, ESM3dG_predict

CKPT = os.path.join(HERE, "..", "data", "esm3dg_weights", "ESM3dG_weights_augmented_1_lora.ckpt")
EX = os.path.join(HERE, "test_examples")

def main():
    print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    print("loading ESM3ΔG from", CKPT)
    t0 = time.time()
    model = ESM3dG(CKPT)          # ESM3(frozen)+LoRA r=4 + stability head + sigmoid scaling
    print(f"  loaded in {time.time()-t0:.1f}s")

    for pdb in ["mgnify_1A0N.pdb", "stability_1998469.pdb"]:
        path = os.path.join(EX, pdb)
        if not os.path.exists(path):
            print(f"  [skip] {pdb} missing"); continue
        t0 = time.time()
        pred_dg, pred_dg_avg, seq = ESM3dG_predict(model, path, chain_id="A")
        print(f"  {pdb}: L={len(seq)}  ΔG={pred_dg_avg[0]:.3f} kcal/mol  ({time.time()-t0:.1f}s)")
    print("SMOKE OK")

if __name__ == "__main__":
    main()
