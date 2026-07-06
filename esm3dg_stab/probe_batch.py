#!/usr/bin/env python3
"""Probe the max TRAINING batch for ESM3dG SKEMPI fine-tuning on the current GPU.
For the N largest SKEMPI complexes (largest = OOM-driving), tries a real TRAINING step
(binding_ddG + loss.backward, trainable=True so activations are retained) at increasing
batch sizes; reports which OOM + peak GPU mem. Also reports StaB's per-complex batch
(stab_tokens // L) and the ESM3dG:StaB effective-batch ratio.
MUST run on the real training GPU (a100-80GB) — local A4500(20GB) is not representative.
"""
import os, sys, argparse
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch, numpy as np
from scorer import ESM3dGScorer
from skempi_data import build_skempi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--batches", default="1,2,3,4,6,8,12,16")
    ap.add_argument("--n_probe", type=int, default=6, help="probe N complexes")
    ap.add_argument("--order", choices=["largest", "smallest", "spread"], default="spread",
                    help="spread = N complexes evenly across the L(complex) spectrum (min..max)")
    ap.add_argument("--chainbreak", action="store_true")
    ap.add_argument("--stab_tokens", type=int, default=10000, help="StaB batch_size token budget")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("GPU:", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
          f"| total {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB" if dev == "cuda" else "", flush=True)

    scorer = ESM3dGScorer(args.ckpt, device=dev, trainable=True)   # trainable => grads/activations retained (training mem)
    items = build_skempi(scorer,
        csv_path=os.path.join(args.data_dir, "SKEMPI", "filtered_skempi.csv"),
        split_path=os.path.join(args.data_dir, "SKEMPI", f"{args.split}_pdb.pkl"),
        pdb_dir=os.path.join(args.data_dir, "SKEMPI2_PDBs"),
        pdb_dict_cache_path=os.path.join(HERE, "..", "cache", f"skempi_esm3_{args.split}_pdb_dict.pkl"),
        chainbreak=args.chainbreak)
    scorer.set_trainable(True)   # re-freeze VQ-VAE etc. after encode; keep LoRA+head trainable
    items_sorted = sorted(items, key=lambda it: it["complex"]["seq"].shape[0])   # ascending L
    if args.order == "spread":
        n = min(args.n_probe, len(items_sorted))
        idxs = [round(i * (len(items_sorted) - 1) / (n - 1)) for i in range(n)] if n > 1 else [len(items_sorted) - 1]
        items = [items_sorted[i] for i in idxs]
    elif args.order == "largest":
        items = items_sorted[::-1][:args.n_probe]
    else:  # smallest
        items = items_sorted[:args.n_probe]
    loss_fn = torch.nn.MSELoss()
    Bs = [int(b) for b in args.batches.split(",")]
    max_ok = {}
    for it in items:
        L = it["complex"]["seq"].shape[0]; N = it["ddG"].shape[0]
        stab_M = max(1, args.stab_tokens // L)
        print(f"\n=== complex {it['name']}  L(complex tokens)={L}  #mut={N}  | StaB batch = {args.stab_tokens}//{L} = {stab_M} seqs ===", flush=True)
        best = 0
        for B in Bs:
            if B > N:
                print(f"  B={B:3d}: > #mut ({N}), skip", flush=True); continue
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for p in scorer.trainable_parameters():
                p.grad = None
            try:
                pred = scorer.binding_ddG(it["complex"], it["binder1"], it["binder2"],
                                          it["complex_mut"][:B], it["binder1_mut"][:B], it["binder2_mut"][:B])
                loss = loss_fn(pred, it["ddG"][:B].to(dev))
                loss.backward()
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  B={B:3d}: OK    peak={peak:5.1f} GB", flush=True)
                best = B
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  B={B:3d}: **OOM**", flush=True)
                break
        max_ok[it["name"]] = (L, stab_M, best)
        if best > 0:
            print(f"  -> max OK batch = {best};  StaB/ESM3dG batch ratio ≈ {stab_M/best:.1f}×", flush=True)

    print("\n===== SUMMARY (largest complexes) =====", flush=True)
    for nm, (L, sM, b) in max_ok.items():
        print(f"  {nm:14s} L={L:5d}  ESM3dG max_batch={b}  StaB batch={sM}  ratio(StaB/ESM3dG)={sM/max(b,1):.1f}×", flush=True)
    print("===== DONE =====", flush=True)


if __name__ == "__main__":
    main()
