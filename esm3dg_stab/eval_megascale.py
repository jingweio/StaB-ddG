#!/usr/bin/env python3
"""Task1(1a): evaluate pretrained ESM3dG (base 3-member ensemble) on the Megascale
TEST split (28 domains) — folding ΔΔG per domain, correlated with ddG_ML.

Metric: per-domain Spearman (mean over domains) + overall Spearman/Pearson.
Raw (unscaled) folding_ddG = dG(mut) - dG(wt), consistent with our ΔΔG pipeline.
Ensemble = average predicted ddG across members (weights_1/2/3).
"""
import os, sys, argparse, time
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from scorer import ESM3dGScorer
from megascale_data import build_megascale


def domain_pred(scorer, item, max_batch):
    """predicted folding ddG [n] for one domain's mutants (raw)."""
    enc = item["enc"]
    wt = enc["seq"].unsqueeze(0).to(scorer.device)
    wt_dG = scorer.folding_dG(enc, wt)                       # [1]
    mut = item["mut"]
    N = mut.shape[0]
    out = []
    M = max_batch
    i = 0
    while i < N:
        b = mut[i:i + M].to(scorer.device)
        try:
            with torch.no_grad():
                mut_dG = scorer.folding_dG(enc, b)           # [B]
            out.append((mut_dG - wt_dG).detach().cpu())
            i += M
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if M == 1:
                raise
            M = max(1, M // 2)
    return torch.cat(out).numpy()


def eval_member(ckpt, data_dir, split, max_batch, limit, dev):
    scorer = ESM3dGScorer(ckpt, device=dev, trainable=False)
    items = build_megascale(scorer, data_dir, split=split, limit=limit)
    res = {}
    t0 = time.time()
    for it in items:
        pred = domain_pred(scorer, it, max_batch)
        res[it["name"]] = (pred, it["ddG"].numpy())
    print(f"  [{os.path.basename(ckpt)}] {len(items)} domains ({time.time()-t0:.0f}s)", flush=True)
    del scorer; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma-sep lora ckpt paths (ensemble)")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    per = [eval_member(ck, args.data_dir, args.split, args.max_batch, args.limit, dev) for ck in members]

    # ensemble: domains present in ALL members
    domains = sorted(set.intersection(*[set(p.keys()) for p in per]))
    per_dom_sp, all_pred, all_lab = [], [], []
    rows = []
    for d in domains:
        preds = np.mean([per[m][d][0] for m in range(len(members))], axis=0)
        lab = per[0][d][1]
        sp = spearmanr(preds, lab)[0]
        per_dom_sp.append(sp)
        all_pred.append(preds); all_lab.append(lab)
        rows.append((d, len(lab), sp))
    all_pred = np.concatenate(all_pred); all_lab = np.concatenate(all_lab)
    print(f"\n=== Megascale {args.split} eval  (ESM3dG base {len(members)}-ens; {len(domains)} domains) ===", flush=True)
    print(f"  per-domain Spearman (mean): {np.nanmean(per_dom_sp):.4f}  (n={len(domains)})", flush=True)
    print(f"  overall Spearman: {spearmanr(all_pred, all_lab)[0]:.4f}  | overall Pearson: {pearsonr(all_pred, all_lab)[0]:.4f}", flush=True)
    for d, n, sp in rows:
        print(f"    {d:32s} n={n:5d}  sp={sp:.3f}", flush=True)
    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(rows, columns=["domain", "n_mut", "spearman"]).to_csv(args.out, index=False)
        print(f"wrote per-domain -> {args.out}", flush=True)
    print("===== DONE =====", flush=True)


if __name__ == "__main__":
    main()
