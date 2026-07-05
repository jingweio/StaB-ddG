#!/usr/bin/env python3
"""Task1(1a): evaluate pretrained ESM3dG (base 3-member ensemble) on the Megascale
TEST split (28 domains) — folding ΔΔG per domain, correlated with ddG_ML.

Megascale is a cDNA-display-proteolysis dataset -> per the paper the sigmoid correction
is ON for cDNA datasets, so we report **scaled (primary)** and raw (extra verification);
both come from ONE forward via scorer.folding_dG_both. ddG = dG(mut) - dG(wt).
Metric: per-domain Spearman (mean over domains) + overall Spearman/Pearson.
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
    """(raw_ddg[N], scaled_ddg[N]) for one domain's mutants — one forward gives both."""
    enc = item["enc"]
    wt = enc["seq"].unsqueeze(0).to(scorer.device)
    with torch.no_grad():
        wt_raw, wt_scl = scorer.folding_dG_both(enc, wt)      # [1],[1]
    mut = item["mut"]
    N = mut.shape[0]
    raw_out, scl_out = [], []
    M = max_batch
    i = 0
    while i < N:
        b = mut[i:i + M].to(scorer.device)
        try:
            with torch.no_grad():
                m_raw, m_scl = scorer.folding_dG_both(enc, b)  # [B],[B]
            raw_out.append((m_raw - wt_raw).detach().cpu())
            scl_out.append((m_scl - wt_scl).detach().cpu())
            i += M
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if M == 1:
                raise
            M = max(1, M // 2)
    return torch.cat(raw_out).numpy(), torch.cat(scl_out).numpy()


def eval_member(ckpt, data_dir, split, max_batch, limit, dev):
    scorer = ESM3dGScorer(ckpt, device=dev, trainable=False)
    items = build_megascale(scorer, data_dir, split=split, limit=limit)
    res = {}
    t0 = time.time()
    for it in items:
        raw, scl = domain_pred(scorer, it, max_batch)
        res[it["name"]] = (raw, scl, it["ddG"].numpy())
    print(f"  [{os.path.basename(ckpt)}] {len(items)} domains ({time.time()-t0:.0f}s)", flush=True)
    del scorer; torch.cuda.empty_cache()
    return res


def aggregate(per, members, domains, which):
    """which: 1=scaled, 0=raw. Returns (per-domain mean sp, overall sp, overall pr, rows)."""
    per_dom, allp, alll, rows = [], [], [], []
    for d in domains:
        preds = np.mean([per[m][d][which] for m in range(len(members))], axis=0)
        lab = per[0][d][2]
        sp = spearmanr(preds, lab)[0]
        per_dom.append(sp); allp.append(preds); alll.append(lab)
        rows.append((d, len(lab), sp))
    allp = np.concatenate(allp); alll = np.concatenate(alll)
    return np.nanmean(per_dom), spearmanr(allp, alll)[0], pearsonr(allp, alll)[0], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma-sep lora ckpt paths (ensemble)")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    per = [eval_member(ck, args.data_dir, args.split, args.max_batch, args.limit, dev) for ck in members]
    domains = sorted(set.intersection(*[set(p.keys()) for p in per]))

    print(f"\n=== Megascale {args.split} eval  (ESM3dG base {len(members)}-ens; {len(domains)} domains) ===", flush=True)
    results = {}
    for tag, which in (("scaled(PRIMARY, cDNA sigmoid-on)", 1), ("raw(extra verification)", 0)):
        pdm, ov_sp, ov_pr, rows = aggregate(per, members, domains, which)
        results[which] = (pdm, ov_sp, ov_pr, rows)
        print(f"  [{tag}] per-domain Spearman (mean): {pdm:.4f}  | overall Spearman: {ov_sp:.4f}  | overall Pearson: {ov_pr:.4f}", flush=True)

    # per-domain rows (scaled primary)
    _, _, _, rows_scaled = results[1]
    _, _, _, rows_raw = results[0]
    for (d, n, sp_s), (_, _, sp_r) in zip(rows_scaled, rows_raw):
        print(f"    {d:32s} n={n:5d}  scaled_sp={sp_s:.3f}  raw_sp={sp_r:.3f}", flush=True)
    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame([(d, n, ss, sr) for (d, n, ss), (_, _, sr) in zip(rows_scaled, rows_raw)],
                     columns=["domain", "n_mut", "scaled_spearman", "raw_spearman"]).to_csv(args.out, index=False)
        print(f"wrote per-domain -> {args.out}", flush=True)
    print("===== DONE =====", flush=True)


if __name__ == "__main__":
    main()
