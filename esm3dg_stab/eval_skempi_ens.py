#!/usr/bin/env python3
"""Exp3: ESM3dG ZERO-SHOT on SKEMPI test binding ΔΔG (base 3-member ensemble, RAW output).
NO SKEMPI-finetuned adapter is loaded — the scorer built from a base member ckpt IS the
zero-shot base ESM3dG (ESM3dG() loads base LoRA+head+scaling in __init__). We deliberately
do NOT import/call load_adapters.

--chainbreak toggles multi-chain construction for the complex/binders:
  (default)      variant a: chains concatenated, NO chainbreak
  --chainbreak   variant b: chains joined with '|' + one NaN atom37 separator row (ESM3-native;
                 '|'=token 31, STRUCTURE_CHAINBREAK applied inside ESM3.forward; masked out of the mean)

Outputs a per-mutation CSV (#Pdb, ddG, ddG_pred) for unified skempi_metrics.py (THRESHOLD=10),
and prints a quick per-structure(>=2)/overall Spearman for sanity. No sign flip (matches skempi_eval.py).
"""
import os, sys, argparse, time
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from scipy.stats import spearmanr, pearsonr
from scorer import ESM3dGScorer
from skempi_data import build_skempi
from finetune import chunked_binding_ddG   # NOTE: load_adapters intentionally NOT used (zero-shot)


def eval_member(ckpt, data_dir, split, batch_tokens, max_batch, limit, chainbreak, dev):
    scorer = ESM3dGScorer(ckpt, device=dev, trainable=False)   # base ckpt => zero-shot; NO load_adapters
    items = build_skempi(scorer,
        csv_path=os.path.join(data_dir, "SKEMPI", "filtered_skempi.csv"),
        split_path=os.path.join(data_dir, "SKEMPI", f"{split}_pdb.pkl"),
        pdb_dir=os.path.join(data_dir, "SKEMPI2_PDBs"),
        pdb_dict_cache_path=os.path.join(HERE, "..", "cache", f"skempi_esm3_{split}_pdb_dict.pkl"),
        limit=limit, chainbreak=chainbreak)
    res, skipped = {}, []
    t0 = time.time()
    with torch.no_grad():
        for it in items:
            try:
                pred = chunked_binding_ddG(scorer, it, batch_tokens, max_batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                try:
                    pred = chunked_binding_ddG(scorer, it, 1, 1)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); skipped.append(it["name"]); continue
            res[it["name"]] = (pred.detach().cpu().numpy(), it["ddG"].numpy())
    print(f"  [{os.path.basename(ckpt)}] {len(res)} complexes, skipped={len(skipped)} {skipped} ({time.time()-t0:.0f}s)", flush=True)
    del scorer; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma-sep base lora ckpt paths (ensemble)")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--chainbreak", action="store_true", help="variant b: '|' chainbreak between chains")
    ap.add_argument("--batch_tokens", type=int, default=4000)
    ap.add_argument("--max_batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    variant = "chainbreak" if args.chainbreak else "concat"

    per = [eval_member(ck, args.data_dir, args.split, args.batch_tokens, args.max_batch,
                       args.limit, args.chainbreak, dev) for ck in members]
    names = sorted(set.intersection(*[set(p.keys()) for p in per]))   # complexes present in ALL members
    rows, per_struct2, allp, alll = [], [], [], []
    for nm in names:
        preds = np.mean([per[m][nm][0] for m in range(len(members))], axis=0)   # per-mutant ensemble mean
        lab = per[0][nm][1]
        for j in range(len(lab)):
            rows.append((nm, float(lab[j]), float(preds[j])))
        if len(lab) >= 2:
            sp = spearmanr(preds, lab)[0]
            if np.isfinite(sp): per_struct2.append(sp)
        allp.append(preds); alll.append(lab)
    allp = np.concatenate(allp); alll = np.concatenate(alll)

    print(f"\n=== SKEMPI {args.split} ZERO-SHOT ESM3dG ({variant}, base {len(members)}-ens, RAW) ===", flush=True)
    print(f"  complexes={len(names)}  mutants={len(alll)}", flush=True)
    print(f"  per-structure Spearman (>=2, quick): {np.nanmean(per_struct2):.4f}  (n={len(per_struct2)})", flush=True)
    print(f"  overall Spearman: {spearmanr(allp, alll)[0]:.4f}  | overall Pearson: {pearsonr(allp, alll)[0]:.4f}", flush=True)
    print(f"  (baseline ProteinMPNN full-StaB: per-structure 0.448 / overall 0.531; run skempi_metrics.py for THRESHOLD=10 primary metric)", flush=True)
    if args.out:
        import pandas as pd
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(rows, columns=["#Pdb", "ddG", "ddG_pred"]).to_csv(args.out, index=False)
        print(f"wrote per-mutation -> {args.out}", flush=True)
    print("===== DONE =====", flush=True)


if __name__ == "__main__":
    main()
