#!/usr/bin/env python3
"""exp4: evaluate FINE-TUNED ESM3dG adapters on SKEMPI test binding ΔΔG.

Loads ONE base member + a fine-tuned adapter (LoRA+head+scaling from finetune.py's save_adapters)
on top, then scores test. The adapter fully overwrites the trainable params, so the base member
only supplies the (identical, frozen) ESM3 trunk — pass the matching base for cleanliness.

Efficient: builds the scorer + encodes the test set ONCE, then loops over --adapters, calling
load_adapters for each and re-scoring (no reload). Writes one per-mutation CSV (#Pdb,ddG,ddG_pred)
per adapter; run skempi_metrics.py on them for the THRESHOLD=10 per-structure metric (vs StaB 0.448).

Match the construction the adapter was TRAINED with: exp4 = --chainbreak --mask_pipe.
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
from finetune import chunked_binding_ddG, load_adapters


def score_items(scorer, items, batch_tokens, max_batch):
    res, skipped = {}, []
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
    return res, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True, help="base member (weights_1 or augmented_1) for the frozen trunk")
    ap.add_argument("--adapters", required=True, help="comma-sep fine-tuned adapter .pt paths to evaluate")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--chainbreak", action="store_true")
    ap.add_argument("--mask_pipe", action="store_true")
    ap.add_argument("--batch_tokens", type=int, default=4000)
    ap.add_argument("--max_batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "ibex-records", "ESM3ΔG-SKEMPIv2", "results"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]

    # build scorer + encode test set ONCE
    scorer = ESM3dGScorer(args.base_ckpt, device=dev, trainable=False, mask_pipe=args.mask_pipe)
    items = build_skempi(scorer,
        csv_path=os.path.join(args.data_dir, "SKEMPI", "filtered_skempi.csv"),
        split_path=os.path.join(args.data_dir, "SKEMPI", f"{args.split}_pdb.pkl"),
        pdb_dir=os.path.join(args.data_dir, "SKEMPI2_PDBs"),
        pdb_dict_cache_path=os.path.join(HERE, "..", "cache", f"skempi_esm3_{args.split}_pdb_dict.pkl"),
        limit=args.limit, chainbreak=args.chainbreak)
    os.makedirs(args.out_dir, exist_ok=True)
    import pandas as pd

    for ad in adapters:
        if not os.path.exists(ad):
            print(f"[skip missing] {ad}", flush=True); continue
        load_adapters(scorer, ad)                       # overwrites LoRA+head+scaling with the fine-tuned ones
        scorer.model.eval()
        t0 = time.time()
        res, skipped = score_items(scorer, items, args.batch_tokens, args.max_batch)
        rows, per2, allp, alll = [], [], [], []
        for nm in sorted(res):
            preds, lab = res[nm]
            for j in range(len(lab)):
                rows.append((nm, float(lab[j]), float(preds[j])))
            if len(lab) >= 2:
                sp = spearmanr(preds, lab)[0]
                if np.isfinite(sp): per2.append(sp)
            allp.append(preds); alll.append(lab)
        allp = np.concatenate(allp); alll = np.concatenate(alll)
        tag = os.path.splitext(os.path.basename(ad))[0]           # e.g. esm3dg_skempi_cbmask_base_lr1e-6_ep10
        out = os.path.join(args.out_dir, f"eval_{tag}.csv")
        pd.DataFrame(rows, columns=["#Pdb", "ddG", "ddG_pred"]).to_csv(out, index=False)
        print(f"[{tag}] complexes={len(res)} skipped={len(skipped)} "
              f"per-struct(>=2)={np.nanmean(per2):.4f} overall_sp={spearmanr(allp, alll)[0]:.4f} "
              f"overall_pr={pearsonr(allp, alll)[0]:.4f}  ({time.time()-t0:.0f}s) -> {os.path.basename(out)}", flush=True)
    print("===== DONE (run skempi_metrics.py on the CSVs for THRESHOLD=10 per-structure) =====", flush=True)


if __name__ == "__main__":
    main()
