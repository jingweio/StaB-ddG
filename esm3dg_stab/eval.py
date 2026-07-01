#!/usr/bin/env python3
"""Evaluate a fine-tuned ESM3ΔG×SKEMPI model on the SKEMPI test split.
Reports per-structure Spearman (mean over complexes) + overall Spearman — matches
the StaB-ddG headline metric (ProteinMPNN baseline per-structure 0.448 / overall 0.531)."""
import os, sys, argparse
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from scorer import ESM3dGScorer
from skempi_data import build_skempi
from finetune import load_adapters, chunked_binding_ddG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_ckpt", required=True)
    ap.add_argument("--checkpoint", required=True, help="fine-tuned adapters to evaluate")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_tokens", type=int, default=8000)
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    scorer = ESM3dGScorer(args.lora_ckpt, device=dev, trainable=False)
    load_adapters(scorer, args.checkpoint)
    scorer.model.eval()

    items = build_skempi(scorer,
        csv_path=os.path.join(args.data_dir, "SKEMPI", "filtered_skempi.csv"),
        split_path=os.path.join(args.data_dir, "SKEMPI", f"{args.split}_pdb.pkl"),
        pdb_dir=os.path.join(args.data_dir, "SKEMPI2_PDBs"),
        pdb_dict_cache_path=os.path.join(HERE, "..", "cache", f"skempi_esm3_{args.split}_pdb_dict.pkl"),
        limit=args.limit)

    per_struct, all_pred, all_label, rows = [], [], [], []
    skipped = []
    with torch.no_grad():
        for it in items:
            try:
                pred = chunked_binding_ddG(scorer, it, args.batch_tokens, args.max_batch).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                try:  # retry one sequence at a time
                    pred = chunked_binding_ddG(scorer, it, 1, 1).cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); skipped.append(it["name"]); continue
            label = it["ddG"].cpu().numpy()
            all_pred.append(pred); all_label.append(label)
            for p, l in zip(pred, label):
                rows.append((it["name"], float(l), float(p)))
            if len(label) >= 2:
                sp, _ = spearmanr(pred, label)
                if np.isfinite(sp): per_struct.append(sp)
    if skipped:
        print(f"WARNING: skipped {len(skipped)} complexes (OOM even at M=1): {skipped}")
    all_pred = np.concatenate(all_pred); all_label = np.concatenate(all_label)
    overall_sp, _ = spearmanr(all_pred, all_label)
    overall_pr, _ = pearsonr(all_pred, all_label)
    print(f"\n=== SKEMPI {args.split} eval ===")
    print(f"complexes={len(items)}  mutants={len(all_label)}")
    print(f"per-structure Spearman (mean): {np.mean(per_struct):.4f}  (n={len(per_struct)})")
    print(f"overall Spearman: {overall_sp:.4f}  | overall Pearson: {overall_pr:.4f}")
    print("(baseline ProteinMPNN: per-structure 0.448 / overall 0.531)")
    if args.out:
        import csv
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["#Pdb", "ddG", "ddG_pred"])
            w.writerows(rows)
        print(f"wrote predictions -> {args.out}")

if __name__ == "__main__":
    main()
