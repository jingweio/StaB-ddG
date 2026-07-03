#!/usr/bin/env python3
"""Task1(1b): evaluate StaB's Megascale-stage1 ProteinMPNN (model_ckpts/stability_finetuned.pt)
on the Megascale TEST split (28 domains) — folding ΔΔG per domain vs ddG_ML.

Reuses StaB's own model (StaBddG.folding_ddG) + data construction (== stability_finetune.py).
ProteinMPNN's stability head is stochastic (random decoding order + backbone noise); we
MC-ensemble `--mc` samples (default 20, matching run_stabddg.py) and average — its reported
configuration. Metric: per-domain Spearman (mean) + overall Spearman/Pearson.
"""
import os, sys, argparse, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from stabddg.mpnn_utils import StructureDataset, ProteinMPNN, parse_PDB
from stabddg.model import StaBddG

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def load_data(data_dir, split, pdb_dir, stability_csv, splits_pkl):
    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    names = splits[split]
    names = names.tolist() if hasattr(names, "tolist") else list(names)

    pdb_dict = []
    kept = []
    for name in names:
        clean = name.split(".pdb", 1)[0] + ".pdb"
        clean = clean.replace("|", ":")
        path = os.path.join(pdb_dir, clean)
        if not os.path.exists(path):
            print(f"  [skip missing structure] {clean}", flush=True); continue
        pdb_dict.append(parse_PDB(path)[0]); kept.append(name)

    df = pd.read_csv(stability_csv, low_memory=False, usecols=["aa_seq", "ddG_ML", "mut_type", "WT_name"])
    df = df[df["ddG_ML"] != "-"]
    df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)

    ddG_data = {}
    for name in kept:
        clean = name.split(".pdb", 1)[0] + ".pdb"
        clean = clean.replace("|", ":")
        sub = df[(df["WT_name"] == name) & (df["mut_type"] != "wt")]
        seqs = sub["aa_seq"].tolist()
        idx = np.vstack([np.asarray([ALPHABET.index(a) for a in s], dtype=np.int64) for s in seqs]) if seqs else np.zeros((0, 0), np.int64)
        ddG_data[clean] = {"mut_seqs": torch.from_numpy(idx),
                           "ddG": torch.tensor(sub["ddG_ML"].to_numpy(dtype=np.float32))}
    for d in pdb_dict:
        d["masked_list"] = ["A"]; d["visible_list"] = []
    return StructureDataset(pdb_dict, truncate=None, max_length=3000), ddG_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "model_ckpts", "stability_finetuned.pt"))
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--mc", type=int, default=20, help="MC ensemble samples (run_stabddg default 20)")
    ap.add_argument("--batch_size", type=int, default=10000, help="token budget per batch")
    ap.add_argument("--noise_level", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pdb_dir = os.path.join(args.data_dir, "megascale", "AlphaFold_model_PDBs")
    stability_csv = os.path.join(args.data_dir, "megascale", "Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
    splits_pkl = os.path.join(args.data_dir, "rocklin", "mega_splits.pkl")
    dataset, ddG_data = load_data(args.data_dir, args.split, pdb_dir, stability_csv, splits_pkl)

    pmpnn = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48,
                        dropout=0.0, augment_eps=0.0)
    ck = torch.load(args.checkpoint, map_location=dev)
    pmpnn.load_state_dict(ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck)
    print("loaded ProteinMPNN:", args.checkpoint, flush=True)
    model = StaBddG(pmpnn=pmpnn, use_antithetic_variates=True, noise_level=args.noise_level, device=dev)
    model.to(dev); model.eval()

    per_dom_sp, all_pred, all_lab, rows = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for sample in dataset:
            name = sample["name"]
            key = f"{name}.pdb"
            if key not in ddG_data or ddG_data[key]["mut_seqs"].shape[0] == 0:
                continue
            ddG = ddG_data[key]["ddG"]
            mut_seqs = ddG_data[key]["mut_seqs"]
            N, L = mut_seqs.shape
            M = max(1, args.batch_size // L)
            mc_preds = []
            for _ in range(args.mc):
                chunk = []
                for bi in range(0, N, M):
                    p = model.folding_ddG(sample, mut_seqs[bi:bi + M])
                    chunk.append(p.detach().cpu())
                mc_preds.append(torch.cat(chunk))
            pred = torch.stack(mc_preds).mean(dim=0).numpy()
            lab = ddG.numpy()
            sp = spearmanr(pred, lab)[0]
            per_dom_sp.append(sp); all_pred.append(pred); all_lab.append(lab)
            rows.append((name, N, sp))
            print(f"    {name:32s} n={N:5d}  sp={sp:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    all_pred = np.concatenate(all_pred); all_lab = np.concatenate(all_lab)
    print(f"\n=== Megascale {args.split} eval  (ProteinMPNN stage1, {args.mc}x MC; {len(rows)} domains) ===", flush=True)
    print(f"  per-domain Spearman (mean): {np.nanmean(per_dom_sp):.4f}  (n={len(rows)})", flush=True)
    print(f"  overall Spearman: {spearmanr(all_pred, all_lab)[0]:.4f}  | overall Pearson: {pearsonr(all_pred, all_lab)[0]:.4f}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pd.DataFrame(rows, columns=["domain", "n_mut", "spearman"]).to_csv(args.out, index=False)
        print(f"wrote per-domain -> {args.out}", flush=True)
    print("===== DONE =====", flush=True)


if __name__ == "__main__":
    main()
