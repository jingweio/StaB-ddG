#!/usr/bin/env python3
"""Unified SKEMPI metric — matches StaB baselines/eval_utils.compute_metrics (THRESHOLD=10):
per-structure Spearman = mean over complexes with >= THRESHOLD mutations; overall Spearman/Pearson.
Use this ONE tool on ALL models' per-mutation CSVs (ESM3dG concat/chainbreak + ProteinMPNN) so the
comparison — and the comparison to the StaB baseline 0.448/0.531 — is apples-to-apples.

Usage: python skempi_metrics.py <csv> [<csv> ...]   (each csv must have columns #Pdb, ddG, ddG_pred)
"""
import sys, argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--threshold", type=int, default=10)
ap.add_argument("--label", default="ddG")
ap.add_argument("--pred", default="ddG_pred")
args = ap.parse_args()

for csv in args.csvs:
    df = pd.read_csv(csv)
    lab = df[args.label].to_numpy(); pred = df[args.pred].to_numpy()
    ov_sp = spearmanr(pred, lab)[0]; ov_pr = pearsonr(pred, lab)[0]
    per = []
    for _, g in df.groupby("#Pdb"):
        if len(g) < args.threshold:
            continue
        sp = spearmanr(g[args.pred], g[args.label])[0]
        if np.isfinite(sp):
            per.append(sp)
    print(f"{csv}")
    print(f"  n_mut={len(df)}  n_complex={df['#Pdb'].nunique()}  n_complex(>={args.threshold} mut)={len(per)}")
    print(f"  per-structure Spearman (mean, THRESHOLD={args.threshold}): {np.mean(per):.4f}")
    print(f"  overall Spearman: {ov_sp:.4f}  | overall Pearson: {ov_pr:.4f}\n")
