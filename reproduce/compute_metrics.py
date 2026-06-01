"""Compute SKEMPI test-split metrics for a StaB-ddG prediction csv and compare
against the author-provided baselines (StaB-ddG / FoldX / Flex ddG).

Reproduces the numbers behind Figure 3 / the main result table of the paper.

Usage:
    python reproduce/compute_metrics.py --pred cache/eval.csv
"""
import os
import sys
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

# eval_utils lives in baselines/
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "baselines"))
from eval_utils import compute_metrics, t_test  # noqa: E402

warnings.filterwarnings("ignore")

KEYS = [
    "Per Structure Spearman", "Per Structure Spearman Standard Error",
    "Per Structure Pearson", "Per Structure Pearson Standard Error",
    "Per Structure RMSE", "Per Structure RMSE Standard Error",
    "Spearman", "Pearson", "RMSE", "ROC AUC",
]


def show(name, m):
    print(f"\n=== {name} ===")
    print(f"  Per-interface Spearman : {m['Per Structure Spearman']:.3f} "
          f"± {m['Per Structure Spearman Standard Error']:.3f}   <-- paper Fig.3 main metric")
    print(f"  Per-interface Pearson  : {m['Per Structure Pearson']:.3f} "
          f"± {m['Per Structure Pearson Standard Error']:.3f}")
    print(f"  Per-interface RMSE     : {m['Per Structure RMSE']:.3f} "
          f"± {m['Per Structure RMSE Standard Error']:.3f}")
    print(f"  Overall  Spearman      : {m['Spearman']:.3f}")
    print(f"  Overall  Pearson       : {m['Pearson']:.3f}")
    print(f"  Overall  RMSE          : {m['RMSE']:.3f}")
    print(f"  ROC AUC                : {m['ROC AUC']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="cache/eval.csv",
                    help="StaB-ddG prediction csv produced by skempi_eval.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # --- our reproduced predictions ---
    mine = pd.read_csv(os.path.join(REPO, args.pred))
    show(f"REPRODUCED ({args.pred})", compute_metrics(mine, bootstrap=True))

    # --- author-provided baselines, filtered to the test split ---
    base = os.path.join(REPO, "baselines")
    with open(os.path.join(REPO, "data/SKEMPI/test_pdb.pkl"), "rb") as f:
        test_pdbs = pickle.load(f)

    stab = pd.read_csv(os.path.join(base, "StaB-ddG.csv"))
    foldx = pd.read_csv(os.path.join(base, "foldx.csv"))
    flex = pd.read_csv(os.path.join(base, "flexddg.csv"))
    foldx = foldx[foldx["#Pdb"].isin(test_pdbs)]
    flex = flex[flex["#Pdb"].isin(test_pdbs)]

    show("AUTHOR StaB-ddG.csv", compute_metrics(stab, bootstrap=True))
    show("FoldX (test split)", compute_metrics(foldx, bootstrap=True))
    show("Flex ddG (test split)", compute_metrics(flex, bootstrap=True))

    # --- agreement between reproduced and author predictions ---
    print("\n=== Sanity: reproduced vs author StaB-ddG ===")
    merged = mine.merge(
        stab[["#Pdb", "Mutation", "ddG_pred"]],
        on=["#Pdb", "Mutation"], suffixes=("_mine", "_author"), how="inner",
    )
    if len(merged):
        r = np.corrcoef(merged["ddG_pred_mine"], merged["ddG_pred_author"])[0, 1]
        print(f"  matched mutations: {len(merged)} / {len(mine)}")
        print(f"  Pearson(reproduced pred, author pred) = {r:.4f}")
    else:
        print("  (could not match on #Pdb+Mutation; check column formatting)")


if __name__ == "__main__":
    main()
