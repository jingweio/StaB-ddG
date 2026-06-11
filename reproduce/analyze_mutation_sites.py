"""Stratify by number of mutation sites and report (1) the distribution across
train / test / full filtered SKEMPI, and (2) StaB-ddG test-set performance per
stratum (every site-count listed, no bucketing).

- Distribution source: data/SKEMPI/filtered_skempi.csv (full filtered set),
  split into train/test via data/SKEMPI/{train,test}_pdb.pkl.
- Performance source: baselines/StaB-ddG.csv (test split, has predictions).
"""
import os
import ast
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sites_liststr(s):
    try:
        return len(ast.literal_eval(s))
    except Exception:
        return len(str(s).split(","))


def metrics(df):
    t = df["ddG"].to_numpy(float); p = df["ddG_pred"].to_numpy(float)
    pear = spear = float("nan")
    if len(df) >= 3 and np.std(t) > 0 and np.std(p) > 0:
        pear = pearsonr(p, t)[0]; spear = spearmanr(p, t)[0]
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    return len(df), pear, spear, rmse, mae


def main():
    # ---------- distribution: train / test / full ----------
    full = pd.read_csv(os.path.join(REPO, "data/SKEMPI/filtered_skempi.csv"))
    full["n_sites"] = full["Mutation(s)_cleaned"].map(lambda s: len(str(s).split(",")))
    train_pdbs = pickle.load(open(os.path.join(REPO, "data/SKEMPI/train_pdb.pkl"), "rb"))
    test_pdbs = pickle.load(open(os.path.join(REPO, "data/SKEMPI/test_pdb.pkl"), "rb"))
    train = full[full["#Pdb"].isin(train_pdbs)]
    test = full[full["#Pdb"].isin(test_pdbs)]

    print("=" * 70)
    print("(1) #mutation-sites 分布:训练集 / 测试集 / 全集")
    print("=" * 70)
    print(f"行数  -> 训练:{len(train)}  测试:{len(test)}  全集:{len(full)}  "
          f"(train+test={len(train)+len(test)})")
    print(f"复合物-> 训练:{train['#Pdb'].nunique()}  测试:{test['#Pdb'].nunique()}  "
          f"全集:{full['#Pdb'].nunique()}\n")
    allk = sorted(set(full["n_sites"]) | set(train["n_sites"]) | set(test["n_sites"]))
    print(f"{'n_sites':>7} | {'训练集':>14} | {'测试集':>14} | {'全集':>14}")
    print("-" * 62)
    dtr, dte, dfu = train["n_sites"].value_counts(), test["n_sites"].value_counts(), full["n_sites"].value_counts()
    for k in allk:
        ctr, cte, cfu = int(dtr.get(k, 0)), int(dte.get(k, 0)), int(dfu.get(k, 0))
        print(f"{k:>7} | {ctr:>6} ({100*ctr/len(train):4.1f}%) | "
              f"{cte:>6} ({100*cte/len(test):4.1f}%) | {cfu:>6} ({100*cfu/len(full):4.1f}%)")

    # ---------- performance on test (StaB-ddG.csv) ----------
    stab = pd.read_csv(os.path.join(REPO, "baselines/StaB-ddG.csv"))
    stab["n_sites"] = stab["Mutation"].map(sites_liststr)
    print("\n" + "=" * 70)
    print("(2) StaB-ddG 测试集表现,按 #mutation-sites 全部列出")
    print("=" * 70)
    print(f"{'n_sites':>7} {'count':>6} {'Pearson':>9} {'Spearman':>9} {'RMSE':>7} {'MAE':>7}")
    print("-" * 50)
    for k in sorted(stab["n_sites"].unique()):
        n, pe, sp, rm, ma = metrics(stab[stab["n_sites"] == k])
        pe_s = f"{pe:>9.3f}" if pe == pe else f"{'n/a':>9}"
        sp_s = f"{sp:>9.3f}" if sp == sp else f"{'n/a':>9}"
        print(f"{k:>7} {n:>6} {pe_s} {sp_s} {rm:>7.3f} {ma:>7.3f}")
    for name, sub in [("=1", stab[stab["n_sites"] == 1]),
                      (">=2", stab[stab["n_sites"] >= 2]),
                      ("ALL", stab)]:
        n, pe, sp, rm, ma = metrics(sub)
        print(f"{name:>7} {n:>6} {pe:>9.3f} {sp:>9.3f} {rm:>7.3f} {ma:>7.3f}")


if __name__ == "__main__":
    main()
