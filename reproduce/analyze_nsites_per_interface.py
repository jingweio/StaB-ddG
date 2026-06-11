"""Per-interface (within-complex, then averaged across complexes) metrics,
stratified by #mutation-sites — the design-relevant view.

For each stratum: subset rows, group by #Pdb, compute each complex's within-
complex Pearson/Spearman/RMSE/MAE over complexes that have >= THRESH mutations
*in that stratum*, then average across qualifying complexes.

Because stratifying by n_sites thins each complex, we report n_cplx (qualifying
complexes) and run THRESH in {10 (paper standard), 5 (relaxed)}, plus a
cumulative (<=k) table.
"""
import os
import ast
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sites(s):
    try:
        return len(ast.literal_eval(s))
    except Exception:
        return len(str(s).split(","))


def per_interface(df, thresh):
    pe, sp, rm, ma = [], [], [], []
    n_cplx = 0
    for _, g in df.groupby("#Pdb"):
        if len(g) < thresh:
            continue
        n_cplx += 1
        t = g["ddG"].to_numpy(float); p = g["ddG_pred"].to_numpy(float)
        rm.append(np.sqrt(np.mean((t - p) ** 2)))
        ma.append(np.mean(np.abs(t - p)))
        if np.std(t) > 0 and np.std(p) > 0:
            pe.append(pearsonr(p, t)[0]); sp.append(spearmanr(p, t)[0])
    mean = lambda a: float(np.mean(a)) if a else float("nan")
    return dict(n_muts=len(df), n_cplx=n_cplx,
                Pearson=mean(pe), Spearman=mean(sp), RMSE=mean(rm), MAE=mean(ma))


def show(df, thresh, cumulative=False):
    ks = sorted(df["n_sites"].unique())
    print(f"\n{'<=k' if cumulative else '=k':>5} {'n_muts':>7} {'n_cplx':>7} "
          f"{'Pearson':>9} {'Spearman':>9} {'RMSE':>7} {'MAE':>7}   (per-interface, THRESH={thresh})")
    print("-" * 72)
    for k in ks:
        sub = df[df["n_sites"] <= k] if cumulative else df[df["n_sites"] == k]
        m = per_interface(sub, thresh)
        g = lambda x: f"{x:>9.3f}" if x == x else f"{'n/a':>9}"
        gr = lambda x: f"{x:>7.3f}" if x == x else f"{'n/a':>7}"
        lab = f"<={k}" if cumulative else f"={k}"
        print(f"{lab:>5} {m['n_muts']:>7} {m['n_cplx']:>7} {g(m['Pearson'])} "
              f"{g(m['Spearman'])} {gr(m['RMSE'])} {gr(m['MAE'])}")
    m = per_interface(df, thresh)
    g = lambda x: f"{x:>9.3f}" if x == x else f"{'n/a':>9}"
    gr = lambda x: f"{x:>7.3f}" if x == x else f"{'n/a':>7}"
    print(f"{'ALL':>5} {m['n_muts']:>7} {m['n_cplx']:>7} {g(m['Pearson'])} "
          f"{g(m['Spearman'])} {gr(m['RMSE'])} {gr(m['MAE'])}")


def main():
    df = pd.read_csv(os.path.join(REPO, "baselines/StaB-ddG.csv"))
    df["n_sites"] = df["Mutation"].map(sites)

    print("=" * 72)
    print("PER-INTERFACE 指标,按 EXACT n_sites(每复合物组内算后跨复合物平均)")
    print("=" * 72)
    show(df, thresh=10)
    show(df, thresh=5)

    print("\n" + "=" * 72)
    print("PER-INTERFACE 指标,按 CUMULATIVE (<=k) n_sites")
    print("=" * 72)
    show(df, thresh=10, cumulative=True)


if __name__ == "__main__":
    main()
