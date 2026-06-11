"""De-biased stratified analysis of StaB-ddG test-set predictions vs #mutation-sites.

Adds scale-free metrics so strata with different ddG dynamic ranges are comparable:
  - std_true     : std of the ground-truth ddG in the stratum (the natural scale)
  - NRMSE        : RMSE / std_true   (mean-predictor baseline = 1.0; <1 beats it)
  - NMAE         : MAE  / mean|true| (error relative to typical effect size)
  - R2           : 1 - MSE/Var(true) (mean-predictor baseline = 0; <0 worse)
  - slope        : OLS slope of pred~true (1.0 = calibrated; <1 = magnitude shrinkage)
Pearson/Spearman are already scale-free (shown with std_true as range-restriction context).

Two tables: per-exact-n_sites, and cumulative (<=k).
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


def row_metrics(df):
    t = df["ddG"].to_numpy(float); p = df["ddG_pred"].to_numpy(float)
    n = len(df)
    std_t = float(np.std(t)); mabs_t = float(np.mean(np.abs(t)))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    mae = float(np.mean(np.abs(t - p)))
    pear = spear = r2 = slope = float("nan")
    if n >= 3 and std_t > 0:
        if np.std(p) > 0:
            pear = pearsonr(p, t)[0]; spear = spearmanr(p, t)[0]
        r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
        slope = float(np.cov(p, t, bias=True)[0, 1] / np.var(t))
    nrmse = rmse / std_t if std_t > 0 else float("nan")
    nmae = mae / mabs_t if mabs_t > 0 else float("nan")
    return dict(n=n, std=std_t, Pear=pear, Spear=spear,
                RMSE=rmse, NRMSE=nrmse, MAE=mae, NMAE=nmae, R2=r2, slope=slope)


def fmt(label, m):
    def g(x, w=7, p=3):
        return f"{x:>{w}.{p}f}" if x == x else f"{'n/a':>{w}}"
    return (f"{label:>9} {m['n']:>5} {g(m['std'])} | "
            f"{g(m['Pear'])} {g(m['Spear'])} | "
            f"{g(m['RMSE'])} {g(m['NRMSE'])} {g(m['MAE'])} {g(m['NMAE'])} | "
            f"{g(m['R2'])} {g(m['slope'])}")


def header():
    print(f"{'stratum':>9} {'n':>5} {'std_t':>7} | {'Pear':>7} {'Spear':>7} | "
          f"{'RMSE':>7} {'NRMSE':>7} {'MAE':>7} {'NMAE':>7} | {'R2':>7} {'slope':>7}")
    print("-" * 96)


def main():
    df = pd.read_csv(os.path.join(REPO, "baselines/StaB-ddG.csv"))
    df["n_sites"] = df["Mutation"].map(sites)
    ks = sorted(df["n_sites"].unique())

    print("=" * 96)
    print("A) 按 EXACT n_sites(每行=恰好该位点数)")
    print("=" * 96)
    header()
    for k in ks:
        print(fmt(f"={k}", row_metrics(df[df["n_sites"] == k])))
    print(fmt("ALL", row_metrics(df)))

    print("\n" + "=" * 96)
    print("B) 按 CUMULATIVE n_sites(每行=<=k 的全部样本)")
    print("=" * 96)
    header()
    for k in ks:
        print(fmt(f"<={k}", row_metrics(df[df["n_sites"] <= k])))

    print("\n注: NRMSE=RMSE/std_true(=1 即等于'预测均值'基线); R2=1-MSE/Var(=0 即基线);")
    print("    slope=pred~true 回归斜率(=1 标定良好, <1 低估幅度); Pearson/Spearman 本身 scale-free。")


if __name__ == "__main__":
    main()
