#!/usr/bin/env python
"""
Mutation-cliff analysis on the SKEMPIv2 test split (homology-OOD), using the
StaB-ddG predictions reproduced on Ibex (job 47437925, bs=25000).

A "mutation cliff" is the protein-binding analogue of an *activity cliff* in
medicinal chemistry: a pair of mutants that are nearly identical in sequence
(a matched mutant pair, MMP) yet differ sharply in binding effect (large
|ΔΔG_bind| gap). We answer two questions:

  Q1. Does the SKEMPIv2 test split exhibit significant mutation cliffs?
  Q2. How well does StaB-ddG handle them?

Methodology
-----------
Mutation token format: {WT}{Chain}{Pos}{MUT}, e.g. 'HB8A' = His->Ala, chain B, pos 8.

Matched mutant pairs (within the same complex #Pdb):
  * SPS (same-position substitution): two SINGLE mutants at the identical
    (complex, chain, WT, position) differing only in the introduced residue.
    Structural context is maximally identical -> the purest cliff probe.
  * H1 (Hamming-1): two mutants whose mutation-token *sets* differ by exactly
    one token (one site added/removed, or one site's identity changed).
    Generalises cliffs to the multi-mutant landscape.

Landscape index (activity-cliff SALI analogue):
  SALI_ij = |ΔΔG_i - ΔΔG_j| / d_ij , with d_ij = mutation (set-symmetric) distance.
  For SPS pairs d_ij = 1, so SALI == |ΔΔG gap|.

Model handling:
  * Pairwise concordance: sign agreement between Δexp and Δpred per MMP,
    stratified by exp gap (cliff vs smooth).
  * Gap-recovery / smoothing: regression of |Δpred| on |Δexp| (slope<1 = model
    compresses/smooths the landscape); signed Spearman of Δpred vs Δexp.
  * Per-site ranking: within multi-substitution sites, can the model rank the
    substitutions (within-site Spearman, averaged over sites)?
"""
import ast
import re
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/home/guoj0f/repos/ibex_records/stab-ddg_repro/eval_skempi_ft_47437925.csv"
OUTDIR = "/home/guoj0f/repos/StaB-ddG/reproduce/mutation-analysis"
RNG = np.random.default_rng(0)

TOK = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")


def parse_token(t):
    """Return (wt, chain, pos, mut) or None."""
    m = TOK.match(t)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def load():
    df = pd.read_csv(CSV, index_col=0).reset_index(drop=True)
    df["muts"] = df["Mutation"].apply(ast.literal_eval)
    df["nsites"] = df["muts"].apply(len)
    # token set (frozenset of full tokens) for Hamming distance over mutation sets
    df["tokset"] = df["muts"].apply(frozenset)
    return df


# ----------------------------------------------------------------------------
# Pair builders
# ----------------------------------------------------------------------------
def sps_pairs(df):
    """Same-position substitution pairs among single mutants."""
    single = df[df["nsites"] == 1].copy()
    parsed = single["muts"].apply(lambda x: parse_token(x[0]))
    single = single[parsed.notna()].copy()
    single["p"] = parsed[parsed.notna()]
    single["site"] = single["p"].apply(lambda p: None)
    single["site"] = single.apply(
        lambda r: (r["#Pdb"], r["p"][1], r["p"][0], r["p"][2]), axis=1
    )  # (pdb, chain, wt, pos)
    single["mut_aa"] = single["p"].apply(lambda p: p[3])
    pairs = []
    for site, g in single.groupby("site"):
        if len(g) < 2:
            continue
        for i, j in itertools.combinations(g.index, 2):
            pairs.append((i, j, 1, site))
    return single, pairs


def hamming1_pairs(df):
    """Pairs whose mutation-token SETS differ by exactly one token (set symmetric
    difference == 1, i.e. add/remove one site) OR one site identity swap
    (symmetric difference == 2 but same positions). We use set-distance = size of
    symmetric difference / 2 rounded; treat d=1 as the smallest sequence step.

    To keep it well-defined and cheap, we restrict to same complex and require
    the union of mutated *positions* to differ minimally."""
    pairs = []
    for pdb, g in df.groupby("#Pdb"):
        idx = g.index.tolist()
        toks = {i: df.at[i, "tokset"] for i in idx}
        # position sets (chain+pos) to detect "same sites, one identity changed"
        possets = {}
        for i in idx:
            ps = set()
            ok = True
            for t in df.at[i, "muts"]:
                p = parse_token(t)
                if p is None:
                    ok = False
                    break
                ps.add((p[1], p[2]))  # chain,pos
            possets[i] = ps if ok else None
        for i, j in itertools.combinations(idx, 2):
            symd = toks[i] ^ toks[j]
            # case A: exactly one token added/removed (sizes differ by 1, subset)
            if len(symd) == 1:
                pairs.append((i, j, 1, ("H1-indel", pdb)))
                continue
            # case B: same position-set, exactly one site's identity differs
            if (
                possets[i] is not None
                and possets[i] == possets[j]
                and len(symd) == 2
            ):
                pairs.append((i, j, 1, ("H1-swap", pdb)))
    return pairs


# ----------------------------------------------------------------------------
# Metrics on a pair list
# ----------------------------------------------------------------------------
def pair_frame(df, pairs):
    rows = []
    for i, j, d, key in pairs:
        ei, ej = df.at[i, "ddG"], df.at[j, "ddG"]
        pi, pj = df.at[i, "ddG_pred"], df.at[j, "ddG_pred"]
        dexp = ei - ej
        dpred = pi - pj
        rows.append(
            dict(
                i=i, j=j, d=d,
                pdb=df.at[i, "#Pdb"],
                dexp=dexp, dpred=dpred,
                abs_dexp=abs(dexp), abs_dpred=abs(dpred),
                sali=abs(dexp) / d,
            )
        )
    return pd.DataFrame(rows)


def background_within_complex(df, n_per_complex=2000):
    """Random within-complex pairs (any two mutants), as a smoothness baseline:
    if matched (sequence-near) pairs have gaps as large as random pairs, the
    landscape is rugged (cliffy)."""
    rows = []
    for pdb, g in df.groupby("#Pdb"):
        idx = g.index.tolist()
        if len(idx) < 2:
            continue
        allp = list(itertools.combinations(idx, 2))
        if len(allp) > n_per_complex:
            sel = RNG.choice(len(allp), n_per_complex, replace=False)
            allp = [allp[k] for k in sel]
        for i, j in allp:
            rows.append(abs(df.at[i, "ddG"] - df.at[j, "ddG"]))
    return np.array(rows)


def concordance(pf, gap_lo=None, gap_hi=None):
    sub = pf
    if gap_lo is not None:
        sub = sub[sub["abs_dexp"] >= gap_lo]
    if gap_hi is not None:
        sub = sub[sub["abs_dexp"] < gap_hi]
    sub = sub[sub["dexp"].abs() > 1e-9]  # drop exp ties
    if len(sub) == 0:
        return dict(n=0, conc=float("nan"), n_pred_tie=0)
    pred_tie = (sub["dpred"].abs() <= 1e-9).sum()
    nz = sub[sub["dpred"].abs() > 1e-9]
    conc = float((np.sign(nz["dexp"]) == np.sign(nz["dpred"])).mean()) if len(nz) else float("nan")
    return dict(n=int(len(sub)), conc=conc, n_pred_tie=int(pred_tie))


def main():
    df = load()
    out = {}
    out["dataset"] = dict(
        rows=int(len(df)),
        complexes=int(df["#Pdb"].nunique()),
        ddG_std=float(df["ddG"].std()),
        n_single=int((df["nsites"] == 1).sum()),
    )

    single, sps = sps_pairs(df)
    h1 = hamming1_pairs(df)
    pf_sps = pair_frame(df, sps)
    pf_h1 = pair_frame(df, h1)

    out["pair_counts"] = dict(
        sps=len(pf_sps),
        sps_sites=int(single.groupby("site").filter(lambda g: len(g) >= 2)["site"].nunique()),
        h1=len(pf_h1),
    )

    # ---------- Q1: cliff existence ----------
    bg = background_within_complex(df)
    thr = [1, 2, 3, 4]
    def frac_ge(arr, t):
        return float((np.asarray(arr) >= t).mean())

    q1 = {}
    for name, gaps in [("SPS", pf_sps["abs_dexp"].values),
                       ("H1", pf_h1["abs_dexp"].values),
                       ("random_within_complex", bg)]:
        q1[name] = dict(
            n=int(len(gaps)),
            median=float(np.median(gaps)),
            mean=float(np.mean(gaps)),
            p90=float(np.percentile(gaps, 90)),
            max=float(np.max(gaps)),
            frac_ge={str(t): frac_ge(gaps, t) for t in thr},
        )
    # Mann-Whitney: are SPS gaps ~ as large as random within-complex gaps?
    u, p = mannwhitneyu(pf_sps["abs_dexp"], bg, alternative="two-sided")
    q1["sps_vs_random_mwu_p"] = float(p)
    q1["sps_to_random_median_ratio"] = float(np.median(pf_sps["abs_dexp"]) / np.median(bg))

    # per-site span (max-min ddG among substitutions at one position)
    spans = []
    for site, g in single.groupby("site"):
        if len(g) >= 2:
            spans.append(g["ddG"].max() - g["ddG"].min())
    spans = np.array(spans)
    q1["site_span"] = dict(
        n_sites=int(len(spans)),
        median=float(np.median(spans)),
        mean=float(np.mean(spans)),
        frac_ge={str(t): frac_ge(spans, t) for t in thr},
        max=float(spans.max()),
    )
    out["Q1_cliff_existence"] = q1

    # top SPS cliffs (concrete examples)
    top = pf_sps.sort_values("abs_dexp", ascending=False).head(15)
    examples = []
    for _, r in top.iterrows():
        i, j = int(r["i"]), int(r["j"])
        examples.append(dict(
            pdb=df.at[i, "#Pdb"],
            mut_a=df.at[i, "muts"][0], ddg_a=round(df.at[i, "ddG"], 2), pred_a=round(df.at[i, "ddG_pred"], 2),
            mut_b=df.at[j, "muts"][0], ddg_b=round(df.at[j, "ddG"], 2), pred_b=round(df.at[j, "ddG_pred"], 2),
            exp_gap=round(r["dexp"], 2), pred_gap=round(r["dpred"], 2),
            sign_ok=bool(np.sign(r["dexp"]) == np.sign(r["dpred"])),
        ))
    out["Q1_top_sps_cliffs"] = examples

    # ---------- Q2: model handling ----------
    q2 = {}
    for name, pf in [("SPS", pf_sps), ("H1", pf_h1)]:
        d = {}
        # overall pairwise ranking
        d["concordance_all"] = concordance(pf)
        # stratified: smooth (<2) vs cliff (>=2) vs strong cliff (>=3)
        d["concordance_smooth_lt1"] = concordance(pf, gap_hi=1)
        d["concordance_1to2"] = concordance(pf, gap_lo=1, gap_hi=2)
        d["concordance_cliff_ge2"] = concordance(pf, gap_lo=2)
        d["concordance_cliff_ge3"] = concordance(pf, gap_lo=3)
        # signed gap correlation
        sp, _ = spearmanr(pf["dpred"], pf["dexp"])
        pr, _ = pearsonr(pf["dpred"], pf["dexp"])
        d["signed_gap_spearman"] = float(sp)
        d["signed_gap_pearson"] = float(pr)
        # smoothing: regress |dpred| on |dexp|
        x = pf["abs_dexp"].values
        y = pf["abs_dpred"].values
        slope, intercept = np.polyfit(x, y, 1)
        d["abs_gap_slope"] = float(slope)
        d["abs_gap_intercept"] = float(intercept)
        # mean predicted gap for strong cliffs vs experimental
        cliff = pf[pf["abs_dexp"] >= 3]
        d["strong_cliff_mean_exp_gap"] = float(cliff["abs_dexp"].mean()) if len(cliff) else None
        d["strong_cliff_mean_pred_gap"] = float(cliff["abs_dpred"].mean()) if len(cliff) else None
        d["strong_cliff_compression"] = (
            float(cliff["abs_dpred"].mean() / cliff["abs_dexp"].mean()) if len(cliff) else None
        )
        q2[name] = d

    # per-site within-site ranking (sites with >=3 substitutions for a meaningful rho)
    site_rhos = []
    site_rhos_ge4 = []
    for site, g in single.groupby("site"):
        if len(g) >= 3:
            rho, _ = spearmanr(g["ddG_pred"], g["ddG"])
            if not np.isnan(rho):
                site_rhos.append(rho)
                if len(g) >= 4:
                    site_rhos_ge4.append(rho)
    q2["per_site_ranking"] = dict(
        n_sites_ge3=len(site_rhos),
        mean_within_site_spearman=float(np.mean(site_rhos)) if site_rhos else None,
        median_within_site_spearman=float(np.median(site_rhos)) if site_rhos else None,
        n_sites_ge4=len(site_rhos_ge4),
        mean_within_site_spearman_ge4=float(np.mean(site_rhos_ge4)) if site_rhos_ge4 else None,
    )
    out["Q2_model_handling"] = q2

    # ---------- figures ----------
    # Fig 1: gap distributions (SPS vs random)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bins = np.linspace(0, 8, 33)
    ax[0].hist(bg, bins=bins, density=True, alpha=0.5, label="random within-complex", color="#999999")
    ax[0].hist(pf_sps["abs_dexp"], bins=bins, density=True, alpha=0.6, label="SPS matched pairs", color="#d62728")
    ax[0].axvline(2, ls="--", c="k", lw=1); ax[0].axvline(3, ls=":", c="k", lw=1)
    ax[0].set_xlabel("|ΔΔG gap| (kcal/mol)"); ax[0].set_ylabel("density")
    ax[0].set_title("Q1: matched pairs smoother on avg, but heavy cliff tail"); ax[0].legend()
    # Fig: pred gap vs exp gap (signed) for SPS
    ax[1].scatter(pf_sps["dexp"], pf_sps["dpred"], s=12, alpha=0.5, color="#1f77b4")
    lim = max(pf_sps["dexp"].abs().max(), pf_sps["dpred"].abs().max()) * 1.05
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y=x")
    s, b = np.polyfit(pf_sps["dexp"], pf_sps["dpred"], 1)
    xs = np.linspace(-lim, lim, 50); ax[1].plot(xs, s * xs + b, "r-", lw=1.2, label=f"fit slope={s:.2f}")
    ax[1].set_xlabel("experimental Δ(ΔΔG)  (mut_i − mut_j)")
    ax[1].set_ylabel("predicted Δ(ΔΔG)")
    ax[1].set_title("Q2: StaB-ddG smooths cliffs (slope<1)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig_cliff_landscape.png", dpi=130)

    # Fig 2: concordance vs exp-gap bin
    edges = [0, 0.5, 1, 1.5, 2, 3, 4, 100]
    labels = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-4", "≥4"]
    concs, ns = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = concordance(pf_sps, gap_lo=lo, gap_hi=hi)
        concs.append(c["conc"]); ns.append(c["n"])
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    bars = ax2.bar(labels, concs, color="#2ca02c", alpha=0.8)
    ax2.axhline(0.5, ls="--", c="k", lw=1, label="chance")
    ax2.set_ylim(0, 1); ax2.set_xlabel("experimental |ΔΔG gap| bin (kcal/mol)")
    ax2.set_ylabel("pairwise sign concordance (SPS)")
    ax2.set_title("Does ranking accuracy rise on bigger cliffs?")
    for bar, n in zip(bars, ns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}", ha="center", fontsize=8)
    ax2.legend(); fig2.tight_layout(); fig2.savefig(f"{OUTDIR}/fig_concordance_by_gap.png", dpi=130)

    out["Q2_concordance_by_gap_bin"] = [
        dict(bin=lab, conc=(None if np.isnan(c) else round(c, 3)), n=n)
        for lab, c, n in zip(labels, concs, ns)
    ]

    with open(f"{OUTDIR}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
