#!/usr/bin/env python3
"""Megascale (Tsuboyama 2023) dataset analysis for the Megascale-vs-MGnify comparison."""
import re, json, sys
import numpy as np
import pandas as pd

MEGA = "/home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace/data/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"

cols = ["name","deltaG","aa_seq","mut_type","WT_name","WT_cluster","ddG_ML"]
df = pd.read_csv(MEGA, usecols=cols, low_memory=False)
print("rows:", len(df))

def categorize(mt):
    if not isinstance(mt, str): return "other"
    if mt == "wt": return "wt"
    if mt.startswith("ins"): return "insertion"
    if mt.startswith("del"): return "deletion"
    if re.fullmatch(r"[A-Z]\d+[A-Z]", mt): return "single_sub"
    if ":" in mt or "_" in mt: return "multi_sub"
    return "other"

df["cat"] = df["mut_type"].map(categorize)
df["dG"] = pd.to_numeric(df["deltaG"], errors="coerce")
df["ddG"] = pd.to_numeric(df["ddG_ML"], errors="coerce")
df["L"] = df["aa_seq"].str.len()

res = {}
res["n_rows"] = int(len(df))
res["n_WT_domains"] = int(df["WT_name"].nunique())
res["n_WT_clusters"] = int(df["WT_cluster"].nunique())
res["variant_type_counts"] = df["cat"].value_counts().to_dict()

# length distribution of WT domains (one per WT_name, take wt rows)
wt = df[df["cat"]=="wt"].drop_duplicates("WT_name")
res["wt_len"] = {
    "min": int(wt["L"].min()), "max": int(wt["L"].max()),
    "mean": round(float(wt["L"].mean()),1), "median": float(wt["L"].median()),
}
# domain length from all aa_seq (full dataset)
res["all_len"] = {"min": int(df["L"].min()), "max": int(df["L"].max()),
                  "mean": round(float(df["L"].mean()),1), "median": float(df["L"].median())}

# dG distribution (measured, non-NaN)
dG = df["dG"].dropna()
res["dG"] = {"n": int(len(dG)), "min": round(float(dG.min()),2), "max": round(float(dG.max()),2),
             "mean": round(float(dG.mean()),2), "median": round(float(dG.median()),2),
             "p05": round(float(dG.quantile(.05)),2), "p95": round(float(dG.quantile(.95)),2)}

# mutants per WT domain (exclude wt rows)
mut = df[df["cat"]!="wt"]
per_wt = mut.groupby("WT_name").size()
res["variants_per_WT"] = {"mean": round(float(per_wt.mean()),1), "median": float(per_wt.median()),
                          "min": int(per_wt.min()), "max": int(per_wt.max())}

# single-substitution saturation per WT domain
# for each WT_name: distinct single-subs measured / (19 * L)
single = df[df["cat"]=="single_sub"].copy()
# position+target from mut_type
def parse_sub(mt):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mt)
    return (int(m.group(2)), m.group(3)) if m else (None,None)
single[["pos","aa"]] = single["mut_type"].apply(lambda x: pd.Series(parse_sub(x)))
wt_len = wt.set_index("WT_name")["L"].to_dict()
sat_rows=[]
posfrac_rows=[]
for name, g in single.groupby("WT_name"):
    L = wt_len.get(name)
    if not L: continue
    n_distinct = g.drop_duplicates(["pos","aa"]).shape[0]
    sat = n_distinct/(19*L)
    sat_rows.append(sat)
    n_pos = g["pos"].nunique()
    posfrac_rows.append(n_pos/L)
sat_arr = np.array(sat_rows); pf = np.array(posfrac_rows)
res["single_sub_saturation"] = {
    "n_domains_with_singles": int(len(sat_rows)),
    "mean_fraction_of_19L_covered": round(float(sat_arr.mean()),3),
    "median_fraction": round(float(np.median(sat_arr)),3),
    "frac_domains_over_80pct": round(float((sat_arr>0.8).mean()),3),
    "mean_positions_mutated_fraction_of_L": round(float(pf.mean()),3),
    "median_positions_fraction": round(float(np.median(pf)),3),
}
# avg substitutions measured per mutated position (across single subs)
res["avg_subs_per_position"] = round(float(single.groupby(["WT_name","mut_type"]).ngroups / max(1,single.groupby(["WT_name","pos"]).ngroups)),2)

print(json.dumps(res, indent=2, default=str))
json.dump(res, open("/tmp/claude-224072/megascale_stats.json","w"), indent=2, default=str)
# also save per-WT arrays for plotting later
np.save("/tmp/claude-224072/mega_sat.npy", sat_arr)
np.save("/tmp/claude-224072/mega_per_wt.npy", per_wt.values)
np.save("/tmp/claude-224072/mega_dG.npy", dG.values)
np.save("/tmp/claude-224072/mega_wt_len.npy", wt["L"].values)
print("saved megascale_stats.json")
