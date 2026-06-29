#!/usr/bin/env python3
"""MGnify Stability dataset analysis."""
import json, re
import numpy as np
import pandas as pd

CONCAT="/home/guoj0f/repos/absolute-stability-predictor/data/230515_K50dG_dmsv4_dmsv5_dmsv7_concat260429.csv"
IDX="/home/guoj0f/repos/absolute-stability-predictor/data/mgnify_training_index.csv"

res={}

# ---- 1. full concat measurement table ----
cols=["name","deltaG","aa_seq","lib","mgnify","dm_design","split"]
df=pd.read_csv(CONCAT, usecols=cols, low_memory=False)
res["concat_n_rows"]=int(len(df))
df["dG"]=pd.to_numeric(df["deltaG"],errors="coerce")
df["L"]=df["aa_seq"].str.len()
res["concat_n_unique_aa_seq"]=int(df["aa_seq"].nunique())
res["lib_counts"]={str(k):int(v) for k,v in df["lib"].value_counts(dropna=False).items()}
res["mgnify_flag"]={str(k):int(v) for k,v in df["mgnify"].value_counts(dropna=False).items()}
res["dm_design_flag"]={str(k):int(v) for k,v in df["dm_design"].value_counts(dropna=False).items()}
res["split_counts"]={str(k):int(v) for k,v in df["split"].value_counts(dropna=False).items()}
# name prefixes
df["prefix"]=df["name"].astype(str).str.replace(r"\d+$","",regex=True)
res["name_prefixes_top"]={str(k):int(v) for k,v in df["prefix"].value_counts().head(12).items()}

# indel scanning depth: group indel variants by their base (WT) domain id
nm=df["name"].astype(str)
ind=nm[nm.str.contains(r"_ins|_del",regex=True)]
ibase=ind.str.replace(r"_(ins|del)[A-Z]\d+$","",regex=True)
gi=ibase.value_counts()
res["indel_scan"]={"n_base_domains_with_indels":int(gi.size),
                   "total_indel_measurements":int(len(ind)),
                   "indels_per_scanned_domain_mean":round(float(gi.mean()),1),
                   "indels_per_scanned_domain_median":float(gi.median()),
                   "min":int(gi.min()),"max":int(gi.max())}
np.save("/tmp/claude-224072/mgnify_indel_per_base.npy", gi.values)
L=df["L"].dropna()
res["len"]={"min":int(L.min()),"max":int(L.max()),"mean":round(float(L.mean()),1),"median":float(L.median()),
            "p05":float(L.quantile(.05)),"p95":float(L.quantile(.95))}
dG=df["dG"].dropna()
res["dG"]={"n":int(len(dG)),"min":round(float(dG.min()),2),"max":round(float(dG.max()),2),
           "mean":round(float(dG.mean()),2),"median":round(float(dG.median()),2),
           "p05":round(float(dG.quantile(.05)),2),"p95":round(float(dG.quantile(.95)),2),
           "frac_negative":round(float((dG<0).mean()),3)}
np.save("/tmp/claude-224072/mgnify_dG.npy", dG.values)
np.save("/tmp/claude-224072/mgnify_len.npy", L.values)

# ---- 2. training index: PDB_name grouping = per-WT depth ----
idx=pd.read_csv(IDX, low_memory=False)
res["idx_n_rows"]=int(len(idx))
res["idx_cols"]=list(idx.columns)
res["idx_n_unique_PDB_name"]=int(idx["PDB_name"].nunique())
res["idx_n_unique_seq"]=int(idx["seq"].nunique())
res["idx_split_counts"]={str(k):int(v) for k,v in idx["split"].value_counts(dropna=False).items()}
# sequences per PDB_name (depth proxy)
per=idx.groupby("PDB_name").size()
res["seqs_per_PDB_name"]={"mean":round(float(per.mean()),2),"median":float(per.median()),
                          "min":int(per.min()),"max":int(per.max()),
                          "frac_singletons":round(float((per==1).mean()),3),
                          "frac_le2":round(float((per<=2).mean()),3),
                          "frac_ge100":round(float((per>=100).mean()),3),
                          "frac_ge500":round(float((per>=500).mean()),3)}
np.save("/tmp/claude-224072/mgnify_per_pdb.npy", per.values)
# exact per-WT depth histogram (how many measurements each WT domain has)
vc=per.value_counts().sort_index()
res["per_WT_depth_hist"]={int(k):int(v) for k,v in vc.items()}
res["frac_only_WT"]=round(float((per==1).mean()),3)
res["n_WT_with_any_mutant"]=int((per>=2).sum())
# is PDB_name a WT-domain id or per-sequence id? sample a high-count group
top=per.sort_values(ascending=False)
res["top_PDB_name_counts"]={str(k):int(v) for k,v in top.head(8).items()}
# look at name patterns within a big group to infer WT/mutant structure
big=top.index[0]
g=idx[idx["PDB_name"]==big]
res["example_big_group"]={"PDB_name":str(big),"n":int(len(g)),
    "name_examples":g["name"].head(6).tolist(),
    "seq_len_range":[int(g["seq"].str.len().min()),int(g["seq"].str.len().max())],
    "n_unique_seq":int(g["seq"].nunique())}

print(json.dumps(res,indent=2,default=str))
json.dump(res, open("/tmp/claude-224072/mgnify_stats.json","w"), indent=2, default=str)
print("saved mgnify_stats.json")
