#!/usr/bin/env python
"""Audit multi-chain mutations in the SKEMPIv2 data actually consumed by StaB-ddG.

Question: do any samples mutate MORE THAN ONE CHAIN at once, and if so, do those
mutations stay on one side of the interface or straddle both binders?

Why the distinction matters
---------------------------
StaB-ddG computes the binding ddG by decomposing it into three folding terms
(stabddg/model.py: binding_ddG):

    ddG_bind = ddG_fold(complex) - [ ddG_fold(binder1) + ddG_fold(binder2) ]

`#Pdb` encodes the two binders as chain groups, e.g. `1AO7_ABC_DE` means binder1
= chains ABC, binder2 = chains DE. So a multi-chain mutant is either
  * multi-chain WITHIN one binder  -> only that binder's term carries them, or
  * straddling BOTH binders        -> binder1 and binder2 each carry a subset.
The second case exercises all three dG terms simultaneously and is the
interesting one.

Mutation token format: {WT}{Chain}{Pos}{MUT}, e.g. 'YA434A' = Tyr->Ala, chain A,
pos 434 (see stabddg/ppi_dataset.py: mutations_to_seq).

Usage
-----
    python reproduce/multichain-mutation-note/analyze_multichain_mutations.py
    # add --verify to also empirically check the per-binder mutation split
    # (needs the `stabddg` env + data/SKEMPI2_PDBs)
"""
import os
import sys
import json
import argparse
import pickle

import pandas as pd

# reproduce/multichain-mutation-note/<this file>  ->  repo root is 3 levels up
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.dirname(os.path.abspath(__file__))

SKEMPI_CSV = os.path.join(REPO, "data/SKEMPI/filtered_skempi.csv")
TRAIN_PKL = os.path.join(REPO, "data/SKEMPI/train_pdb.pkl")
TEST_PKL = os.path.join(REPO, "data/SKEMPI/test_pdb.pkl")


def classify(csv_path=SKEMPI_CSV, train_pkl=TRAIN_PKL, test_pkl=TEST_PKL):
    """One row per SKEMPI measurement, annotated with chain/side/split info."""
    df = pd.read_csv(csv_path)
    train = pickle.load(open(train_pkl, "rb"))
    test = pickle.load(open(test_pkl, "rb"))

    rows = []
    for _, r in df.iterrows():
        pdb = r["#Pdb"]
        *_base, b1, b2 = pdb.split("_")
        muts = str(r["Mutation(s)_cleaned"]).split(",")
        chains = sorted({m[1] for m in muts if len(m) >= 3})

        sides, unknown = set(), []
        for c in chains:
            if c in b1:
                sides.add("B1")
            elif c in b2:
                sides.add("B2")
            else:
                unknown.append(c)  # mutation on a chain outside the interface def

        rows.append(dict(
            pdb=pdb, b1=b1, b2=b2,
            n_mut=len(muts), n_chain=len(chains),
            chains="".join(chains), sides="".join(sorted(sides)),
            unknown="".join(unknown),
            muts=r["Mutation(s)_cleaned"], ddG=r["ddG"],
            split="train" if pdb in train else ("test" if pdb in test else "other"),
        ))
    return pd.DataFrame(rows)


def report(d):
    out = {}
    n = len(d)
    print(f"filtered_skempi.csv rows (what the model consumes): {n}")
    print(f"distinct interfaces (#Pdb): {d.pdb.nunique()}")

    print("\n=== 1) distinct chains mutated per row ===")
    dist = d.n_chain.value_counts().sort_index()
    print(dist.to_string())
    multi = d[d.n_chain >= 2]
    print(f"\nrows mutating >=2 chains: {len(multi)} / {n} = {len(multi)/n*100:.1f}%")
    out["n_rows"] = n
    out["chain_count_distribution"] = {int(k): int(v) for k, v in dist.items()}
    out["n_multichain"] = len(multi)

    print("\n=== 2) which side of the interface ===")
    side_counts = d.groupby("sides").size().sort_values(ascending=False)
    print(side_counts.to_string())
    cross = d[d.sides == "B1B2"]
    within = multi[multi.sides != "B1B2"]
    print(f"\nstraddling BOTH binders (B1B2): {len(cross)} = {len(cross)/n*100:.1f}%")
    print(f"multi-chain but within ONE binder: {len(within)}")
    out["side_counts"] = {k: int(v) for k, v in side_counts.items()}
    out["n_cross_interface"] = len(cross)
    out["n_multichain_within_one_binder"] = len(within)

    print("\n=== 3) sanity: mutations on chains outside b1 U b2 ===")
    unk = d[d.unknown != ""]
    print(f"rows: {len(unk)}" + ("" if len(unk) else "  (none -> clean)"))
    if len(unk):
        print(unk[["pdb", "muts", "unknown"]].head(10).to_string(index=False))
    out["n_unknown_chain_rows"] = len(unk)

    print("\n=== 4) train / test breakdown ===")
    split_stats = {}
    for sp in ["train", "test", "other"]:
        s = d[d.split == sp]
        if not len(s):
            continue
        mc, cx = s[s.n_chain >= 2], s[s.sides == "B1B2"]
        print(f"{sp:6s} rows {len(s):5d} | multi-chain {len(mc):4d} ({len(mc)/len(s)*100:5.1f}%)"
              f" | cross-interface {len(cx):4d} ({len(cx)/len(s)*100:5.1f}%)")
        split_stats[sp] = dict(rows=len(s), multichain=len(mc), cross=len(cx),
                               multichain_pct=round(len(mc)/len(s)*100, 2),
                               cross_pct=round(len(cx)/len(s)*100, 2))
    out["split_stats"] = split_stats

    print("\n=== 5) cross-interface examples ===")
    print(cross[["pdb", "n_mut", "chains", "muts", "ddG"]].head(12).to_string(index=False))

    print("\n=== 6) rows touching the most chains ===")
    print(d.nlargest(4, "n_chain")[["pdb", "n_mut", "n_chain", "chains", "sides", "muts"]]
          .to_string(index=False))

    print("\n=== 7) complexes contributing cross-interface rows ===")
    print(f"{cross.pdb.nunique()} complexes / {d.pdb.nunique()} total")
    top = cross.pdb.value_counts().head(10)
    print(top.to_string())
    out["cross_interface_complexes"] = int(cross.pdb.nunique())
    out["cross_interface_top_complexes"] = {k: int(v) for k, v in top.items()}
    return out


def verify_split(pdb="1A4Y_A_B", mutation="RB5A,YA434A"):
    """Empirically confirm each binder receives only its own side's mutations."""
    import tempfile
    if REPO not in sys.path:            # allow running this file by path
        sys.path.insert(0, REPO)
    from stabddg.ppi_dataset import SKEMPIDataset

    df = pd.read_csv(SKEMPI_CSV)
    sub = df[(df["#Pdb"] == pdb) & (df["Mutation(s)_cleaned"] == mutation)]
    assert len(sub), f"{pdb} / {mutation} not found in filtered_skempi.csv"

    tmp = tempfile.mkdtemp()
    one = os.path.join(tmp, "one.csv")
    sub.to_csv(one, index=False)

    ds = SKEMPIDataset(csv_path=one, pdb_dir=os.path.join(REPO, "data/SKEMPI2_PDBs"),
                       pdb_dict_cache_path="")
    s = ds[0]
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"

    def changed(struct, mut_seqs):
        wt = struct["seq"]
        mut = "".join(alphabet[i] for i in mut_seqs[0])
        return [(i + 1, wt[i], mut[i]) for i in range(len(wt)) if wt[i] != mut[i]]

    print(f"\n=== verify: {pdb}  {mutation} ===")
    print("complex :", changed(s["complex"], s["complex_mut_seqs"].numpy()))
    print("binder1 :", changed(s["binder1"], s["binder1_mut_seqs"].numpy()))
    print("binder2 :", changed(s["binder2"], s["binder2_mut_seqs"].numpy()))
    print("lengths : b1=%d b2=%d complex=%d" % (
        len(s["binder1"]["seq"]), len(s["binder2"]["seq"]), len(s["complex"]["seq"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="also run the per-binder split check (needs torch + PDBs)")
    ap.add_argument("--json", default=os.path.join(OUTDIR, "results.json"))
    args = ap.parse_args()

    d = classify()
    stats = report(d)

    with open(args.json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {args.json}")

    if args.verify:
        verify_split()
