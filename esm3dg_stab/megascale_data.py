#!/usr/bin/env python3
"""Megascale (Tsuboyama) → ESM3 adapter for stage-1 folding-ΔΔG fine-tuning (task2).

Mirrors StaB's stability_finetune.py data construction: per WT_name, collect the
substitution mutants' aa_seq + ddG_ML (indels excluded), threaded onto the domain's
AlphaFold2 structure. Each domain's WT structure is encoded once for ESM3 (from its
seq + backbone coords, Strategy 1 — same as the SKEMPI adapter)."""
import os, sys, pickle
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from stabddg.mpnn_utils import parse_PDB
from esm3dg_model import SEQUENCE_VOCAB
from skempi_data import struct_to_seq_coords, STAB_ALPH
import pandas as pd

_VOCAB_IDX = {a: SEQUENCE_VOCAB.index(a) for a in STAB_ALPH}
def _aa_id(a): return _VOCAB_IDX.get(a, SEQUENCE_VOCAB.index("X"))


def aa_strs_to_esm_tokens(wt_enc, aa_seqs):
    wt_tok = wt_enc["seq"].detach().cpu()
    Ltok = wt_tok.shape[0]; L = Ltok - 2
    ids = torch.tensor([[_aa_id(a) for a in s] for s in aa_seqs], dtype=torch.long)
    assert ids.shape[1] == L, f"mut seq len {ids.shape[1]} != struct L {L}"
    out = wt_tok.unsqueeze(0).repeat(len(aa_seqs), 1).clone()
    out[:, 1:L + 1] = ids
    return out


def build_megascale(scorer, data_dir, split="train", limit=None,
                    splits_pkl=None, stability_csv=None, pdb_dir=None):
    splits_pkl = splits_pkl or os.path.join(data_dir, "rocklin", "mega_splits.pkl")
    stability_csv = stability_csv or os.path.join(
        data_dir, "megascale", "Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
    pdb_dir = pdb_dir or os.path.join(data_dir, "megascale", "AlphaFold_model_PDBs")

    with open(splits_pkl, "rb") as f:
        splits = pickle.load(f)
    names = splits[split]
    names = names.tolist() if hasattr(names, "tolist") else list(names)

    # only load the columns we need — the full Tsuboyama CSV (697 MB × 38 cols) otherwise
    # balloons CPU RAM and gets OOM-killed; these 4 cols suffice.
    df = pd.read_csv(stability_csv, low_memory=False,
                     usecols=["aa_seq", "ddG_ML", "mut_type", "WT_name"])
    df = df[df["ddG_ML"] != "-"]
    df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)
    grouped = {k: v for k, v in df[df.mut_type != "wt"].groupby("WT_name")}

    items = []
    for name in names:
        if limit is not None and len(items) >= limit:
            break
        if name not in grouped:
            continue
        clean = name.split(".pdb", 1)[0] + ".pdb"
        clean = clean.replace("|", ":")
        path = os.path.join(pdb_dir, clean)
        if not os.path.exists(path):
            continue
        sd = parse_PDB(path)[0]
        sd["masked_list"] = [k.split("_")[-1] for k in sd if k.startswith("seq_chain_")]
        seq, coords = struct_to_seq_coords(sd)
        enc = scorer.encode_seq_coords(seq, coords, cache_key=clean)
        mut_df = grouped[name]
        aa_seqs = mut_df["aa_seq"].tolist()
        ddG = mut_df["ddG_ML"].to_numpy(dtype=np.float32)
        try:
            mut = aa_strs_to_esm_tokens(enc, aa_seqs)
        except AssertionError as e:
            print(f"  [skip {name}] {e}"); continue
        items.append({"name": name, "enc": enc, "mut": mut,
                      "ddG": torch.from_numpy(ddG)})
    print(f"build_megascale[{split}]: {len(items)} domains encoded for ESM3.")
    return items
