#!/usr/bin/env python3
"""Reproduce the paper's ESM3ΔG absolute folding-ΔG result on the MGnify Stability
TEST split, to validate our pretrained-ESM3dG deployment.

For each test row (name|split|seq|dG|PDB_name):
  - structure = ESMFold2-Fast prediction  <PDB_name>.cif.gz  (WT scaffold; mutants thread onto it)
  - thread the row's `seq` onto the scaffold structure -> ESM3 forward
  - absolute dG = masked-mean of per-residue stability head; scaled (SigmoidScaling,
    calibrated ~[-1,5] kcal/mol, matches ESM3dG_predict sigmoid_on=True) and raw.
  - ensemble = average over the base members (weights_1/2/3).
Then Spearman / Pearson / RMSE vs the `dG` label, compared to paper (Spearman~0.87, RMSE 0.80).

Usage:
  HF_HOME=<hf_cache> HF_HUB_OFFLINE=1 python eval_mgnify.py \
    --members ../data/esm3dg_weights/ESM3dG_weights_1_lora.ckpt,..._2_...,..._3_... \
    --index_csv ../data/mgnify/mgnify_training_index.csv \
    --struct_dir ../data/mgnify/structures_test --split test --out ...csv
"""
import os, sys, argparse, time
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "esm3dg_stab"))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

from scorer import ESM3dGScorer
from esm3dg_model import parse_CIF, SEQUENCE_VOCAB

_STD_AA = "ACDEFGHIKLMNPQRSTVWY"
_X = SEQUENCE_VOCAB.index("X")
_LUT = {a: SEQUENCE_VOCAB.index(a) for a in _STD_AA}


def thread_seq_to_tokens(enc, aa_str, device):
    """Thread a raw aa string onto the scaffold's [<cls>,res...,<eos>] token layout."""
    wt_tok = enc["seq"].detach().cpu()
    Ltok = wt_tok.shape[0]
    L = len(aa_str)
    assert Ltok == L + 2, f"token len {Ltok} != len(seq)+2 = {L+2} (cls/eos)"
    out = wt_tok.clone()
    out[1:L + 1] = torch.tensor([_LUT.get(a, _X) for a in aa_str], dtype=out.dtype)
    return out.unsqueeze(0).to(device)


def eval_member(ckpt, rows, struct_dir, device):
    """Return dict row_idx -> (pred_scaled, pred_raw) for one member ckpt."""
    scorer = ESM3dGScorer(ckpt, device=device, trainable=False)
    enc_cache = {}
    preds = {}
    t0 = time.time()
    n_skip = 0
    for i, r in enumerate(rows):
        pdb = r["PDB_name"]
        if pdb not in enc_cache:
            path = os.path.join(struct_dir, pdb + ".cif.gz")
            seq, coords = parse_CIF(path, "A")
            if seq is None:
                seq, coords = parse_CIF(path, None)     # fallback: all chains
            if seq is None:
                enc_cache[pdb] = None
            else:
                enc_cache[pdb] = scorer.encode_seq_coords(seq, coords, cache_key=pdb)
        enc = enc_cache[pdb]
        if enc is None or len(r["seq"]) + 2 != enc["seq"].shape[0]:
            n_skip += 1
            continue
        toks = thread_seq_to_tokens(enc, r["seq"], device)
        with torch.no_grad():
            ds = scorer.folding_dG(enc, toks, scaled=True).item()
            dr = scorer.folding_dG(enc, toks, scaled=False).item()
        preds[i] = (ds, dr)
        if (i + 1) % 500 == 0:
            print(f"  [{os.path.basename(ckpt)}] {i+1}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [{os.path.basename(ckpt)}] done {len(preds)}/{len(rows)}  skipped={n_skip}  ({time.time()-t0:.0f}s)", flush=True)
    del scorer
    torch.cuda.empty_cache()
    return preds


def metrics(pred, label):
    pred, label = np.asarray(pred), np.asarray(label)
    sp = spearmanr(pred, label)[0]
    pr = pearsonr(pred, label)[0]
    rmse = float(np.sqrt(np.mean((pred - label) ** 2)))
    # offset-removed RMSE (center both) — calibration-offset-invariant
    rmse_off = float(np.sqrt(np.mean(((pred - pred.mean()) - (label - label.mean())) ** 2)))
    return sp, pr, rmse, rmse_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="comma-sep lora ckpt paths (ensemble)")
    ap.add_argument("--index_csv", default=os.path.join(HERE, "..", "data", "mgnify", "mgnify_training_index.csv"))
    ap.add_argument("--struct_dir", default=os.path.join(HERE, "..", "data", "mgnify", "structures_test"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.index_csv)
    df = df[df.split == args.split].reset_index(drop=True)
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    rows = df.to_dict("records")
    print(f"MGnify {args.split}: {len(rows)} rows  |  unique structures: {df.PDB_name.nunique()}", flush=True)

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    per_member = []
    for ck in members:
        print(f"=== member {ck} ===", flush=True)
        per_member.append(eval_member(ck, rows, args.struct_dir, dev))

    # ensemble average over members where ALL members produced a pred
    idx = sorted(set.intersection(*[set(p.keys()) for p in per_member])) if per_member else []
    ens_scaled = [float(np.mean([per_member[m][i][0] for m in range(len(members))])) for i in idx]
    ens_raw = [float(np.mean([per_member[m][i][1] for m in range(len(members))])) for i in idx]
    label = [float(rows[i]["dG"]) for i in idx]
    print(f"\n=== MGnify {args.split} eval  (n={len(idx)}, ensemble of {len(members)}) ===", flush=True)

    for tag, pred in (("scaled(calibrated)", ens_scaled), ("raw", ens_raw)):
        sp, pr, rmse, rmse_off = metrics(pred, label)
        print(f"  [{tag:18s}] Spearman={sp:.4f}  Pearson={pr:.4f}  RMSE={rmse:.4f}  RMSE(offset-removed)={rmse_off:.4f}", flush=True)
    # per-member (scaled) for reference
    for m, ck in enumerate(members):
        ps = [per_member[m][i][0] for i in idx]
        sp = spearmanr(ps, label)[0]
        print(f"  [member {os.path.basename(ck)}] scaled Spearman={sp:.4f}", flush=True)
    print("  (paper ESM3ΔG on MGnify test 3283: Spearman~0.87 / RMSE 0.80 kcal/mol)", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out = pd.DataFrame({
            "name": [rows[i]["name"] for i in idx],
            "PDB_name": [rows[i]["PDB_name"] for i in idx],
            "label_dG": label, "pred_scaled": ens_scaled, "pred_raw": ens_raw,
        })
        out.to_csv(args.out, index=False)
        print(f"wrote predictions -> {args.out}", flush=True)
    print("===== DONE =====", flush=True)


if __name__ == "__main__":
    main()
