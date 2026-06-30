#!/usr/bin/env python3
"""SKEMPI → ESM3 adapter for the ESM3ΔG×SKEMPI task.

Reuses StaB's own SKEMPIDataset (parses filtered_skempi.csv, splits #Pdb into
complex/binder1/binder2, builds validated per-structure mutant sequences + ddG).
For each structure we re-encode it for ESM3 by feeding StaB's exact concatenated
sequence + SKEMPI backbone coords (atom37) into the ESM3 encoder — this guarantees
the ESM3 WT sequence == StaB sequence (so StaB's mutant positions map 1:1 to ESM3
token positions p→p+1), and keeps the structure source identical to the ProteinMPNN
baseline.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # worktree root (has stabddg/)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from stabddg.ppi_dataset import SKEMPIDataset           # StaB's own loader (reuse OK)
from esm3dg_model import SEQUENCE_VOCAB, atom_order

STAB_ALPH = "ACDEFGHIKLMNPQRSTVWYX"
# lookup: StaB alphabet index -> ESM3 sequence token id (SEQUENCE_VOCAB index)
_LUT = torch.tensor([SEQUENCE_VOCAB.index(aa) for aa in STAB_ALPH], dtype=torch.long)


def struct_to_seq_coords(sd):
    """StaB parse_PDB dict -> (concat seq str, atom37 coords [L,37,3]) in masked_list order."""
    chains = sd["masked_list"]
    seq = ""
    coords = []
    for ch in chains:
        cs = sd[f"seq_chain_{ch}"]
        cc = sd[f"coords_chain_{ch}"]
        N = np.asarray(cc[f"N_chain_{ch}"]); CA = np.asarray(cc[f"CA_chain_{ch}"])
        C = np.asarray(cc[f"C_chain_{ch}"]); O = np.asarray(cc[f"O_chain_{ch}"])
        seq += cs
        for i in range(len(cs)):
            a = np.full((37, 3), np.nan, dtype=np.float32)
            a[atom_order["N"]] = N[i]; a[atom_order["CA"]] = CA[i]
            a[atom_order["C"]] = C[i]; a[atom_order["O"]] = O[i]
            coords.append(a)
    return seq, np.stack(coords)


def stab_mut_to_esm_tokens(wt_enc, stab_mut_seqs):
    """stab_mut_seqs [n,L] (int over STAB_ALPH) -> ESM3 seq tokens [n, L+2], threaded
    onto the WT token layout ([<cls>, residues, <eos>]). Asserts WT alignment."""
    wt_tok = wt_enc["seq"].detach().cpu()
    Ltok = wt_tok.shape[0]
    L = stab_mut_seqs.shape[1]
    assert Ltok == L + 2, f"ESM token len {Ltok} != L+2={L+2} (cls+L+eos) — encoding misaligned"
    # verify WT alignment: ESM WT residue tokens must equal StaB WT seq encoded
    wt_aa = wt_enc["wt_aa"]
    wt_ids = torch.tensor([SEQUENCE_VOCAB.index(a if a in STAB_ALPH else "X") for a in wt_aa],
                          dtype=torch.long)
    assert torch.equal(wt_ids, wt_tok[1:L + 1]), "ESM WT tokens != StaB WT seq — alignment broken"
    esm_res = _LUT[stab_mut_seqs.long()]                  # [n, L]
    out = wt_tok.unsqueeze(0).repeat(stab_mut_seqs.shape[0], 1).clone()
    out[:, 1:L + 1] = esm_res
    return out


def build_skempi(scorer, csv_path, split_path, pdb_dir, pdb_dict_cache_path, limit=None):
    ds = SKEMPIDataset(csv_path=csv_path, split_path=split_path, pdb_dir=pdb_dir,
                       pdb_dict_cache_path=pdb_dict_cache_path)
    items = []
    for d in ds:
        if limit is not None and len(items) >= limit:
            break
        encs = {}
        ok = True
        for role in ("complex", "binder1", "binder2"):
            sd = d[role]
            seq, coords = struct_to_seq_coords(sd)
            encs[role] = scorer.encode_seq_coords(seq, coords, cache_key=sd["name"])
        try:
            cm = stab_mut_to_esm_tokens(encs["complex"], d["complex_mut_seqs"])
            b1 = stab_mut_to_esm_tokens(encs["binder1"], d["binder1_mut_seqs"])
            b2 = stab_mut_to_esm_tokens(encs["binder2"], d["binder2_mut_seqs"])
        except AssertionError as e:
            print(f"  [skip {d['name']}] {e}")
            ok = False
        if not ok:
            continue
        items.append({
            "name": d["name"],
            "complex": encs["complex"], "binder1": encs["binder1"], "binder2": encs["binder2"],
            "complex_mut": cm, "binder1_mut": b1, "binder2_mut": b2,
            "ddG": d["ddG"].float(),
        })
    print(f"build_skempi: {len(items)} complexes encoded for ESM3.")
    return items
