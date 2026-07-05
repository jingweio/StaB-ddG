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


def struct_to_seq_coords(sd, chainbreak=False):
    """StaB parse_PDB dict -> (seq str, atom37 coords [L,37,3]) in masked_list order.
    chainbreak=False (default): naive concat (variant a) — unchanged behavior.
    chainbreak=True (variant b): join chains with '|' + ONE all-NaN atom37 row per '|'
      (== ESM3 ProteinComplex.from_chains layout; '|' -> SEQUENCE_CHAINBREAK_TOKEN=31 at encode,
       STRUCTURE_CHAINBREAK_TOKEN applied inside ESM3.forward; NaN row safely masked by build_affine3d)."""
    chains = sd["masked_list"]
    seq_parts, coord_parts = [], []
    for ch in chains:
        cs = sd[f"seq_chain_{ch}"]
        cc = sd[f"coords_chain_{ch}"]
        N = np.asarray(cc[f"N_chain_{ch}"]); CA = np.asarray(cc[f"CA_chain_{ch}"])
        C = np.asarray(cc[f"C_chain_{ch}"]); O = np.asarray(cc[f"O_chain_{ch}"])
        a = np.full((len(cs), 37, 3), np.nan, dtype=np.float32)
        a[:, atom_order["N"]] = N; a[:, atom_order["CA"]] = CA
        a[:, atom_order["C"]] = C; a[:, atom_order["O"]] = O
        seq_parts.append(cs); coord_parts.append(a)
    if chainbreak and len(seq_parts) > 1:
        sep = np.full((1, 37, 3), np.nan, dtype=np.float32)   # ESM chainbreak separator row
        seq = "|".join(seq_parts)
        merged = []
        for i, a in enumerate(coord_parts):
            if i > 0:
                merged.append(sep)
            merged.append(a)
        coords = np.concatenate(merged, 0)                    # len == len(seq) (1 char + 1 row per '|')
    else:
        seq = "".join(seq_parts)
        coords = np.concatenate(coord_parts, 0)
    return seq, coords


def stab_mut_to_esm_tokens(wt_enc, stab_mut_seqs):
    """stab_mut_seqs [n,P] (int over STAB_ALPH; P = REAL residues, no pipes) -> ESM3 seq tokens
    [n, Ltok] threaded onto the WT token layout. PIPE-AWARE: with chainbreak the WT layout is
    [<cls>, res/|.., <eos>], so thread ONLY into real-residue positions and keep '|' (token 31)
    fixed. For pipe-free (variant a / Megascale / MGnify) this is identical to the old
    out[:,1:P+1] behavior. Asserts WT alignment on the real residues."""
    wt_tok = wt_enc["seq"].detach().cpu()                 # [Ltok], may contain '|'=31, incl <cls>/<eos>
    wt_aa = wt_enc["wt_aa"]                                # str, may contain '|'
    n, P = stab_mut_seqs.shape                             # P = real residues (StaB seq, no pipes)
    # token positions of real residues (skip <cls>@0, each '|', <eos>). char i -> token i+1.
    res_pos = torch.tensor([i + 1 for i, a in enumerate(wt_aa) if a != "|"], dtype=torch.long)
    assert len(res_pos) == P, f"{len(res_pos)} residue slots != P={P} (pipe-aware misalign)"
    wt_ids = torch.tensor([SEQUENCE_VOCAB.index(a if a in STAB_ALPH else "X")
                           for a in wt_aa if a != "|"], dtype=torch.long)
    assert torch.equal(wt_ids, wt_tok[res_pos]), "ESM WT tokens != StaB WT seq — alignment broken"
    esm_res = _LUT[stab_mut_seqs.long()]                  # [n, P]
    out = wt_tok.unsqueeze(0).repeat(n, 1).clone()
    out[:, res_pos] = esm_res                             # only real residues; '|'(31) kept fixed
    return out


def build_skempi(scorer, csv_path, split_path, pdb_dir, pdb_dict_cache_path, limit=None, chainbreak=False):
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
            seq, coords = struct_to_seq_coords(sd, chainbreak=chainbreak)
            # cache_key includes chainbreak flag so concat vs chainbreak encodings never collide
            encs[role] = scorer.encode_seq_coords(seq, coords, cache_key=f"{sd['name']}|cb{int(chainbreak)}")
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
