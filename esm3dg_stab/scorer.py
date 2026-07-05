#!/usr/bin/env python3
"""ESM3dGScorer — wraps ESM3ΔG (ESM3 + LoRA + stability head) so it produces a
folding ΔG for sequences threaded onto a FIXED backbone, and the StaB-ddG
folding/binding ΔΔG decomposition on top.

Design (faithful fine-tune of ESM3ΔG):
  folding_dG(wt_enc, seq_tokens[B,L]) = mean over non-special residues of the
    per-residue stability head output, with the structure tokens + coords held
    fixed at the WT backbone (mutants are threaded as sequence only).
  folding_ddG = dG(mut) - dG(wt)
  binding_ddG = folding_ddG(complex) - folding_ddG(binder1) - folding_ddG(binder2)   [StaB decomposition]

Only the LoRA adapters + stability head + output scaling are trainable; the ESM3
trunk stays frozen (matches ESM3ΔG; per user, LoRA not full-FT).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from esm3dg_model import ESM3dG, tied_featurize, get_esm3_input_info_direct


def is_adapter_name(name):
    """The ONLY trainable params: LoRA adapters + stability head + output scaling.
    Everything else (ESM3 trunk AND the lazily-loaded VQ-VAE structure encoder) is frozen."""
    nl = name.lower()
    return ("lora" in nl) or ("stability_head" in name) or ("output_scaling" in name)


class ESM3dGScorer:
    def __init__(self, lora_ckpt, device="cuda", trainable=False):
        self.device = device
        # build ESM3ΔG; freeze_weights=False so grads CAN flow, then we選擇性解冻
        class _Cfg:
            training = type("o", (), {"rank": 4, "dropout": 0.15})()
            model = type("o", (), {"freeze_weights": False})()
            testing = type("o", (), {"ddg_scanning": False})()
        self.model = ESM3dG(lora_ckpt, cfg=_Cfg())   # TransferModel
        self.base = self.model.esm3_stability_model.base_model  # raw ESM3 (for encoding)
        self._enc_cache = {}
        self.set_trainable(trainable)

    # ---- trainable-param control: LoRA + head + scaling only -------------------
    # NOTE: the VQ-VAE structure encoder loads lazily on the first encode() (after
    # __init__), so call set_trainable() AGAIN after structures are encoded to make
    # sure it gets frozen. trainable_parameters()/adapter_state_dict() are name-based
    # (authoritative) so the optimizer/checkpoint never touch the trunk or encoder.
    def set_trainable(self, trainable):
        n_train = 0
        for name, p in self.model.named_parameters():
            p.requires_grad = bool(trainable and is_adapter_name(name))
            if p.requires_grad:
                n_train += p.numel()
        self.model.train() if trainable else self.model.eval()
        return n_train

    def trainable_parameters(self):
        return [p for n, p in self.model.named_parameters() if is_adapter_name(n)]

    def adapter_state_dict(self):
        return {n: p.detach().cpu() for n, p in self.model.named_parameters() if is_adapter_name(n)}

    # ---- structure encoding (cached per (pdb,chain)) ---------------------------
    def encode(self, pdb_path, chain_id):
        key = (pdb_path, tuple(chain_id) if isinstance(chain_id, (list, tuple)) else chain_id)
        if key not in self._enc_cache:
            info, seq = get_esm3_input_info_direct(pdb_path, chain_id, self.base)
            if info is None:
                raise ValueError(f"failed to encode {pdb_path} chain={chain_id}")
            # info: {seq, struct, coord} (ESM3 token tensors, incl <cls>/<eos>)
            self._enc_cache[key] = {"seq": info["seq"].detach(),
                                    "struct": info["struct"].detach(),
                                    "coord": info["coord"].detach(),
                                    "wt_aa": seq}
        return self._enc_cache[key]

    def encode_seq_coords(self, seq, coords37, cache_key=None):
        """Encode an ESM3 input from an explicit sequence + atom37 coords (Strategy 1:
        feed StaB's seq + SKEMPI backbone coords so the WT seq is guaranteed to match)."""
        if cache_key is not None and cache_key in self._enc_cache:
            return self._enc_cache[cache_key]
        from esm.sdk.api import ESMProtein
        coords = torch.as_tensor(coords37, dtype=torch.float32)
        prot = ESMProtein(sequence=seq, coordinates=coords)
        self.base.eval()
        with torch.no_grad():
            enc = self.base.encode(prot)
        out = {"seq": enc.sequence.detach(), "struct": enc.structure.detach(),
               "coord": enc.coordinates.detach(), "wt_aa": seq}
        if cache_key is not None:
            self._enc_cache[cache_key] = out
        return out

    # ---- ΔG of a batch of sequences on a fixed backbone ------------------------
    def folding_dG(self, enc, seq_tokens, scaled=False):
        """seq_tokens: LongTensor [B, L] of ESM3 sequence-token ids (L incl special toks,
        same length as enc['seq']). Returns dG [B] = masked-mean per-residue stability.
        scaled=False -> raw head output (used for ΔΔG, default); scaled=True -> per-residue
        SigmoidScaling applied first (calibrated absolute ΔG in ~[-1,5] kcal/mol, for the
        MGnify absolute-dG reproduction — matches ESM3dG_predict sigmoid_on=True)."""
        B = seq_tokens.shape[0]
        batch = [{"seq": seq_tokens[i], "struct": enc["struct"], "coord": enc["coord"]}
                 for i in range(B)]
        self.model.ddg_scanning = False
        dg, scaled_dg, mask = self.model(batch)          # dg,scaled_dg:[B,L] per-residue, mask:[B,L]
        vals = scaled_dg if scaled else dg
        mask = mask.to(vals.dtype)
        valid = mask.sum(dim=-1).clamp_min(1.0)
        dG = (vals * mask).sum(dim=-1) / valid           # [B] masked mean
        return dG

    def folding_dG_both(self, enc, seq_tokens):
        """Return (raw_dG, scaled_dG) [B] from ONE forward — masked-mean of dg and scaled_dg.
        Used by the Megascale eval to report both口径 without doubling forwards."""
        B = seq_tokens.shape[0]
        batch = [{"seq": seq_tokens[i], "struct": enc["struct"], "coord": enc["coord"]}
                 for i in range(B)]
        self.model.ddg_scanning = False
        dg, scaled_dg, mask = self.model(batch)
        m = mask.to(dg.dtype)
        valid = m.sum(dim=-1).clamp_min(1.0)
        raw = (dg * m).sum(dim=-1) / valid
        scl = (scaled_dg * m).sum(dim=-1) / valid
        return raw, scl

    def folding_ddG(self, enc, mut_seq_tokens, wt_seq_tokens=None):
        wt = enc["seq"].unsqueeze(0) if wt_seq_tokens is None else wt_seq_tokens
        wt_dG = self.folding_dG(enc, wt.to(self.device))
        mut_dG = self.folding_dG(enc, mut_seq_tokens.to(self.device))
        return mut_dG - wt_dG                            # broadcast [B] - [1]

    def binding_ddG(self, complex_enc, b1_enc, b2_enc,
                    complex_mut, b1_mut, b2_mut):
        c = self.folding_ddG(complex_enc, complex_mut)
        a = self.folding_ddG(b1_enc, b1_mut)
        b = self.folding_ddG(b2_enc, b2_mut)
        return c - (a + b)
