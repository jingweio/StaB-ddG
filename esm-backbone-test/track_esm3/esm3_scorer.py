"""
esm3_scorer.py — structure-conditioned ESM3 sequence scorer for StaB-ddG.

``ESM3Scorer`` is the inverse-folding analogue of ProteinMPNN's ``MPNNScorer``:
it holds a FIXED structure (per domain, via the Task-4 ``ESM3StructTokenizer``)
and scores varying sequences against it,

    dG(domain, seqs) = sum_i  log P(s_i | structure)

where the sum runs over the real-residue positions only (BOS / EOS / chainbreak
positions are excluded).  ``folding_ddG = dG(mut) - dG(wt)`` comes from the
``SequenceScorer`` base class.

Sequence/structure length alignment
-----------------------------------
The structure tokens from ``ESM3StructTokenizer`` have length

    T = (sum of chain lengths) + (k - 1) + 2

for a k-chain domain: ``k-1`` inter-chain CHAINBREAK positions plus BOS and EOS.
``insert_chainbreaks`` with ``chain_lengths(domain)`` produces exactly ``k-1``
``'|'`` separators, so after ``tokenize_sequence(..., add_special_tokens=True)``
the sequence-token length equals ``T``, with the ``'|'`` positions aligning to
the structure CHAINBREAK positions.  ``folding_dG`` asserts this alignment and
raises a clear error if it is ever violated rather than scoring silently
misaligned sequence/structure pairs.
"""

import os

os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")

import torch

from esm.pretrained import ESM3_sm_open_v0
from esm.utils.encoding import tokenize_sequence

from common.scorer import SequenceScorer
from common.seq_codec import (
    STAB_ALPHABET,
    chain_lengths,
    insert_chainbreaks,
    int_seq_to_str,
)
from track_esm3.esm3_struct import ESM3StructTokenizer


class ESM3Scorer(SequenceScorer):
    """Structure-conditioned ESM3 scorer (inverse-folding analogue of MPNN).

    Parameters
    ----------
    device : str
        ``"cpu"`` (float32) or a CUDA device (bfloat16).
    pdb_dir : str | None
        Directory of ``{domain['name']}.pdb`` files, forwarded to the
        structure tokenizer.
    model, struct_tokenizer :
        Optional pre-built ESM3 model / ``ESM3StructTokenizer`` for reuse / tests.
    """

    def __init__(self, device="cuda", pdb_dir=None, model=None, struct_tokenizer=None):
        super().__init__(device)
        # The esm3 checkpoint mixes bf16/float params; a forward fails unless we
        # cast to a uniform dtype after load (float32 on CPU, bfloat16 on GPU).
        dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
        self.model = (model or ESM3_sm_open_v0(device)).to(dtype).eval()
        self.seq_tok = self.model.tokenizers.sequence
        self.cb_id = self.seq_tok.get_vocab()["|"]
        self.struct = struct_tokenizer or ESM3StructTokenizer(device, pdb_dir=pdb_dir)

    def get_wt_seq(self, domain):
        """Return the wild-type sequence as a [1, L] StaB-alphabet int tensor."""
        seq = "".join(c if c in STAB_ALPHABET else "X" for c in domain["seq"])
        return torch.tensor(
            [[STAB_ALPHABET.index(c) for c in seq]], dtype=torch.long
        )

    def folding_dG(self, domain, seqs):
        """Predict dG = sum_i log P(s_i | fixed structure) for each sequence row."""
        st, coords, plddt = self.struct.tokens_for(domain)  # FIXED structure [1,T]
        lens = chain_lengths(domain)
        dGs = []
        for row in seqs:
            aa = insert_chainbreaks(int_seq_to_str(row), lens)
            seq_tokens = (
                tokenize_sequence(aa, self.seq_tok, add_special_tokens=True)
                .to(self.device)
                .unsqueeze(0)
            )
            # CRITICAL: sequence and structure token lengths must match, with the
            # '|' (seq chainbreak) positions aligned to the structure CHAINBREAK
            # positions.  Fail loudly instead of scoring a misaligned pair.
            assert seq_tokens.shape[1] == st.shape[1], (
                "sequence/structure length mismatch: seq_tokens "
                f"{tuple(seq_tokens.shape)} vs structure {tuple(st.shape)} "
                f"(domain {domain.get('name')!r}, chain_lengths={lens}). "
                "Check chain-break reconciliation (lengths AND ordering)."
            )
            with torch.no_grad():
                out = self.model.forward(
                    sequence_tokens=seq_tokens,
                    structure_tokens=st,
                    structure_coords=coords,
                    per_res_plddt=plddt,
                )
            logp = torch.log_softmax(out.sequence_logits.float(), dim=-1)[0]  # [T,64]
            tgt = seq_tokens[0]
            g = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            valid = ~(
                (tgt == self.seq_tok.cls_token_id)
                | (tgt == self.seq_tok.eos_token_id)
                | (tgt == self.cb_id)
            )
            dGs.append(g[valid].sum())
        return torch.stack(dGs).to(self.device)
