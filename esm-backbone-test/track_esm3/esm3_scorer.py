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

os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")

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

    @property
    def backbone_module(self):
        return self.model

    def get_wt_seq(self, domain):
        """Return the wild-type sequence as a [1, L] StaB-alphabet int tensor."""
        seq = "".join(c if c in STAB_ALPHABET else "X" for c in domain["seq"])
        return torch.tensor(
            [[STAB_ALPHABET.index(c) for c in seq]], dtype=torch.long
        )

    def folding_dG(self, domain, seqs):
        """Predict dG = sum_i log P(s_i | fixed structure) for each sequence row.

        The structure ``(st, coords, plddt)`` is FIXED per domain and every row
        of ``seqs`` shares the same length L (substitutions only, no indels), so
        all rows tokenize to the SAME length T == ``st.shape[1]``. We therefore
        stack the B rows into [B, T], broadcast the fixed structure across the
        batch, and run ONE forward — no padding, no attention mask.
        """
        st, coords, plddt = self.struct.tokens_for(domain)  # FIXED structure [1,T]
        # On GPU the ESM3 model params are bfloat16, but ESM3.forward internally
        # forces ``average_plddt``/``per_res_plddt`` to float32 (esm3.py:334-335),
        # so its float32 plddt feeds a bf16 ``plddt_projection`` Linear and a CPU
        # test never exercises this path.  Run the forward under autocast so the
        # bf16 Linear layers accept the float32 plddt/coord-derived inputs
        # ("mat1 and mat2 must have the same dtype" otherwise).  On CPU the model
        # is already float32 and autocast is a no-op (disabled).
        model_dtype = next(self.model.parameters()).dtype
        use_autocast = self.device != "cpu" and model_dtype != torch.float32
        lens = chain_lengths(domain)

        # Tokenize every row; all rows share L -> all share T (asserted below).
        rows = [
            tokenize_sequence(
                insert_chainbreaks(int_seq_to_str(row), lens),
                self.seq_tok,
                add_special_tokens=True,
            )
            for row in seqs
        ]
        T = rows[0].shape[0]
        assert all(r.shape[0] == T for r in rows), (
            "all rows must tokenize to the same length T (substitutions only); "
            f"got lengths {[int(r.shape[0]) for r in rows]}"
        )
        seq_tokens = torch.stack(rows, dim=0).to(self.device)  # [B, T]
        B = seq_tokens.shape[0]

        # CRITICAL: sequence and structure token lengths must match, with the
        # '|' (seq chainbreak) positions aligned to the structure CHAINBREAK
        # positions.  Fail loudly instead of scoring a misaligned pair.
        assert seq_tokens.shape[1] == st.shape[1], (
            "sequence/structure length mismatch: seq_tokens "
            f"{tuple(seq_tokens.shape)} vs structure {tuple(st.shape)} "
            f"(domain {domain.get('name')!r}, chain_lengths={lens}). "
            "Check chain-break reconciliation (lengths AND ordering)."
        )

        # Broadcast the FIXED structure across the B sequence rows.
        st_b = st.expand(B, -1)                       # [B, T]
        coords_b = coords.expand(B, -1, -1, -1)       # [B, T, 37, 3]
        plddt_b = plddt.expand(B, -1)                 # [B, T]

        # NOTE: no internal ``torch.no_grad()`` here — finetuning needs gradients
        # to flow through this forward. Eval paths (``run/eval.py``) wrap the
        # forward in ``torch.no_grad()`` externally, so inference is unaffected.
        with torch.autocast(
            device_type="cuda", dtype=model_dtype, enabled=use_autocast
        ):
            out = self.model.forward(
                sequence_tokens=seq_tokens,
                structure_tokens=st_b,
                structure_coords=coords_b,
                per_res_plddt=plddt_b,
            )
        logp = torch.log_softmax(out.sequence_logits.float(), dim=-1)  # [B, T, 64]
        g = logp.gather(-1, seq_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T]
        valid = ~(
            (seq_tokens == self.seq_tok.cls_token_id)
            | (seq_tokens == self.seq_tok.eos_token_id)
            | (seq_tokens == self.cb_id)
        )  # [B, T]
        dGs = (g * valid).sum(dim=-1)  # [B]
        return dGs.to(self.device)
