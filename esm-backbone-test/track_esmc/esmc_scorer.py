"""
esmc_scorer.py — sequence-only SequenceScorer backed by ESM-C.

ESMCScorer scores a sequence's folding free energy as

    dG(domain, seq) = sum_i log P(s_i | seq)

using ESM-C's per-residue (masked-LM-style) logits. Multi-chain complexes are
joined with the ESM '|' chain-break token, and all special tokens
(<cls>, <eos>, pad, '|') are excluded from the sum.

The scorer is sequence-only: ``folding_dG`` ignores backbone structure entirely
and depends only on the amino-acid identities and the per-chain lengths of the
domain (used to place chain-breaks).

Correctness note (padding): sequences are scored one at a time (B=1, no
padding). This is provably free of the pad-attention corruption that would
arise from padding a batch without supplying a validity mask to ESM-C. The
per-row loop is the bottleneck for large batches; this can later be swapped for
a masked batched forward (passing a bool ``sequence_id`` mask) if needed.
"""

import torch

from common.scorer import SequenceScorer
from common.seq_codec import (
    STAB_ALPHABET,
    int_seq_to_str,
    chain_lengths,
    insert_chainbreaks,
)
from common import hf_compat

# Apply the HF .pth-fallback patch at import time so the scorer works when
# imported from production scripts (finetune/eval) that don't load conftest.
hf_compat.apply()

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils import encoding
from esm.utils.constants.models import ESMC_600M, ESMC_6B

_MODEL_NAMES = {"esmc_600m": ESMC_600M, "esmc_6b": ESMC_6B}


class ESMCScorer(SequenceScorer):
    """ESM-C-backed sequence-only scorer (dG = sum_i log P(s_i))."""

    def __init__(
        self,
        model_name="esmc_600m",
        device="cuda",
        use_flash_attn=True,
        model=None,
    ):
        super().__init__(device)
        hf_compat.apply()

        self.tokenizer = get_esmc_model_tokenizers()
        vocab = self.tokenizer.get_vocab()
        self.cb_id = vocab["|"]

        # Set of token ids to exclude from the dG sum (special tokens).
        self._special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.cb_id,
        }

        if model is not None:
            self.model = model
        else:
            self.model = ESMC.from_pretrained(
                _MODEL_NAMES[model_name],
                device=torch.device(device),
                use_flash_attn=use_flash_attn,
            )
        self.model.eval()

    def get_wt_seq(self, domain) -> torch.Tensor:
        """Wild-type sequence as a [1, L] StaB-index long tensor."""
        idx = STAB_ALPHABET.index
        x_id = STAB_ALPHABET.index("X")
        ids = [idx(c) if c in STAB_ALPHABET else x_id for c in domain["seq"]]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    @torch.no_grad()
    def folding_dG(self, domain, seqs) -> torch.Tensor:
        """dG = sum_i log P(s_i) over real residues, for each row in ``seqs``.

        Returns a [B] tensor. Scores one sequence at a time (no padding).
        """
        self._cur_lengths = chain_lengths(domain)
        model_device = next(self.model.parameters()).device

        dGs = []
        for row in seqs:
            aa = insert_chainbreaks(int_seq_to_str(row), self._cur_lengths)
            tokens = encoding.tokenize_sequence(
                aa, self.tokenizer, add_special_tokens=True
            ).to(model_device)
            tokens = tokens.unsqueeze(0)  # [1, T]

            out = self.model(sequence_tokens=tokens)
            logits = out.sequence_logits.float()  # [1, T, V]
            log_probs = torch.log_softmax(logits, dim=-1)  # [1, T, V]

            tok = tokens[0]  # [T]
            tok_lp = log_probs[0, torch.arange(tok.shape[0], device=model_device), tok]

            # Mask out special tokens (cls / eos / pad / '|') from the sum.
            keep = torch.ones_like(tok, dtype=torch.bool)
            for sid in self._special_ids:
                if sid is not None:
                    keep &= tok != sid

            dGs.append(tok_lp[keep].sum())

        return torch.stack(dGs).to(self.device)
