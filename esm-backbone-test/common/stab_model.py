"""
stab_model.py — backbone-agnostic StaB-ddG model.

``StaBddG`` here holds a ``SequenceScorer`` (any backbone) and computes the
binding ddG by decomposing it into three folding ddG terms — for the whole
complex and each of the two binders — exactly as the original
``stabddg.model.StaBddG`` does, but delegating all scoring to the injected
``scorer`` rather than hard-coding ProteinMPNN.
"""

import torch
from torch import nn

from common.scorer import SequenceScorer


class StaBddG(nn.Module):
    """Binding-ddG predictor parameterised by a :class:`SequenceScorer`."""

    def __init__(self, scorer: SequenceScorer):
        super().__init__()
        self.scorer = scorer

    def binding_ddG(self, complex, binder1, binder2,
                    complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs):
        """Binding ddG = complex folding ddG - (binder1 + binder2 folding ddG).

        We calculate the binding ddG by decomposing it into three folding ddG
        terms, corresponding to the entire complex and each individual binder.
        """
        complex_ddG_fold = self.scorer.folding_ddG(complex, complex_mut_seqs)
        binder1_ddG_fold = self.scorer.folding_ddG(binder1, binder1_mut_seqs)
        binder2_ddG_fold = self.scorer.folding_ddG(binder2, binder2_mut_seqs)

        ddG = complex_ddG_fold - (binder1_ddG_fold + binder2_ddG_fold)

        return ddG

    def forward(self, complex, binder1, binder2,
                complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs):
        return self.binding_ddG(complex, binder1, binder2,
                                complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs)
