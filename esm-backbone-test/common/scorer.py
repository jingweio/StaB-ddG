"""
scorer.py — backbone-agnostic sequence-scoring interface for StaB-ddG.

This module extracts StaB-ddG's scoring primitive into a small abstract
interface, ``SequenceScorer``, and provides ``MPNNScorer`` which wraps the
exact ProteinMPNN logic that currently lives in ``stabddg/model.py``.

The point of ``MPNNScorer`` is *behaviour preservation*: its ``folding_dG`` /
``folding_ddG`` reproduce the original ``stabddg.model.StaBddG`` computation
bit-for-bit (see ``tests/test_mpnn_scorer_regression.py``).  Later tasks add
alternative backbones (ESM-C / ESM-3) behind the same ``SequenceScorer``
interface.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod

from stabddg.mpnn_utils import featurize


class SequenceScorer(nn.Module, ABC):
    """Abstract scoring primitive: maps (structure domain, sequences) -> dG.

    Concrete subclasses implement a backbone-specific ``folding_dG`` and
    ``get_wt_seq``.  ``folding_ddG`` is defined generically as the difference
    of folding free energies between mutant and wild-type sequences.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    @abstractmethod
    def folding_dG(self, domain, seqs) -> torch.Tensor:
        """Predict folding stability (dG) for a batch of sequences -> [B]."""
        ...

    @abstractmethod
    def get_wt_seq(self, domain) -> torch.Tensor:
        """Return the wild-type sequence tensor for ``domain`` -> [1, L]."""
        ...

    @property
    @abstractmethod
    def backbone_module(self) -> nn.Module:
        """The trainable backbone ``nn.Module``.

        Finetuning trains ``scorer.backbone_module.parameters()`` and saves
        ``scorer.backbone_module.state_dict()`` (see ``run/finetune.py``).  This
        decouples the optimizer/checkpoint from the scorer wrapper so the exact
        same training loop drives any backbone (ProteinMPNN / ESM-C / ESM3).
        """
        ...

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None) -> torch.Tensor:
        """ddG = dG(mutant) - dG(wild-type)."""
        wt = self.get_wt_seq(domain) if set_wt_seq is None else set_wt_seq
        return self.folding_dG(domain, mut_seqs) - self.folding_dG(domain, wt)


class MPNNScorer(SequenceScorer):
    """ProteinMPNN-backed scorer.

    Lifts the exact scoring logic from ``stabddg.model.StaBddG`` so that, with
    ``use_antithetic_variates=False`` and matched RNG, predictions are
    bit-identical to the original implementation.
    """

    def __init__(self, pmpnn, use_antithetic_variates=True, noise_level=0.1, device="cuda"):
        super().__init__(device)
        self.pmpnn = pmpnn
        self.use_antithetic_variates = use_antithetic_variates
        self.noise_level = noise_level

    @property
    def backbone_module(self):
        return self.pmpnn

    def get_wt_seq(self, domain):
        """Returns the wild type sequence of a protein."""
        _, wt_seq, *_ = featurize([domain], self.device)
        return wt_seq

    def folding_dG(self, domain, seqs, decoding_order=None, backbone_noise=None):
        """Predicts the folding stability (dG) for a list of sequences."""
        B = seqs.shape[0]

        X_, _, mask_, _, chain_M_, residue_idx_, _, chain_encoding_all_ = featurize([domain], self.device)
        X_, S_, mask_ = X_.repeat(B, 1, 1, 1), seqs.to(self.device), mask_.repeat(B, 1)
        chain_M_ = chain_M_.repeat(B, 1)
        residue_idx_, chain_encoding_all_ = residue_idx_.repeat(B, 1), chain_encoding_all_.repeat(B, 1)

        order = decoding_order.repeat(B, 1) if self.use_antithetic_variates else None
        backbone_noise = backbone_noise.repeat(B, 1, 1, 1) if self.use_antithetic_variates else None

        log_probs = self.pmpnn(X_, S_, mask_, chain_M_, residue_idx_, chain_encoding_all_,
                               fix_order=order, fix_backbone_noise=backbone_noise)

        seq_oh = torch.nn.functional.one_hot(seqs, 21).to(self.device)
        dG = torch.sum(seq_oh * log_probs, dim=(1, 2))

        return dG

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None):
        """Predicts the folding ddG."""
        X, wt_seq, _, _, chain_M, _, _, _ = featurize([domain], self.device)

        if not set_wt_seq is None:
            wt_seq = set_wt_seq

        decoding_order = self._get_decoding_order(chain_M) if self.use_antithetic_variates else None
        backbone_noise = self._get_backbone_noise(X) if self.use_antithetic_variates else None

        wt_dG = self.folding_dG(domain, wt_seq, decoding_order=decoding_order, backbone_noise=backbone_noise)
        mut_dG = self.folding_dG(domain, mut_seqs, decoding_order=decoding_order, backbone_noise=backbone_noise)

        ddG = mut_dG - wt_dG

        return ddG

    def _get_decoding_order(self, chain_M):
        """Generate a random decoding order with the same shape as chain_M."""
        return torch.argsort(torch.abs(torch.randn(chain_M.shape, device=self.device)))

    def _get_backbone_noise(self, X):
        """Generate random backbone noise. Defaults to 0.1A."""
        return self.noise_level * torch.randn_like(X, device=self.device)
