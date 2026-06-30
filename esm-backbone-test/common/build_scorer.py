"""
build_scorer.py — backbone factory for StaB-ddG.

``build_scorer(backbone, ...)`` returns a :class:`common.scorer.SequenceScorer`
for one of the supported backbones:

  - ``"mpnn"``      : ProteinMPNN, loaded from a StaB-ddG checkpoint.
  - ``"esmc_600m"`` / ``"esmc_6b"`` : sequence-only ESM-C scorer.
  - ``"esm3"``      : structure-conditioned ESM3 scorer.

The returned scorer is plugged into :class:`common.stab_model.StaBddG` to form a
backbone-agnostic binding-ddG model.
"""


def build_scorer(backbone, device="cuda", checkpoint=None, pdb_dir=None):
    """Build a SequenceScorer for the requested backbone.

    Parameters
    ----------
    backbone : str
        One of ``{"mpnn", "esmc_600m", "esmc_6b", "esm3"}``.
    device : str
        Torch device string (e.g. ``"cuda"`` / ``"cpu"``).
    checkpoint : str | None
        ProteinMPNN checkpoint path (``"mpnn"`` backbone only). Defaults to
        ``"model_ckpts/stabddg.pt"`` when ``None``.
    pdb_dir : str | None
        Directory of ``{name}.pdb`` files, used by the structure-conditioned
        ``"esm3"`` backbone to look up backbones for each domain.
    """
    if backbone == "mpnn":
        import torch
        from stabddg.mpnn_utils import ProteinMPNN
        from common.scorer import MPNNScorer

        # EXACT constructor from run_stabddg.py:130-144 (verbatim, incl. dropout=0.0).
        pmpnn = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                            num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48,
                            dropout=0.0, augment_eps=0.0)
        ckpt = torch.load(checkpoint or "model_ckpts/stabddg.pt", map_location=device)
        pmpnn.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        return MPNNScorer(pmpnn.to(device).eval(), device=device)

    if backbone in ("esmc_600m", "esmc_6b"):
        from track_esmc.esmc_scorer import ESMCScorer
        return ESMCScorer(backbone, device=device)

    if backbone == "esm3":
        from track_esm3.esm3_scorer import ESM3Scorer
        return ESM3Scorer(device=device, pdb_dir=pdb_dir)

    if backbone == "esm3_stab":
        from track_esm3stab.esm3_stab_scorer import ESM3StabScorer
        return ESM3StabScorer(device=device, pdb_dir=pdb_dir)

    raise ValueError(f"unknown backbone {backbone}")
