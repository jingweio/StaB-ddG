"""
esm3_struct.py — ESM3 structure tokenizer for StaB domains.

`ESM3StructTokenizer` converts a StaB ``domain`` (a parsed-PDB dict, as produced
by ``stabddg.ppi_dataset.SKEMPIDataset``) into ESM3 structure tokens via the
ESM3 VQ-VAE structure encoder, wraps them with the canonical BOS/EOS special
tokens, and caches the result per domain name.  This feeds the ESM3Scorer
(Task 5).

Structure-token special ids (note BOS > EOS — intentional, matching ESM3):
    STRUCT_BOS  = 4098
    STRUCT_EOS  = 4097
    STRUCT_MASK = 4096
Interior (real) structure tokens live in [0, 4096).

Chain construction (robustness is the goal):
    1. If ``pdb_dir`` is set and ``{pdb_dir}/{domain['name']}.pdb`` exists, load
       it from disk (validated path).  Single-chain PDBs go through
       ``ProteinChain.from_pdb(path, chain_id="detect")``; multi-chain PDBs go
       through ``ProteinComplex.from_pdb(path).as_chain(force_conversion=True)``
       so every chain's residues are kept and fed to the encoder together.
    2. Otherwise reconstruct from the domain's backbone coords
       (``coords_chain_<id>`` dicts) into an atom37 array (N->slot0, CA->slot1,
       C->slot2, O->slot4, all other slots NaN) and use
       ``ProteinChain.from_atom37(atom37, sequence=concat_seqs)``.

The encode + BOS/EOS-wrap below mirrors ``esm.utils.encoding.tokenize_structure``
exactly (keeping a leading batch dim of 1).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex

# ESM3 structure-token special ids.
STRUCT_BOS = 4098
STRUCT_EOS = 4097
STRUCT_MASK = 4096

# atom37 slot indices for the backbone atoms StaB stores (N, CA, C, O).
# ESM / AlphaFold atom37 ordering: 0=N, 1=CA, 2=C, 3=CB, 4=O, ...
_ATOM37_SLOTS = {"N": 0, "CA": 1, "C": 2, "O": 4}
_ATOM37_NUM = 37


class ESM3StructTokenizer:
    """Tokenize StaB domains into BOS/EOS-wrapped ESM3 structure tokens.

    Results are cached by ``domain['name']``; a repeat call with the same name
    returns the *same* tuple object.
    """

    def __init__(self, device: str = "cuda", pdb_dir: str | None = None):
        # The structure encoder runs in float32 for numerical stability — bf16/
        # fp16 degrades the geometric kNN features and corrupts the tokens.
        self.encoder = (
            ESM3_structure_encoder_v0(device).to(torch.float32).eval()
        )
        self.device = device
        self.pdb_dir = pdb_dir
        self._cache: dict[str, tuple] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def tokens_for(self, domain: dict):
        """Return ``(structure_tokens[1,T], coords[1,T,37,3], plddt[1,T])``.

        ``T == L + 2`` (L real residues plus BOS and EOS).  Cached by
        ``domain['name']`` — repeat calls return the identical cached object.
        """
        name = domain["name"]
        if name in self._cache:
            return self._cache[name]

        chain = self._chain_for(domain)
        result = self._encode_and_wrap(chain)
        self._cache[name] = result
        return result

    # ------------------------------------------------------------------ #
    # Chain construction                                                  #
    # ------------------------------------------------------------------ #
    def _chain_for(self, domain: dict) -> ProteinChain:
        """Build a ProteinChain for ``domain`` using the priority in the module
        docstring."""
        name = domain["name"]

        # Priority 1: load from disk if a matching PDB exists (validated path).
        if self.pdb_dir is not None:
            pdb_path = os.path.join(self.pdb_dir, f"{name}.pdb")
            if os.path.exists(pdb_path):
                return self._chain_from_pdb(pdb_path)

        # Priority 2: reconstruct from the domain's backbone coords.
        return self._chain_from_domain_coords(domain)

    @staticmethod
    def _chain_from_pdb(pdb_path: str) -> ProteinChain:
        """Load a (possibly multi-chain) PDB into a single ProteinChain.

        ESM's ``ProteinChain.from_pdb`` only reads a single author chain
        (``"detect"`` => first chain), so for multi-chain files we read the whole
        complex and collapse it with ``as_chain(force_conversion=True)``, which
        keeps every chain's residues (the structure encoder distinguishes chains
        via gaps in ``residue_index``).
        """
        complex_ = ProteinComplex.from_pdb(pdb_path)
        if complex_.num_chains <= 1:
            # Single chain: the simpler ProteinChain path is cleaner (no forced
            # chain-id collapse) and is the explicitly validated route.
            return ProteinChain.from_pdb(pdb_path, chain_id="detect")
        return complex_.as_chain(force_conversion=True)

    @staticmethod
    def _chain_from_domain_coords(domain: dict) -> ProteinChain:
        """Reconstruct a ProteinChain from a domain's backbone-only coords.

        Builds an atom37 array (N/CA/C/O populated, all other slots NaN) by
        concatenating the domain's chains in numeric/sorted chain order, then
        defers to ``ProteinChain.from_atom37``.
        """
        # Discover the per-chain coord dicts: keys look like 'coords_chain_<id>'
        # with matching sequence keys 'seq_chain_<id>'.
        chain_ids = sorted(
            k[len("coords_chain_"):]
            for k in domain
            if k.startswith("coords_chain_")
        )
        if not chain_ids:
            raise ValueError(
                f"domain {domain.get('name')!r} has no 'coords_chain_*' entries; "
                "cannot reconstruct a structure (and no PDB was found on disk)."
            )

        per_chain_atom37 = []
        seqs = []
        for cid in chain_ids:
            coords_dict = domain[f"coords_chain_{cid}"]
            seq = domain[f"seq_chain_{cid}"]
            ca = np.asarray(coords_dict[f"CA_chain_{cid}"], dtype=np.float32)
            n_res = ca.shape[0]
            atom37 = np.full((n_res, _ATOM37_NUM, 3), np.nan, dtype=np.float32)
            for atom_name, slot in _ATOM37_SLOTS.items():
                arr = np.asarray(
                    coords_dict[f"{atom_name}_chain_{cid}"], dtype=np.float32
                )
                atom37[:, slot, :] = arr
            per_chain_atom37.append(atom37)
            seqs.append(seq)

        atom37 = np.concatenate(per_chain_atom37, axis=0)
        sequence = "".join(seqs)

        try:
            return ProteinChain.from_atom37(atom37, sequence=sequence)
        except Exception as exc:  # noqa: BLE001 — report a clear, actionable error
            raise RuntimeError(
                f"ProteinChain.from_atom37 failed for domain "
                f"{domain.get('name')!r} (likely NaN side-chain atoms in the "
                f"backbone-only reconstruction): {type(exc).__name__}: {exc}. "
                "Provide a real PDB via pdb_dir to use the validated from_pdb "
                "path."
            ) from exc

    # ------------------------------------------------------------------ #
    # Encode + BOS/EOS wrap (mirrors encoding.tokenize_structure)         #
    # ------------------------------------------------------------------ #
    def _encode_and_wrap(self, chain: ProteinChain):
        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.to(self.device)               # (1, L, 37, 3)
        plddt = plddt.to(self.device)                 # (1, L)
        residue_index = residue_index.to(self.device)  # (1, L)

        with torch.no_grad():
            _, structure_tokens = self.encoder.encode(
                coords.float(), residue_index=residue_index
            )  # (1, L), ids in [0, 4096)

        # Wrap with BOS / EOS, matching encoding.tokenize_structure (the
        # interior padding uses the MASK id, then the ends are overwritten).
        coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)  # (1, L+2, 37, 3)
        plddt = F.pad(plddt, (1, 1), value=0)                        # (1, L+2)
        structure_tokens = F.pad(
            structure_tokens, (1, 1), value=STRUCT_MASK
        )  # (1, L+2)
        structure_tokens[:, 0] = STRUCT_BOS
        structure_tokens[:, -1] = STRUCT_EOS

        return structure_tokens, coords, plddt
