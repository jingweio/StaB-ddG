import torch

STAB_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"   # mpnn_utils.py:211 ; index 20 = X (unknown/gap)

def int_seq_to_str(int_row) -> str:
    """[L] int64 (StaB alphabet) -> AA string (length L)."""
    return "".join(STAB_ALPHABET[i] for i in int_row.tolist())

def chain_lengths(domain) -> list:
    """Per-chain residue counts in the SAME order featurize concatenates 'seq'
    (insertion order of 'seq_chain_*' keys)."""
    return [len(v) for k, v in domain.items() if k.startswith("seq_chain_")]

def insert_chainbreaks(aa_str: str, lengths: list, sep="|") -> str:
    """Join chain slices with `sep` so total residues = sum(lengths). Never trailing sep."""
    out, p = [], 0
    for n in lengths:
        out.append(aa_str[p:p + n]); p += n
    return sep.join(out)

def stab_int_to_esm_ids(int_row, esm_vocab: dict, unk_token="X") -> list:
    """Map each StaB 21-index residue to an ESM token id (per-residue, no specials)."""
    ids = []
    for i in int_row.tolist():
        ch = STAB_ALPHABET[i]
        ids.append(esm_vocab.get(ch, esm_vocab[unk_token]))
    return ids
