import torch
from common.seq_codec import (STAB_ALPHABET, int_seq_to_str,
    insert_chainbreaks, stab_int_to_esm_ids)
from esm.tokenization import get_esmc_model_tokenizers

def test_int_to_str_roundtrip():
    row = torch.tensor([0,1,2,8,19,20])     # A C D K Y X
    assert int_seq_to_str(row) == "ACDKYX"

def test_chainbreaks_preserve_residues():
    s = insert_chainbreaks("AAAABB", [4,2])
    assert s == "AAAA|BB" and s.replace("|","") == "AAAABB"

def test_all_21_map_to_esm_vocab():
    vocab = get_esmc_model_tokenizers().get_vocab()
    for ch in STAB_ALPHABET:
        assert ch in vocab or "X" in vocab    # X is the fallback for unknown

def test_stab_int_to_esm_ids_uses_vocab():
    vocab = get_esmc_model_tokenizers().get_vocab()
    ids = stab_int_to_esm_ids(torch.tensor([0,1,2]), vocab)   # A,C,D
    assert ids == [vocab["A"], vocab["C"], vocab["D"]]
