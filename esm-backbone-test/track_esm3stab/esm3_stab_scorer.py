"""esm3_stab_scorer.py — ESM3 + LoRA + stability-head folding-energy scorer.

The parameter-efficient analogue of the old likelihood ESM3Scorer: ESM3 base is
FROZEN, only LoRA r=4 adapters + a small per-residue regression head train. The
head maps ESM3 per-residue embeddings -> a per-residue dG; folding_dG SUMS them
over the real-residue positions (extensive, so StaB's complex-binder1-binder2
binding-ddG decomposition stays additive). No sigmoid scaling: we supervise ddG
only (StaB's loss), so the absolute dG offset is free.

Implementation note on peft API
---------------------------------
ESM3 is a plain nn.Module (not a transformers.PreTrainedModel), so
``get_peft_model`` wraps it in a PeftModelForSequenceClassification whose
``forward`` assumes transformers kwargs (``return_dict``, ``config``) and breaks
ESM3's signature.  We use ``inject_adapter_in_model`` instead: it injects the
LoRA layers directly into ESM3's named submodules *in-place* and returns the
original object, so ``base.forward(sequence_tokens=..., ...)`` keeps working
unchanged.  Saving/loading adapters uses the model's own ``state_dict`` filtered
by the ``lora_`` prefix — equivalent to peft's ``save_pretrained`` but without
the PeftModel wrapper.
"""
import os

os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")

import torch
from torch import nn

from esm.pretrained import ESM3_sm_open_v0
from esm.utils.encoding import tokenize_sequence
from peft import LoraConfig, TaskType, inject_adapter_in_model

from common.scorer import SequenceScorer
from common.seq_codec import (
    STAB_ALPHABET,
    chain_lengths,
    insert_chainbreaks,
    int_seq_to_str,
)
from track_esm3.esm3_struct import ESM3StructTokenizer


class ESM3StabilityHead(nn.Module):
    """Per-residue stability head: Linear -> LayerNorm -> ReLU -> Linear(->1)."""

    def __init__(self, input_dim=1536, output_dim=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.classifier(x)


class ESM3StabScorer(SequenceScorer):
    """Structure-conditioned ESM3 + LoRA + stability-head scorer."""

    LORA_TARGET_MODULES = ["layernorm_qkv.1", "out_proj"]

    def __init__(self, device="cuda", pdb_dir=None, model=None, struct_tokenizer=None,
                 lora_rank=4, lora_dropout=0.15):
        super().__init__(device)
        # esm3 checkpoint mixes bf16/float; cast to a uniform dtype after load
        # (float32 on CPU, bf16 on GPU) — same as the likelihood ESM3Scorer.
        dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
        base = (model or ESM3_sm_open_v0(device)).to(dtype)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=self.LORA_TARGET_MODULES,
            r=lora_rank,
            lora_dropout=lora_dropout,
        )
        # inject_adapter_in_model modifies base in-place and returns the same
        # object (ESM3 is not a PreTrainedModel; get_peft_model wraps it in a
        # PeftModelForSequenceClassification that breaks ESM3's forward signature).
        inject_adapter_in_model(peft_config, base)
        self.model = base
        # Head stays float32 for stability / dtype-safety (embeddings cast to float).
        self.head = ESM3StabilityHead(input_dim=1536, output_dim=1).to(device)
        self.seq_tok = self.model.tokenizers.sequence
        self.cb_id = self.seq_tok.get_vocab()["|"]
        self.struct = struct_tokenizer or ESM3StructTokenizer(device, pdb_dir=pdb_dir)
        self.freeze_base()
        self.eval()  # dropout off by default; finetune.py calls .train() for training

    @property
    def backbone_module(self):
        # The trainable unit is LoRA(model) + head together; the scorer IS that
        # nn.Module. finetune.py uses .train()/trainable_parameters()/save_adapters.
        return self

    def freeze_base(self):
        """Freeze every ESM3 base param; keep LoRA adapters + head trainable."""
        for n, p in self.model.named_parameters():
            p.requires_grad = "lora_" in n
        for p in self.head.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(
            self.head.parameters()
        )

    def get_wt_seq(self, domain):
        seq = "".join(c if c in STAB_ALPHABET else "X" for c in domain["seq"])
        return torch.tensor([[STAB_ALPHABET.index(c) for c in seq]], dtype=torch.long)

    def folding_dG(self, domain, seqs):
        """dG = sum_i head(emb_i) over real-residue positions, per sequence row."""
        st, coords, plddt = self.struct.tokens_for(domain)  # FIXED structure [1,T]
        model_dtype = next(self.model.parameters()).dtype
        use_autocast = self.device != "cpu" and model_dtype != torch.float32
        lens = chain_lengths(domain)

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
            f"got {[int(r.shape[0]) for r in rows]}"
        )
        seq_tokens = torch.stack(rows, dim=0).to(self.device)  # [B, T]
        B = seq_tokens.shape[0]
        assert seq_tokens.shape[1] == st.shape[1], (
            f"seq/structure length mismatch: {tuple(seq_tokens.shape)} vs "
            f"{tuple(st.shape)} (domain {domain.get('name')!r}, chain_lengths={lens})"
        )

        st_b = st.expand(B, -1)
        coords_b = coords.expand(B, -1, -1, -1)
        plddt_b = plddt.expand(B, -1)

        with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=use_autocast):
            out = self.model.forward(
                sequence_tokens=seq_tokens,
                structure_tokens=st_b,
                structure_coords=coords_b,
                per_res_plddt=plddt_b,
            )
        emb = out.embeddings.float()                  # [B, T, 1536]
        per_res = self.head(emb).squeeze(-1)          # [B, T]
        valid = ~(
            (seq_tokens == self.seq_tok.cls_token_id)
            | (seq_tokens == self.seq_tok.eos_token_id)
            | (seq_tokens == self.cb_id)
        )                                             # [B, T]
        dGs = (per_res * valid).sum(dim=-1)           # [B] extensive
        return dGs.to(self.device)

    def save_adapters(self, path):
        """Save ONLY LoRA + head weights (small file)."""
        sd = {f"model.{k}": v for k, v in self.model.state_dict().items() if "lora_" in k}
        sd.update({f"head.{k}": v for k, v in self.head.state_dict().items()})
        torch.save(sd, path)

    def load_adapters(self, path):
        sd = torch.load(path, map_location=self.device)
        model_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        head_sd = {k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")}
        self.model.load_state_dict(model_sd, strict=False)
        self.head.load_state_dict(head_sd, strict=True)
