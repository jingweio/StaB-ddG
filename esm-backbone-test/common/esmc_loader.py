"""Robust ESMC weight loader.

The official `ESMC.from_pretrained` path was BROKEN for our use on Ibex:
  1. it uses accelerate meta-device init + huggingface_hub `load_torch_model`, which
     left params on the meta device -> ".to(cuda)" raised "Cannot copy out of meta tensor";
  2. the checkpoint keys carry an `esmc.` prefix and name the output head `lm_head`,
     whereas the model expects no prefix and `sequence_head` -> a naive load_state_dict
     silently leaves 808/808 keys unmatched (RANDOM weights), which passes shape-only
     tests but yields garbage logits (self-recovery ~0).

This loader fixes both: real-init on CPU (so computed buffers exist), then load the
sharded safetensors (6B) or .pth (600M) with the key remap. VERIFIED on a100:
both esmc_600m and esmc_6b -> missing=0, unexpected=0, self-recovery=0.828.
It RAISES if any model weight is missing after remap (loud failure, never silent random).
"""
import os
import glob
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers

_CFG = {
    "esmc_600m": dict(d_model=1152, n_heads=18, n_layers=36, repo="biohub/esmc-600m-2024-12"),
    "esmc_6b":   dict(d_model=2560, n_heads=40, n_layers=80, repo="biohub/esmc-6b-2024-12"),
}


def _remap(k):
    if k.startswith("esmc."):
        k = k[len("esmc."):]
    if k.startswith("lm_head."):
        k = "sequence_head." + k[len("lm_head."):]
    return k


def load_esmc(model_name, device="cuda", use_flash_attn=True, dtype=None, tokenizer=None):
    """Return a correctly-loaded ESMC model on `device`.

    dtype: defaults to bfloat16 on cuda (required by flash-attn), float32 on cpu.
    Pass dtype=torch.float32 explicitly for FSDP (keep fp32 master weights; let
    FSDP MixedPrecision handle bf16 compute).
    """
    if model_name not in _CFG:
        raise ValueError(f"unknown ESMC model {model_name}")
    cfg = _CFG[model_name]
    tok = tokenizer or get_esmc_model_tokenizers()
    model = ESMC(d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                 tokenizer=tok, use_flash_attn=use_flash_attn)
    path = snapshot_download(repo_id=cfg["repo"])
    raw = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        raw.update(load_file(f))
    if not raw:  # 600m stores weights as .pth under data/weights/
        for f in sorted(glob.glob(os.path.join(path, "**", "*.pth"), recursive=True)):
            raw.update(torch.load(f, map_location="cpu", weights_only=False))
    if not raw:
        raise RuntimeError(f"no weight files found for {model_name} under {path}")
    sd = {_remap(k): v for k, v in raw.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"ESMC {model_name}: {len(missing)} missing keys after remap (would be random "
            f"weights!), e.g. {missing[:5]}. Check the esmc./lm_head remap."
        )
    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return model.to(device).to(dtype).eval()
