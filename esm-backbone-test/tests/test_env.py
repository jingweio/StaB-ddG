"""
test_env.py — smoke test: can we load ESMC-600M from the local cache offline?

Run from esm-backbone-test/:
    cd esm-backbone-test && python -m pytest tests/test_env.py -v
"""

import os

os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def test_esmc_600m_loads_offline():
    import torch
    from esm.models.esmc import ESMC
    from esm.utils.constants.models import ESMC_600M

    m = ESMC.from_pretrained(ESMC_600M, device=torch.device("cpu"), use_flash_attn=False)
    assert sum(p.numel() for p in m.parameters()) > 5e8  # ~600M params
