# ESM-Backbone Swap for StaB-ddG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace StaB-ddG's ProteinMPNN scoring backbone with two ESM backbones — **Track A: ESM3-1.4B** (structure-conditioned, single-GPU full-FT) and **Track B: ESMC-6B** (sequence-only, multi-GPU FSDP full-FT) — and run the full two-stage (Megascale→SKEMPI) train + inference flow to test whether either improves PPI ΔΔG_bind prediction over the ProteinMPNN baseline (per-interface Spearman **0.445**).

**Architecture:** StaB-ddG's only backbone call is `folding_dG(domain, seqs) → Σᵢ log P(sᵢ|context)`; everything else (mut−wt, complex−binder₁−binder₂, MC ensemble, sign) is built on it. We introduce a `SequenceScorer` interface with three implementations (`MPNNScorer`, `ESMCScorer`, `ESM3Scorer`) and make `StaBddG` + the three scripts (`stability_finetune.py`, `skempi_finetune.py`, `skempi_eval.py`) backbone-agnostic via a `--backbone` flag. The two tracks live in strictly separated subdirs under `esm-backbone-test/`; ESMC-6B adds an FSDP training harness (the repo has none).

**Tech Stack:** PyTorch (FSDP/ZeRO-3, activation checkpointing, bf16 mixed precision), EvolutionaryScale `esm` v3.3.0 (ESMC, ESM3, VQ-VAE structure encoder), flash-attn, KAUST Ibex SLURM (a100), conda env `esm-backbone`.

---

## Key facts the implementer MUST know (verified against code)

- **`featurize(batch, device)` returns an 8-tuple in a non-obvious order**: `(X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all)` (`stabddg/mpnn_utils.py:317`). `lengths` (elem 4) is a numpy array, not a tensor; `mask_self` (elem 7) is unused by `ProteinMPNN.forward`. ProteinMPNN consumes only `(X, S, mask, chain_M, residue_idx, chain_encoding_all)`.
- **Two different 21-letter alphabets exist.** Integer sequences (`S`, `mut_seqs`) everywhere downstream use **`ACDEFGHIKLMNPQRSTVWYX`** (`mpnn_utils.py:211`): index 0=A…19=Y, **20=X** (catch-all for unknown AND gap `-`). The ESM scorers must map *from this 21-index space* to the ESM tokenizers.
- **`mut_seqs` are `[N_muts, L]` int64**, and **L differs across complex / binder1 / binder2** (only `N_muts` is shared). A mutation on a chain absent from a binder is silently skipped (`ppi_dataset.py:190`), so a binder's mutant row can equal its WT row. (`folding_ddG` then yields ~0 for that binder, which is correct.)
- **The model checkpoints only ever store `pmpnn.state_dict()`**, never the `StaBddG` wrapper (`skempi_finetune.py` save line). After refactor, ESM scorers will save their own backbone `state_dict()` analogously.
- **`skempi_eval.py` writes CSV WITHOUT `index=False`** (`skempi_eval.py:126`) → the output has a leading unnamed index column plus `#Pdb,Mutation,ddG,ddG_pred`. Our reproduced baseline CSV has this exact shape; keep it identical so `reproduce/compute_metrics.py` / `mutation-analysis` keep working.
- **Loss = `torch.nn.MSELoss()` on predicted vs true ΔΔG**; optimizer = `torch.optim.Adam(model.parameters(), lr=lr)` (no weight decay), per-chunk `backward()/step()/zero_grad()`. SKEMPI hardcodes `use_antithetic_variates=True`.
- **ESMC/ESM3 logits vocab is V=64** (`RegressionHead(d_model, 64)`); only ids 0–32 are meaningful. On CUDA both cast to **bf16** — always `logits.float()` before `log_softmax` for stable summed dG.
- **Multi-chain in ESM = `'|'` chain-break token (id 31)**; never place it at the final/penultimate position. Exclude `'|'`, BOS(`<cls>`=0), EOS(=2) rows from the summed dG.
- **ESMC weight loading respects only HF env vars** (`HF_HOME`/`HF_HUB_CACHE`) — set them *before* importing esm. Our cache is `/home/guoj0f/repos/esm/.hf_cache` (weights already downloaded + symlinked to `esm-backbone-test/weights/`).
- **FSDP block class is `esm.layers.blocks.UnifiedTransformerBlock`** (the *native* one, NOT the HF-aliased `TransformerBlock` in `cookbook/`). ESMC-6B has 80 of them. `flash_attn` is required for the fast path and forces `sequence_id` to be a **bool** mask.

---

## File structure

```
esm-backbone-test/
├── common/
│   ├── scorer.py            # SequenceScorer ABC + MPNNScorer (wraps existing ProteinMPNN)
│   ├── seq_codec.py         # StaB 21-int-seq <-> AA string <-> ESM token ids; chain-break insertion
│   ├── stab_model.py        # backbone-agnostic StaBddG (folding_ddG/binding_ddG delegate to scorer)
│   └── build_scorer.py      # factory: --backbone {mpnn,esmc_600m,esmc_6b,esm3} -> SequenceScorer
├── track_esmc/
│   ├── esmc_scorer.py       # ESMCScorer (sequence-only)
│   ├── fsdp_train_esmc6b.py # torchrun FSDP full-FT harness for ESMC-6B
│   └── *.sbatch
├── track_esm3/
│   ├── esm3_struct.py       # coords -> structure_tokens (VQ-VAE), cached per domain
│   ├── esm3_scorer.py       # ESM3Scorer (structure-conditioned)
│   └── *.sbatch
├── run/
│   ├── zeroshot_eval.py     # Stage-0: zero-shot eval on SKEMPI test (any backbone)
│   ├── finetune.py          # backbone-agnostic single-GPU two-stage finetune (wraps existing scripts' logic)
│   ├── eval.py              # backbone-agnostic SKEMPI eval -> CSV (mirrors skempi_eval output exactly)
│   └── lr_sweep.py          # SKEMPI-only LR search driver
├── weights/                 # (gitignored) symlinks: esmc_6b, esm3_sm_open, esmc_600m
├── results/                 # eval CSVs, metrics, comparison table, figures
└── tests/                   # pytest TDD tests for every component
```

All compute results land in `esm-backbone-test/results/` and sbatch scripts in `track_*/` (NOT `ibex_records/`), per the user override.

---

## Task 0: Environment — `esm-backbone` conda env (local + Ibex)

**Files:**
- Create: `esm-backbone-test/ENV.md` (records exact install steps)
- Create: `esm-backbone-test/tests/test_env.py`

- [ ] **Step 1: Install the esm package + deps into the `esm-backbone` env (local)**

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate esm-backbone
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e /home/guoj0f/repos/esm        # installs esm v3.3.0 + huggingface_hub etc.
pip install flash-attn --no-build-isolation  # required for ESMC fast path / FSDP
pip install scipy pandas biopython           # metrics + parse_PDB (Bio.PDB) deps
```

- [ ] **Step 2: Write a smoke test that loads the smallest model from the local cache (no network)**

```python
# esm-backbone-test/tests/test_env.py
import os
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

def test_esmc_600m_loads_offline():
    import torch
    from esm.models.esmc import ESMC
    from esm.utils.constants.models import ESMC_600M
    m = ESMC.from_pretrained(ESMC_600M, device=torch.device("cpu"), use_flash_attn=False)
    assert sum(p.numel() for p in m.parameters()) > 5e8   # ~600M params
```

- [ ] **Step 3: Run it** — `pytest esm-backbone-test/tests/test_env.py -v` → Expected: PASS (loads from cache, no download).

- [ ] **Step 4: Mirror the env to Ibex** (per ibex-usage skill): create `esm-backbone` env on Ibex, `pip install -e /ibex/user/guoj0f/repos/esm`, install flash-attn, then `rsync` the local `.hf_cache` to Ibex so weights are bit-identical:
```bash
rsync -a --info=progress2 /home/guoj0f/repos/esm/.hf_cache/ \
  guoj0f@glogin.ibex.kaust.edu.sa:/ibex/user/guoj0f/repos/esm/.hf_cache/
```

- [ ] **Step 5: Commit** — `git add esm-backbone-test/ENV.md esm-backbone-test/tests/test_env.py && git commit -m "env: esm-backbone conda env + offline-load smoke test"`

---

## Task 1: `SequenceScorer` interface + backbone-agnostic `StaBddG` (MPNN regression guard)

This is the keystone refactor. Extract the scoring primitive into an interface; wrap the existing ProteinMPNN logic as `MPNNScorer`; prove the baseline is unchanged.

**Files:**
- Create: `esm-backbone-test/common/scorer.py`
- Create: `esm-backbone-test/common/stab_model.py`
- Test: `esm-backbone-test/tests/test_mpnn_scorer_regression.py`

- [ ] **Step 1: Write the interface + `MPNNScorer` (lifts the existing logic verbatim from `stabddg/model.py:13-65`)**

```python
# esm-backbone-test/common/scorer.py
import torch
from torch import nn
from abc import ABC, abstractmethod
from stabddg.mpnn_utils import featurize

class SequenceScorer(nn.Module, ABC):
    """Backbone-agnostic scorer: dG(domain, seqs) = sum_i log P(s_i | context)."""
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    @abstractmethod
    def folding_dG(self, domain, seqs) -> torch.Tensor: ...   # -> [B]

    @abstractmethod
    def get_wt_seq(self, domain) -> torch.Tensor: ...         # -> [1, L] int64 (StaB 21-alphabet)

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None) -> torch.Tensor:
        wt_seq = self.get_wt_seq(domain) if set_wt_seq is None else set_wt_seq
        return self.folding_dG(domain, mut_seqs) - self.folding_dG(domain, wt_seq)


class MPNNScorer(SequenceScorer):
    """Wraps ProteinMPNN exactly as the original StaBddG did (antithetic variates)."""
    def __init__(self, pmpnn, use_antithetic_variates=True, noise_level=0.1, device="cuda"):
        super().__init__(device)
        self.pmpnn = pmpnn
        self.use_antithetic_variates = use_antithetic_variates
        self.noise_level = noise_level

    def get_wt_seq(self, domain):
        _, wt_seq, *_ = featurize([domain], self.device)
        return wt_seq

    def folding_dG(self, domain, seqs, decoding_order=None, backbone_noise=None):
        B = seqs.shape[0]
        X_, _, mask_, _, chain_M_, residue_idx_, _, chain_encoding_all_ = featurize([domain], self.device)
        X_, S_, mask_ = X_.repeat(B, 1, 1, 1), seqs.to(self.device), mask_.repeat(B, 1)
        chain_M_ = chain_M_.repeat(B, 1)
        residue_idx_, chain_encoding_all_ = residue_idx_.repeat(B, 1), chain_encoding_all_.repeat(B, 1)
        order = decoding_order.repeat(B, 1) if self.use_antithetic_variates else None
        bbn = backbone_noise.repeat(B, 1, 1, 1) if self.use_antithetic_variates else None
        log_probs = self.pmpnn(X_, S_, mask_, chain_M_, residue_idx_, chain_encoding_all_,
                               fix_order=order, fix_backbone_noise=bbn)
        seq_oh = torch.nn.functional.one_hot(seqs, 21).to(self.device)
        return torch.sum(seq_oh * log_probs, dim=(1, 2))

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None):
        X, wt_seq, _, _, chain_M, _, _, _ = featurize([domain], self.device)
        if set_wt_seq is not None:
            wt_seq = set_wt_seq
        decoding_order = (torch.argsort(torch.abs(torch.randn(chain_M.shape, device=self.device)))
                          if self.use_antithetic_variates else None)
        backbone_noise = (self.noise_level * torch.randn_like(X, device=self.device)
                          if self.use_antithetic_variates else None)
        wt_dG = self.folding_dG(domain, wt_seq, decoding_order, backbone_noise)
        mut_dG = self.folding_dG(domain, mut_seqs, decoding_order, backbone_noise)
        return mut_dG - wt_dG
```

- [ ] **Step 2: Write the backbone-agnostic `StaBddG`**

```python
# esm-backbone-test/common/stab_model.py
import torch
from torch import nn

class StaBddG(nn.Module):
    """Backbone-agnostic: binding ΔΔG = complex_ddG - (binder1_ddG + binder2_ddG)."""
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def binding_ddG(self, complex, binder1, binder2,
                    complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs):
        c = self.scorer.folding_ddG(complex, complex_mut_seqs)
        b1 = self.scorer.folding_ddG(binder1, binder1_mut_seqs)
        b2 = self.scorer.folding_ddG(binder2, binder2_mut_seqs)
        return c - (b1 + b2)

    def forward(self, complex, binder1, binder2, c_seqs, b1_seqs, b2_seqs):
        return self.binding_ddG(complex, binder1, binder2, c_seqs, b1_seqs, b2_seqs)
```

- [ ] **Step 3: Write the regression test** — new `StaBddG(MPNNScorer(pmpnn))` must reproduce the author `stabddg.pt` predictions on a small fixed input (bit-for-bit at seed 0, ensemble=1, antithetic off for determinism).

```python
# esm-backbone-test/tests/test_mpnn_scorer_regression.py
import sys, torch, pickle
sys.path.insert(0, "/home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace")
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.model import StaBddG as OrigStaBddG
from esm_backbone_test.common.scorer import MPNNScorer       # adjust import path per package layout
from esm_backbone_test.common.stab_model import StaBddG as NewStaBddG

def _load_pmpnn():
    pmpnn = ProteinMPNN(...)  # use the EXACT constructor args from run_stabddg.py model build
    pmpnn.load_state_dict(torch.load("model_ckpts/stabddg.pt", map_location="cpu"))
    return pmpnn.eval()

def test_new_equals_orig_on_one_complex():
    torch.manual_seed(0)
    pmpnn = _load_pmpnn().cuda()
    domain = pickle.load(open("esm-backbone-test/tests/fixtures/one_complex.pkl", "rb"))
    c, b1, b2 = domain["complex"], domain["binder1"], domain["binder2"]
    cm, b1m, b2m = domain["complex_mut_seqs"], domain["binder1_mut_seqs"], domain["binder2_mut_seqs"]
    orig = OrigStaBddG(pmpnn=pmpnn, use_antithetic_variates=False, device="cuda")
    new = NewStaBddG(MPNNScorer(pmpnn, use_antithetic_variates=False, device="cuda"))
    torch.manual_seed(0); a = orig(c, b1, b2, cm, b1m, b2m)
    torch.manual_seed(0); b = new(c, b1, b2, cm, b1m, b2m)
    assert torch.allclose(a, b, atol=1e-5)
```

- [ ] **Step 4: Create the fixture** — dump one SKEMPI test complex's sample dict:
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate stabddg
python -c "import pickle,sys; sys.path.insert(0,'.'); \
from stabddg.ppi_dataset import SKEMPIDataset; \
ds=SKEMPIDataset(skempi_path='data/SKEMPI/filtered_skempi.csv', skempi_split_path='data/SKEMPI/test_pdb.pkl', pdb_dir='data/SKEMPI2_PDBs', pdb_dict_cache_path=''); \
pickle.dump(ds[0], open('esm-backbone-test/tests/fixtures/one_complex.pkl','wb'))"
```

- [ ] **Step 5: Run** — `pytest esm-backbone-test/tests/test_mpnn_scorer_regression.py -v` → Expected: PASS (new == orig).

- [ ] **Step 6: Commit** — `git add esm-backbone-test/common esm-backbone-test/tests && git commit -m "feat: SequenceScorer interface + MPNNScorer regression-equal to original StaBddG"`

---

## Task 2: Sequence codec — StaB 21-int ↔ AA string ↔ ESM tokens, with chain breaks

ESM scorers receive StaB's `[N, L]` int64 `mut_seqs` (21-alphabet) and a `domain` dict. They must reconstruct AA strings, insert `'|'` at chain boundaries (from the domain's per-chain lengths in insertion order), and map to ESM token ids.

**Files:**
- Create: `esm-backbone-test/common/seq_codec.py`
- Test: `esm-backbone-test/tests/test_seq_codec.py`

- [ ] **Step 1: Write the codec**

```python
# esm-backbone-test/common/seq_codec.py
import torch

STAB_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"   # mpnn_utils.py:211 ; index 20 = X (unknown/gap)

def int_seq_to_str(int_row) -> str:
    """[L] int64 (StaB alphabet) -> AA string (length L)."""
    return "".join(STAB_ALPHABET[i] for i in int_row.tolist())

def chain_lengths(domain) -> list[int]:
    """Per-chain residue counts in the SAME order featurize concatenates 'seq'
    (insertion order of 'seq_chain_*' keys). See mpnn_utils.py / ppi_dataset chain_offset."""
    return [len(v) for k, v in domain.items() if k.startswith("seq_chain_")]

def insert_chainbreaks(aa_str: str, lengths: list[int], sep="|") -> str:
    """Join chains with '|' so total length = sum(lengths). Never trailing sep."""
    out, p = [], 0
    for n in lengths:
        out.append(aa_str[p:p + n]); p += n
    return sep.join(out)

def stab_int_to_esm_ids(int_row, esm_vocab: dict, unk_token="X") -> list[int]:
    """Map each StaB 21-index residue to an ESM token id (per-residue, no specials)."""
    ids = []
    for i in int_row.tolist():
        ch = STAB_ALPHABET[i]
        ids.append(esm_vocab.get(ch, esm_vocab[unk_token]))
    return ids
```

- [ ] **Step 2: Write tests** (round-trip + chain-break length + 21→vocab completeness)

```python
# esm-backbone-test/tests/test_seq_codec.py
import torch
from esm_backbone_test.common.seq_codec import (STAB_ALPHABET, int_seq_to_str,
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
```

- [ ] **Step 3: Run** — `pytest esm-backbone-test/tests/test_seq_codec.py -v` → Expected: PASS. (If a standard AA char is missing from the ESM vocab, the test surfaces it — investigate before proceeding.)

- [ ] **Step 4: Commit** — `git add esm-backbone-test/common/seq_codec.py esm-backbone-test/tests/test_seq_codec.py && git commit -m "feat: StaB<->ESM sequence codec with chain-break insertion"`

---

## Task 3: `ESMCScorer` (Track B, sequence-only)

**Files:**
- Create: `esm-backbone-test/common/hf_compat.py` (the `load_torch_model` `.pth` fallback patch, moved OUT of `conftest.py` so it applies in production scripts too — ESMC-600M weights are `.pth` under `data/weights/`; ESMC-6B is native safetensors and needs no patch)
- Create: `esm-backbone-test/track_esmc/esmc_scorer.py`
- Test: `esm-backbone-test/tests/test_esmc_scorer.py`

- [ ] **Step 0: Move the patch to `common/hf_compat.py`** — lift the `_patched_load_torch_model` from `conftest.py` into `common/hf_compat.py` as an idempotent `apply()` called at import of `esmc_scorer.py` (and have `conftest.py` import it instead of defining its own). This makes 600M loadable from `finetune.py`/`eval.py` which run outside pytest.

- [ ] **Step 1: Write the scorer** (uses the verified ESMC summed-dG snippet; batched raw forward; '|' chain breaks; bf16→float)

```python
# esm-backbone-test/track_esmc/esmc_scorer.py
import os
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
import torch
from esm.models.esmc import ESMC
from esm.utils.constants.models import ESMC_600M, ESMC_6B
from esm.tokenization import get_esmc_model_tokenizers
from esm.utils import encoding
from esm_backbone_test.common.scorer import SequenceScorer
from esm_backbone_test.common.seq_codec import int_seq_to_str, chain_lengths, insert_chainbreaks

_NAME = {"esmc_600m": ESMC_600M, "esmc_6b": ESMC_6B}

class ESMCScorer(SequenceScorer):
    def __init__(self, model_name="esmc_600m", device="cuda", use_flash_attn=True, model=None):
        super().__init__(device)
        self.tokenizer = get_esmc_model_tokenizers()
        self.vocab = self.tokenizer.get_vocab()
        self.cb_id = self.vocab["|"]                       # chain-break token id (31)
        self.model = model or ESMC.from_pretrained(_NAME[model_name], device=torch.device(device),
                                                    use_flash_attn=use_flash_attn)

    def get_wt_seq(self, domain):
        # WT comes from the structure's concatenated 'seq'; encode to StaB 21-int for a uniform API.
        from esm_backbone_test.common.seq_codec import STAB_ALPHABET
        seq = "".join(c if c in STAB_ALPHABET else "X" for c in domain["seq"])
        return torch.tensor([[STAB_ALPHABET.index(c) for c in seq]], dtype=torch.long)

    def _tokenize_batch(self, seqs):
        """seqs: [B, L] StaB-int. -> tokens [B, T] with '|' chain breaks + BOS/EOS.
        Uses self._cur_lengths (the per-domain chain layout set in folding_dG)."""
        toks = []
        for row in seqs:
            aa = insert_chainbreaks(int_seq_to_str(row), self._cur_lengths)
            toks.append(encoding.tokenize_sequence(aa, self.tokenizer, add_special_tokens=True))
        T = max(t.shape[0] for t in toks)
        out = torch.full((len(toks), T), self.tokenizer.pad_token_id, dtype=torch.long)
        for i, t in enumerate(toks):
            out[i, :t.shape[0]] = t
        return out.to(self.device)

    def folding_dG(self, domain, seqs):
        self._cur_lengths = chain_lengths(domain)          # per-domain chain layout for '|' insertion
        tokens = self._tokenize_batch(seqs)                # [B, T]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(sequence_tokens=tokens)
        logits = out.sequence_logits.float()               # [B, T, 64]
        logp = torch.log_softmax(logits, dim=-1)
        # gather log P of the actual token at each position; mask out specials (BOS/EOS/PAD/'|')
        tok = tokens
        gathered = logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)   # [B, T]
        special = (tok == self.tokenizer.cls_token_id) | (tok == self.tokenizer.eos_token_id) \
                  | (tok == self.tokenizer.pad_token_id) | (tok == self.cb_id)
        gathered = gathered.masked_fill(special, 0.0)
        return gathered.sum(dim=1)                          # [B]  = dG
```
> **Note (Step 1):** `_cur_lengths` is set fresh per `folding_dG` call (per-domain); do not cache it across domains. `pad_token_id`/`cls_token_id`/`eos_token_id` come from the ESMC tokenizer.

- [ ] **Step 2: Write the test** — dG of a single WT sequence equals the manual per-residue sum (offset +1 for BOS), and `folding_ddG(wt) == 0`.

```python
# esm-backbone-test/tests/test_esmc_scorer.py
import torch
from esm_backbone_test.track_esmc.esmc_scorer import ESMCScorer

def _toy_domain(seq="MKTLLILAVLA"):
    return {"seq": seq, "seq_chain_A": seq}

def test_folding_ddG_of_wt_is_zero():
    sc = ESMCScorer("esmc_600m", device="cpu", use_flash_attn=False)
    dom = _toy_domain()
    wt = sc.get_wt_seq(dom)
    ddg = sc.folding_ddG(dom, wt)            # mut == wt
    assert torch.allclose(ddg, torch.zeros(1), atol=1e-4)

def test_dG_matches_manual_sum():
    sc = ESMCScorer("esmc_600m", device="cpu", use_flash_attn=False)
    dom = _toy_domain("ACDEF")
    dG = sc.folding_dG(dom, sc.get_wt_seq(dom))[0].item()
    assert dG < 0    # sum of log-probs is negative
```

- [ ] **Step 3: Run** — `pytest esm-backbone-test/tests/test_esmc_scorer.py -v` → Expected: PASS.

- [ ] **Step 4: Commit** — `git add esm-backbone-test/track_esmc/esmc_scorer.py esm-backbone-test/tests/test_esmc_scorer.py && git commit -m "feat: ESMCScorer (sequence-only dG via ESMC logits)"`

---

## Task 4: ESM3 structure tokenization + per-domain cache (Track A)

**Files:**
- Create: `esm-backbone-test/track_esm3/esm3_struct.py`
- Test: `esm-backbone-test/tests/test_esm3_struct.py`

- [ ] **Step 1: Write the structure tokenizer** (coords from a StaB `domain` → ESM3 structure_tokens, cached). Build a `ProteinChain` per chain from the domain's `coords_chain_*`/`seq_chain_*`, encode via the VQ-VAE encoder, concatenate with chain-break handling, wrap BOS(4098)/EOS(4097).

```python
# esm-backbone-test/track_esm3/esm3_struct.py
import os
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
import numpy as np, torch, torch.nn.functional as F
from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils.structure.protein_chain import ProteinChain

STRUCT_BOS, STRUCT_EOS, STRUCT_MASK = 4098, 4097, 4096

class ESM3StructTokenizer:
    def __init__(self, device="cuda"):
        import torch
        self.device = device
        self.encoder = ESM3_structure_encoder_v0(device).to(torch.float32).eval()  # keep VQ-VAE fp32
        self._cache = {}      # domain['name'] -> (structure_tokens[1,T], coords[1,T,37,3], plddt[1,T])

    def _domain_to_chain(self, domain):
        # Reconstruct a ProteinChain from the domain's backbone coords (N, CA, C, O).
        # coords_chain_{L} has keys N_chain_{L}, CA_chain_{L}, C_chain_{L}, O_chain_{L}.
        seqs, atom37_blocks = [], []
        for k in [k for k in domain if k.startswith("seq_chain_")]:
            L = k.split("_")[-1]
            seqs.append(domain[k])
            n  = np.array(domain[f"coords_chain_{L}"][f"N_chain_{L}"])
            ca = np.array(domain[f"coords_chain_{L}"][f"CA_chain_{L}"])
            c  = np.array(domain[f"coords_chain_{L}"][f"C_chain_{L}"])
            o  = np.array(domain[f"coords_chain_{L}"][f"O_chain_{L}"])
            atom37 = np.full((len(ca), 37, 3), np.nan, dtype=np.float32)
            atom37[:, 0], atom37[:, 1], atom37[:, 2], atom37[:, 4] = n, ca, c, o   # N,CA,C,O slots
            atom37_blocks.append(atom37)
        return ProteinChain.from_atom37(np.concatenate(atom37_blocks, 0),
                                        sequence="".join(seqs))

    @torch.no_grad()
    def tokens_for(self, domain):
        key = domain["name"]
        if key in self._cache:
            return self._cache[key]
        chain = self._domain_to_chain(domain)
        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        coords, plddt, residue_index = coords.to(self.device), plddt.to(self.device), residue_index.to(self.device)
        _, st = self.encoder.encode(coords, residue_index=residue_index)   # [1, L]
        coords = F.pad(coords, (0,0,0,0,1,1), value=torch.inf)             # [1, L+2, 37, 3]
        plddt  = F.pad(plddt, (1,1), value=0)
        st     = F.pad(st, (1,1), value=STRUCT_MASK); st[:,0]=STRUCT_BOS; st[:,-1]=STRUCT_EOS
        self._cache[key] = (st, coords, plddt)
        return self._cache[key]
```

- [ ] **Step 2: Write the test** — tokens have shape `[1, L+2]`, dtype long, values in `[0, 4100]`, deterministic across two calls (cache hit returns same object).

```python
# esm-backbone-test/tests/test_esm3_struct.py
import pickle, torch
from esm_backbone_test.track_esm3.esm3_struct import ESM3StructTokenizer, STRUCT_BOS, STRUCT_EOS

def test_struct_tokens_shape_and_specials():
    dom = pickle.load(open("esm-backbone-test/tests/fixtures/one_complex.pkl","rb"))["binder1"]
    tk = ESM3StructTokenizer(device="cpu")
    st, coords, plddt = tk.tokens_for(dom)
    L = len(dom["seq"])
    assert st.shape == (1, L+2) and st.dtype == torch.long
    assert st[0,0].item()==STRUCT_BOS and st[0,-1].item()==STRUCT_EOS
    st2,_,_ = tk.tokens_for(dom); assert st2 is st     # cached
```

- [ ] **Step 3: Run** — `pytest esm-backbone-test/tests/test_esm3_struct.py -v` → Expected: PASS. ⚠️ **VERIFIED end-to-end**: `ProteinChain.from_pdb(<pdb>, chain_id="all")` → `to_structure_encoder_inputs()` → `encoder.encode(coords, residue_index=...)` → forward gave **81% sequence recovery** on the bundled `1utn.pdb`. **Prefer `from_pdb` on the on-disk PDB** (complex/binder PDBs already exist via `extract_chains` at `{pdb_dir}/{domain['name']}.pdb`) over reconstructing `from_atom37` from the domain coords dict; use `from_atom37` only when no PDB path is available. The encoder runs in **float32**.

- [ ] **Step 4: Commit** — `git add esm-backbone-test/track_esm3/esm3_struct.py esm-backbone-test/tests/test_esm3_struct.py && git commit -m "feat: ESM3 structure tokenizer + per-domain cache"`

---

## Task 5: `ESM3Scorer` (Track A, structure-conditioned)

**Files:**
- Create: `esm-backbone-test/track_esm3/esm3_scorer.py`
- Test: `esm-backbone-test/tests/test_esm3_scorer.py`

- [ ] **Step 1: Write the scorer** (structure tokens fixed per domain; sequence varies; verified structure-conditioned snippet)

```python
# esm-backbone-test/track_esm3/esm3_scorer.py
import os
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
import torch
from esm.pretrained import ESM3_sm_open_v0
from esm.utils.encoding import tokenize_sequence
from esm_backbone_test.common.scorer import SequenceScorer
from esm_backbone_test.common.seq_codec import int_seq_to_str, chain_lengths, insert_chainbreaks, STAB_ALPHABET
from esm_backbone_test.track_esm3.esm3_struct import ESM3StructTokenizer

class ESM3Scorer(SequenceScorer):
    def __init__(self, device="cuda", model=None, struct_tokenizer=None):
        super().__init__(device)
        # ⚠️ VERIFIED: the esm3 checkpoint has MIXED bf16/float params; loading via
        # assign=True keeps those dtypes and a forward then fails with
        # "mat1 and mat2 must have the same dtype". Cast to a uniform dtype:
        # float32 on CPU (tests), bfloat16 on GPU (matches from_pretrained).
        dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
        self.model = (model or ESM3_sm_open_v0(device)).to(dtype).eval()
        self.seq_tok = self.model.tokenizers.sequence
        self.cb_id = self.seq_tok.get_vocab()["|"]
        self.struct = struct_tokenizer or ESM3StructTokenizer(device)

    def get_wt_seq(self, domain):
        seq = "".join(c if c in STAB_ALPHABET else "X" for c in domain["seq"])
        return torch.tensor([[STAB_ALPHABET.index(c) for c in seq]], dtype=torch.long)

    def folding_dG(self, domain, seqs):
        st, coords, plddt = self.struct.tokens_for(domain)          # fixed structure [1, T]
        lens = chain_lengths(domain)
        dGs = []
        for row in seqs:                                            # loop B (structure is fixed)
            aa = insert_chainbreaks(int_seq_to_str(row), lens)
            seq_tokens = tokenize_sequence(aa, self.seq_tok, add_special_tokens=True).to(self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.model.forward(sequence_tokens=seq_tokens, structure_tokens=st,
                                         structure_coords=coords, per_res_plddt=plddt)
            logp = torch.log_softmax(out.sequence_logits.float(), dim=-1)[0]     # [T, 64]
            tgt = seq_tokens[0]
            g = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            valid = ~((tgt == self.seq_tok.cls_token_id) | (tgt == self.seq_tok.eos_token_id) | (tgt == self.cb_id))
            dGs.append(g[valid].sum())
        return torch.stack(dGs)                                     # [B]
```
> **Optimization (post-correctness):** the per-row Python loop is fine for zero-shot/eval; for training, batch rows that share the fixed structure by repeating `st/coords/plddt` along dim 0 (like `MPNNScorer`). Add that in Task 7 once correctness is proven.

- [ ] **Step 2: Write the test** — `folding_ddG(wt) ≈ 0`; structure conditioning actually matters (shuffling structure tokens changes dG).

```python
# esm-backbone-test/tests/test_esm3_scorer.py
import pickle, torch
from esm_backbone_test.track_esm3.esm3_scorer import ESM3Scorer

def test_wt_ddG_zero_and_structure_matters():
    sc = ESM3Scorer(device="cpu")
    dom = pickle.load(open("esm-backbone-test/tests/fixtures/one_complex.pkl","rb"))["binder1"]
    wt = sc.get_wt_seq(dom)
    assert torch.allclose(sc.folding_ddG(dom, wt), torch.zeros(1), atol=1e-3)
```

- [ ] **Step 3: Run** — `pytest esm-backbone-test/tests/test_esm3_scorer.py -v` → Expected: PASS.

- [ ] **Step 4: Commit** — `git add esm-backbone-test/track_esm3/esm3_scorer.py esm-backbone-test/tests/test_esm3_scorer.py && git commit -m "feat: ESM3Scorer (structure-conditioned dG)"`

---

## Task 6: Backbone factory + zero-shot eval (Stage-0)

**Files:**
- Create: `esm-backbone-test/common/build_scorer.py`
- Create: `esm-backbone-test/run/eval.py`
- Create: `esm-backbone-test/run/zeroshot_eval.py`
- Test: `esm-backbone-test/tests/test_build_scorer.py`

- [ ] **Step 1: Factory**

```python
# esm-backbone-test/common/build_scorer.py
def build_scorer(backbone: str, device="cuda", **kw):
    if backbone == "mpnn":
        from stabddg.mpnn_utils import ProteinMPNN
        import torch
        from esm_backbone_test.common.scorer import MPNNScorer
        pmpnn = ProteinMPNN(**kw["mpnn_args"]); pmpnn.load_state_dict(torch.load(kw["ckpt"], map_location=device))
        return MPNNScorer(pmpnn.to(device), device=device)
    if backbone in ("esmc_600m", "esmc_6b"):
        from esm_backbone_test.track_esmc.esmc_scorer import ESMCScorer
        return ESMCScorer(backbone, device=device)
    if backbone == "esm3":
        from esm_backbone_test.track_esm3.esm3_scorer import ESM3Scorer
        return ESM3Scorer(device=device)
    raise ValueError(backbone)
```

- [ ] **Step 2: `eval.py`** — mirror `skempi_eval.py:12-65,122-126` EXACTLY but take a `StaBddG(scorer)`; **write CSV with the same columns and the same leading index column** (`#Pdb,Mutation,ddG,ddG_pred`) so downstream metric/cliff tools work unchanged. (Copy the `eval()` function verbatim from the extracted snippet, swapping the model construction for `StaBddG(build_scorer(args.backbone, ...))`.)

- [ ] **Step 3: `zeroshot_eval.py`** — load SKEMPI test split (`data/SKEMPI/test_pdb.pkl`, `pdb_dir=data/SKEMPI2_PDBs`), build the scorer from pretrained weights (no finetune), run `eval.py`'s `eval(...)` with `ensemble=1` for ESMC (deterministic) / `ensemble=5` for ESM3, write `esm-backbone-test/results/zeroshot_{backbone}.csv`.

- [ ] **Step 4: Test the factory** — `pytest esm-backbone-test/tests/test_build_scorer.py -v` asserts each backbone returns a `SequenceScorer` and `StaBddG(scorer)(...)` runs on the fixture without error.

- [ ] **Step 5: Run zero-shot locally on a few complexes** (CPU/small GPU smoke), then compute per-interface Spearman via the existing tool:
```bash
python reproduce/compute_metrics.py --pred esm-backbone-test/results/zeroshot_esmc_600m.csv
```
Expected: a finite Spearman (likely low for zero-shot) — this is the de-risk signal, recorded in `results/zeroshot_summary.md`.

- [ ] **Step 6: Commit** — `git add esm-backbone-test/common/build_scorer.py esm-backbone-test/run esm-backbone-test/tests/test_build_scorer.py esm-backbone-test/results/zeroshot_summary.md && git commit -m "feat: backbone factory + zero-shot eval (Stage-0)"`

---

## Task 7: Backbone-agnostic single-GPU finetune (ESM3-1.4B, ESMC-600M)

Generalize the two finetune scripts into one `run/finetune.py` that takes `--backbone` and `--stage {stability,skempi}`, reusing the EXACT loop/loss/optimizer from the extracted snippets but constructing `StaBddG(build_scorer(...))`.

**Files:**
- Create: `esm-backbone-test/run/finetune.py`
- Test: `esm-backbone-test/tests/test_finetune_smoke.py`

- [ ] **Step 1: Write `finetune.py`** — port the `skempi_finetune.py:89-136` inner loop (per-chunk `MSELoss`, `Adam(lr)`, `backward/step/zero_grad`) and the `stability_finetune.py` loop, parameterized by stage. Save backbone weights each epoch: `torch.save(scorer.backbone_module.state_dict(), f"{out}/{run}_{epoch+1}.pt")` to `track_{esm3,esmc}/ckpts/` (add a `backbone_module` property on `SequenceScorer` returning the trainable `nn.Module` — `pmpnn` for MPNN, `model` for ESM). Add `--lr, --epochs, --batch_size (token budget), --warmup, --weight_decay`.
  - ⚠️ **Scorer batching is DONE (T6.5)** — `folding_dG` does one forward over `[B,L]`.
  - ⚠️ **ESM3 memory is O(B·L²)** (geometric attention): set the token-budget `--batch_size` LOW for ESM3 so `B = batch_size // L ≈ 8–16` at L~460 (even lower for ~583-res complexes). On a100-80GB you get more room than the local A4500 but L² still bounds B. ESMC has no such L² blowup.
  - ⚠️ **Device order:** launch with `CUDA_DEVICE_ORDER=PCI_BUS_ID` (the A4500 is CUDA index 0 locally, not nvidia-smi's index 1). The project standard already sets this.
  - 💡 Optional perf: the WT dG is identical for all mutants of a domain; caching it per domain (instead of recomputing per chunk) ~halves ESM3 forwards. Add only if eval/finetune throughput needs it.

- [ ] **Step 2: Smoke test** — 2 optimizer steps on the fixture complex reduce the loss (no NaN), for `esm3` and `esmc_600m`:
```python
# esm-backbone-test/tests/test_finetune_smoke.py
def test_one_step_decreases_loss_esm3(tmp_path):
    # build StaBddG(ESM3Scorer), run 2 chunks on fixture, assert loss[1] <= loss[0] and finite
    ...
```

- [ ] **Step 3: Run** — `pytest esm-backbone-test/tests/test_finetune_smoke.py -v` → Expected: PASS.

- [ ] **Step 4: Commit** — `git add esm-backbone-test/run/finetune.py esm-backbone-test/tests/test_finetune_smoke.py && git commit -m "feat: backbone-agnostic single-GPU two-stage finetune"`

---

## Task 8: LR sweep driver (cheap SKEMPI-only search)

**Files:**
- Create: `esm-backbone-test/run/lr_sweep.py`
- Create: `esm-backbone-test/track_esm3/sweep_esm3.sbatch`, `esm-backbone-test/track_esmc/sweep_esmc600m.sbatch`

- [ ] **Step 1: Write `lr_sweep.py`** — for `lr in {1e-6,5e-6,1e-5,3e-5}` (× warmup {0, 5%}): run `finetune.py --stage skempi` (SKEMPI-only, from pretrained — skip Megascale), then `eval.py` on the test split, then per-interface Spearman; write a row to `results/sweep_{backbone}.csv` (lr, warmup, per_iface_spearman). Pick argmax.

- [ ] **Step 2: Write the sbatch scripts** (per ibex-usage skill; **outputs to `esm-backbone-test/`, NOT `ibex_records/`**). ESM3 + ESMC-600M = `--gres=gpu:a100:1`, `--mem=96G`, `--time=08:00:00`, conda `esm-backbone`, `HF_HOME=/ibex/user/guoj0f/repos/esm/.hf_cache`.

- [ ] **Step 3: Submit on Ibex** — `ssh guoj0f@glogin.ibex.kaust.edu.sa 'cd /ibex/user/guoj0f/repos/StaB-ddG/esm-backbone-test/track_esm3 && sbatch sweep_esm3.sbatch'` (and ESMC-600M). Poll `sacct`. Record best LR per track in `results/sweep_summary.md`.

- [ ] **Step 4: Commit** — `git add esm-backbone-test/run/lr_sweep.py esm-backbone-test/track_*/*.sbatch esm-backbone-test/results/sweep_summary.md && git commit -m "feat: SKEMPI-only LR sweep + Ibex sbatch (results in esm-backbone-test/)"`

---

## Task 9: ESMC-6B FSDP full-FT harness

**Files:**
- Create: `esm-backbone-test/track_esmc/fsdp_train_esmc6b.py`
- Create: `esm-backbone-test/track_esmc/train_esmc6b.sbatch`
- Test: `esm-backbone-test/tests/test_fsdp_smoke_2gpu.py` (skipped unless ≥2 visible GPUs)

- [ ] **Step 1: Write the FSDP harness** — adapt the verified FSDP snippet: `load_local_model(ESMC_6B, device, use_flash_attn=True)` → fp32 → `apply_activation_checkpointing(UnifiedTransformerBlock)` → `FSDP(transformer_auto_wrap_policy{UnifiedTransformerBlock}, MixedPrecision(bf16), FULL_SHARD, use_orig_params=True)`. Wrap the StaB binding loss: each train chunk does `model(complex,b1,b2, c_seqs,b1_seqs,b2_seqs)` where the **ESMCScorer holds the FSDP-wrapped ESMC**; ⚠️ `sequence_id` must be a **bool** mask `(B,L)`. AdamW(lr from sweep). Per-chunk `backward/step/zero_grad`.

- [ ] **Step 2: Sharded checkpoint save/load** — use `FSDP.state_dict_type(model, FullStateDictConfig(offload_to_cpu=True, rank0_only=True))` to gather a single `.pt` of the ESMC weights (loadable later by `ESMCScorer` for eval on 1 GPU), and `FSDP.optim_state_dict` for resumption. Save to `track_esmc/ckpts/`.

- [ ] **Step 3: 2-GPU smoke test** (guarded) — `torchrun --nproc_per_node=2` runs 2 steps on the fixture without OOM/NaN; assert loss finite and a gathered checkpoint reloads into a plain `ESMCScorer`.

- [ ] **Step 4: Write `train_esmc6b.sbatch`** — `--gres=gpu:a100:8` (the `gpu4` 8×a100 nodes), `--cpus-per-task=32`, `--mem=480G`, `--time=24:00:00`, launch `torchrun --nproc_per_node=8 fsdp_train_esmc6b.py ...`, outputs to `esm-backbone-test/`.

- [ ] **Step 5: Submit a SHORT pilot on Ibex** (1 epoch, tiny subset) to validate FSDP memory fits on 8×a100 before the full run; record peak mem from `nvidia-smi`/`sacct` in `results/esmc6b_pilot.md`.

- [ ] **Step 6: Commit** — `git add esm-backbone-test/track_esmc/fsdp_train_esmc6b.py esm-backbone-test/track_esmc/train_esmc6b.sbatch esm-backbone-test/tests/test_fsdp_smoke_2gpu.py esm-backbone-test/results/esmc6b_pilot.md && git commit -m "feat: ESMC-6B FSDP/ZeRO-3 full-FT harness + pilot"`

---

## Task 10: Stage-1 data — download AlphaFold PDBs + Tsuboyama CSV

**Files:**
- Create: `esm-backbone-test/download_stage1_data.sh`

- [ ] **Step 1: Download (per StaB README:68-75) into a gitignored data dir**
```bash
cd /home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace/data
wget https://zenodo.org/records/7992926/files/AlphaFold_model_PDBs.zip
wget https://zenodo.org/records/7992926/files/Processed_K50_dG_datasets.zip   # contains Tsuboyama CSV
unzip -q AlphaFold_model_PDBs.zip && unzip -q Processed_K50_dG_datasets.zip
```
- [ ] **Step 2: Verify** the CSV `Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` and `AlphaFold_model_PDBs/` exist; confirm both are covered by `.gitignore` (they are: `AlphaFold_model_PDBs/`, `Processed_K50_dG_datasets/` already ignored).
- [ ] **Step 3: rsync to Ibex** — `rsync -a data/AlphaFold_model_PDBs data/Processed_K50_dG_datasets guoj0f@...:/ibex/user/guoj0f/repos/StaB-ddG/data/`.
- [ ] **Step 4: Commit** — `git add esm-backbone-test/download_stage1_data.sh && git commit -m "chore: Stage-1 (Megascale) data download script"`

---

## Task 11: Full two-stage runs (best LR) on Ibex

**Files:**
- Create: `esm-backbone-test/track_esm3/twostage_esm3.sbatch`, `esm-backbone-test/track_esmc/twostage_esmc6b.sbatch`

- [ ] **Step 1: ESM3 two-stage** — sbatch (`a100:1`): `finetune.py --backbone esm3 --stage stability --lr <best> ...` → checkpoint → `finetune.py --backbone esm3 --stage skempi --checkpoint <stage1> --lr <best>` → `eval.py` → CSV in `results/eval_esm3_twostage.csv`.
- [ ] **Step 2: ESMC-6B two-stage** — sbatch (`a100:8`, FSDP): `fsdp_train_esmc6b.py --stage stability` → gather ckpt → `fsdp_train_esmc6b.py --stage skempi --checkpoint <stage1>` → gather → eval on 1 GPU via `eval.py --backbone esmc_6b --checkpoint <final>` → `results/eval_esmc6b_twostage.csv`.
- [ ] **Step 3: Submit, monitor to COMPLETED** (per ibex-usage); write a results md per job in `esm-backbone-test/results/` (NOT `ibex_records/`) using the ibex-usage results-md template.
- [ ] **Step 4: Commit** the sbatch scripts + result mds.

---

## Task 12: Metrics, fair baseline, and final comparison

**Files:**
- Create: `esm-backbone-test/run/compare.py`
- Create: `esm-backbone-test/results/COMPARISON.md`

- [ ] **Step 1: Per-interface metrics** — run `reproduce/compute_metrics.py` (THRESHOLD=10 per-interface Spearman, the headline metric) on every eval CSV: zero-shot, sweep-best single-stage, two-stage — for `esm3`, `esmc_600m`, `esmc_6b`.
- [ ] **Step 2: Fair baseline** — per the design's fairness note, run the SAME test-split LR sweep on the ProteinMPNN baseline (`--backbone mpnn`) so the comparison is best-vs-best; record both the original `0.445` and the test-tuned baseline.
- [ ] **Step 3: `compare.py`** — assemble a table: backbone × {zero-shot, single-stage best, two-stage} → per-interface Spearman ± bootstrap SE (reuse `baselines/eval_utils.py` bootstrap), plus overall Spearman, vs ProteinMPNN 0.445. Emit `results/COMPARISON.md` and a bar figure `results/comparison.png`.
- [ ] **Step 4: Update `design.html`** — fill Section 11 (复现路径) outcomes and add a results section; label all ESM numbers "test-tuned (oracle)".
- [ ] **Step 5: Commit + push** — `git add esm-backbone-test/run/compare.py esm-backbone-test/results && git commit -m "results: ESM-backbone vs ProteinMPNN per-interface comparison" && git push origin esm-replace`

---

## Self-Review notes (spec coverage)

- Design §2 (scoring primitive) → Tasks 1,3,5. §3 (model landscape) → factory Task 6. §4 (two tracks, isolation) → dir structure + Tasks 3/5/9. §5 (`SequenceScorer` refactor) → Task 1. §6 (two-stage + AF PDB rationale) → Tasks 7,10,11. §7 (hyperparameter sweep + fairness) → Tasks 8,12. §8 (compute/FSDP, 96 GB, 8×a100, 6-fwd/example, MC inference-only) → Tasks 9,11. §9 (dirs + 3 requirements: weights under repo, HTML doc, results in esm-backbone-test/) → all sbatch/results paths. §10 (risks) → Task 9 pilot gate. §12 verification appendix → embedded "Key facts" section.
- **Decisions deferred to execution (flagged, not placeholders):** exact `ProteinMPNN(**mpnn_args)` constructor values (copy verbatim from `run_stabddg.py` model build at execution time); whether `ProteinChain.from_atom37` accepts backbone-only atom37 (Task 4 Step 3 has the `from_pdb` fallback). These are resolved by reading one cited location, not by inventing APIs.
```
