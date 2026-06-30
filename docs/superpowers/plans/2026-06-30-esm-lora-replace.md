# ESM-LoRA-Replace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace StaB-ddG's ProteinMPNN folding-energy scorer with ESM3 fine-tuned via **LoRA + a per-residue stability head**, keeping StaB's data/two-stage/ΔΔG-loss/eval unchanged, and measure SKEMPI binding-ΔΔG per-interface Spearman vs ProteinMPNN 0.445 and the old full-FT-likelihood ESM3 0.242.

**Architecture:** A new `ESM3StabScorer` (frozen ESM3 + LoRA r=4 + MLP head → per-residue ΔG summed to a domain ΔG) implements the existing `SequenceScorer` interface, so the existing `StaBddG` binding-decomposition, `finetune.py` two-stage loop, and `eval.py` work unchanged. `finetune.py`/`eval.py` get **additive, conditional** parameter-efficiency hooks (freeze base, train only LoRA+head, save/load small adapter checkpoints) that leave all existing backbones bit-for-bit unchanged.

**Tech Stack:** Python 3.11, PyTorch 2.5.1+cu124, `esm` 3.3.0 (EvolutionaryScale), `peft` (LoRA), conda env `esm-backbone`; KAUST Ibex (a100) for training/eval.

## Global Constraints

- **Isolation (hard):** NEVER modify `esm-backbone-test/results/`, `esm-backbone-test/track_esm3/`, or the `--backbone esm3` code path. The old 0.242 run must stay exactly reproducible. New code is purely additive.
- **Branch:** work on `esm-replace` only; new vs old experiment distinguished by **config** (`--backbone esm3_stab` vs `esm3`). Never push to `main`; `git push origin HEAD`.
- **Loss:** StaB's original **ΔΔG MSE only** — no sigmoid scaling, no absolute-ΔG term, no Megascale-ΔG-column changes.
- **LoRA config (verbatim from the reference repo):** `LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=["layernorm_qkv.1", "out_proj"], r=4, lora_dropout=0.15)`.
- **Stability head:** `Linear(1536,1536) → LayerNorm(1536) → ReLU → Linear(1536,1)`, kept in **float32** (numerical stability + dtype-safety, mirroring the existing scorer's `.float()` on logits).
- **`folding_dG`:** per-residue head output **summed** over valid residues (exclude BOS/EOS/`|`) — extensive, so `binding_ddG = ΔΔG_complex − (ΔΔG_b1 + ΔΔG_b2)` stays additive.
- **Backbone key:** `esm3_stab` (deliberately NOT `esm3dg`, to avoid confusion with the released model).
- **Ibex:** a100 only; per-branch path `/ibex/user/guoj0f/StaB-ddG/esm-replace/`; records under `ibex-records/esm-lora-replace/` (sbatch in `sh/`, md at root, pkls in `results/`); commit scripts+md, gitignore `*.out`/`*.err`/`*.pkl`/ckpts.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `esm-backbone-test/track_esm3stab/__init__.py` | package marker | Create |
| `esm-backbone-test/track_esm3stab/esm3_stab_scorer.py` | `ESM3StabScorer` + `ESM3StabilityHead` | Create |
| `esm-backbone-test/common/build_scorer.py` | add `esm3_stab` branch | Modify |
| `esm-backbone-test/run/finetune.py` | conditional param-efficient hooks; `esm3_stab` choice + batch default | Modify |
| `esm-backbone-test/run/eval.py` | `esm3_stab` choice + eval batch default | Modify |
| `esm-backbone-test/tests/test_esm3_stab_scorer.py` | scorer contract + factory (CPU) | Create |
| `esm-backbone-test/tests/test_esm3_stab_finetune.py` | one-step train: base frozen / adapters update; ckpt round-trip (CPU) | Create |
| `ibex-records/esm-lora-replace/sh/*.sh` | sbatch scripts | Create (Tasks 6–9) |
| `ibex-records/esm-lora-replace/*.md` | results md | Create (Tasks 6–9) |

All Python tests run from `esm-backbone-test/` (its `conftest.py` sets `sys.path`). ESM3 loads on **CPU** for tests (a few minutes per load — acceptable, matches existing `test_esm3_scorer.py`).

---

### Task 1: `ESM3StabScorer` — architecture + `folding_dG`

**Files:**
- Create: `esm-backbone-test/track_esm3stab/__init__.py` (empty)
- Create: `esm-backbone-test/track_esm3stab/esm3_stab_scorer.py`
- Test: `esm-backbone-test/tests/test_esm3_stab_scorer.py`

**Interfaces:**
- Consumes: `common.scorer.SequenceScorer`; `common.seq_codec.{STAB_ALPHABET,chain_lengths,insert_chainbreaks,int_seq_to_str}`; `track_esm3.esm3_struct.ESM3StructTokenizer`; `esm.pretrained.ESM3_sm_open_v0`; `esm.utils.encoding.tokenize_sequence`; `peft.{get_peft_model,LoraConfig,TaskType}`.
- Produces: `ESM3StabScorer(device, pdb_dir, lora_rank=4, lora_dropout=0.15)` with `folding_dG(domain, seqs)->[B]`, `get_wt_seq(domain)->[1,L]`, `backbone_module->self`, `freeze_base()`, `trainable_parameters()->list`, `save_adapters(path)`, `load_adapters(path)`; and `ESM3StabilityHead(input_dim=1536, output_dim=1)`.

- [ ] **Step 1: Install `peft` into the `esm-backbone` env**

Run:
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate esm-backbone && pip install "peft>=0.11,<0.18"
python -c "import peft; print('peft', peft.__version__)"
```
Expected: prints a `peft` version, no error.

- [ ] **Step 2: Write the failing test**

Create `esm-backbone-test/tests/test_esm3_stab_scorer.py`:
```python
"""test_esm3_stab_scorer.py — ESM3StabScorer (LoRA + stability head) on CPU.

Run from esm-backbone-test/:  python -m pytest tests/test_esm3_stab_scorer.py -v
Uses tests/fixtures/one_complex.pkl (complex 1A4Y; binders 1A4Y_A/_B) under
data/SKEMPI2_PDBs. ESM3 loads on CPU once (module-scoped fixture; ~minutes).
"""
import os
import pickle

import pytest
import torch

from track_esm3stab.esm3_stab_scorer import ESM3StabScorer, ESM3StabilityHead

_THIS = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
PDB_DIR = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI2_PDBs")
FIX = os.path.join(_THIS, "fixtures", "one_complex.pkl")


def _load():
    with open(FIX, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def scorer():
    return ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)


def test_folding_dG_shape_finite(scorer):
    dom = _load()["binder1"]
    dG = scorer.folding_dG(dom, scorer.get_wt_seq(dom))
    assert dG.shape == (1,) and torch.isfinite(dG).all()


def test_wt_ddG_zero(scorer):
    dom = _load()["binder1"]
    assert torch.allclose(
        scorer.folding_ddG(dom, scorer.get_wt_seq(dom)), torch.zeros(1), atol=1e-4
    )


def test_mutation_changes_dG(scorer):
    dom = _load()["binder1"]
    wt = scorer.get_wt_seq(dom)
    mut = wt.clone()
    mut[0, 5] = (mut[0, 5] + 1) % 20
    d = scorer.folding_ddG(dom, mut)
    assert d.shape == (1,) and torch.isfinite(d).all() and d.abs().item() > 1e-6


def test_batch_consistency(scorer):
    dom = _load()["binder1"]
    s1 = scorer.get_wt_seq(dom)
    s2 = s1.clone()
    s2[0, 3] = (s2[0, 3] + 1) % 20
    batch = torch.cat([s1, s2], 0)
    dG_b = scorer.folding_dG(dom, batch)
    dG_e = torch.cat([scorer.folding_dG(dom, s1), scorer.folding_dG(dom, s2)])
    assert torch.allclose(dG_b, dG_e, atol=1e-3), (dG_b, dG_e)


def test_factory_returns_scorer():
    from common.build_scorer import build_scorer
    sc = build_scorer("esm3_stab", device="cpu", pdb_dir=PDB_DIR)
    assert isinstance(sc, ESM3StabScorer)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd esm-backbone-test && python -m pytest tests/test_esm3_stab_scorer.py -x -q`
Expected: collection/import error — `ModuleNotFoundError: No module named 'track_esm3stab'` (the factory test also fails until Task 3). This confirms the test targets unwritten code.

- [ ] **Step 4: Write the scorer**

Create `esm-backbone-test/track_esm3stab/__init__.py` (empty file).

Create `esm-backbone-test/track_esm3stab/esm3_stab_scorer.py`:
```python
"""esm3_stab_scorer.py — ESM3 + LoRA + stability-head folding-energy scorer.

The parameter-efficient analogue of the old likelihood ESM3Scorer: ESM3 base is
FROZEN, only LoRA r=4 adapters + a small per-residue regression head train. The
head maps ESM3 per-residue embeddings -> a per-residue dG; folding_dG SUMS them
over the real-residue positions (extensive, so StaB's complex-binder1-binder2
binding-ddG decomposition stays additive). No sigmoid scaling: we supervise ddG
only (StaB's loss), so the absolute dG offset is free.
"""
import os

os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")

import torch
from torch import nn

from esm.pretrained import ESM3_sm_open_v0
from esm.utils.encoding import tokenize_sequence
from peft import LoraConfig, TaskType, get_peft_model

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
        get_peft_model(base, peft_config)  # injects LoRA in-place, freezes base
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
```

- [ ] **Step 5: Run the scorer tests (factory test still expected to fail)**

Run: `cd esm-backbone-test && python -m pytest tests/test_esm3_stab_scorer.py -v -k "not factory"`
Expected: `test_folding_dG_shape_finite`, `test_wt_ddG_zero`, `test_mutation_changes_dG`, `test_batch_consistency` all PASS. (`test_factory_returns_scorer` is wired in Task 3.)

- [ ] **Step 6: Commit**

```bash
git add esm-backbone-test/track_esm3stab/ esm-backbone-test/tests/test_esm3_stab_scorer.py
git commit -m "feat(esm3_stab): ESM3+LoRA+stability-head scorer (folding_dG regression)"
```

---

### Task 2: Wire `esm3_stab` into the `build_scorer` factory

**Files:**
- Modify: `esm-backbone-test/common/build_scorer.py`
- Test: `esm-backbone-test/tests/test_esm3_stab_scorer.py::test_factory_returns_scorer` (already written in Task 1)

**Interfaces:**
- Consumes: `track_esm3stab.esm3_stab_scorer.ESM3StabScorer`.
- Produces: `build_scorer("esm3_stab", device, checkpoint, pdb_dir) -> ESM3StabScorer`.

- [ ] **Step 1: Run the factory test to confirm it fails**

Run: `cd esm-backbone-test && python -m pytest tests/test_esm3_stab_scorer.py::test_factory_returns_scorer -x -q`
Expected: FAIL — `ValueError: unknown backbone esm3_stab`.

- [ ] **Step 2: Add the branch**

In `esm-backbone-test/common/build_scorer.py`, after the `if backbone == "esm3":` block (before `raise ValueError`), insert:
```python
    if backbone == "esm3_stab":
        from track_esm3stab.esm3_stab_scorer import ESM3StabScorer
        return ESM3StabScorer(device=device, pdb_dir=pdb_dir)
```

- [ ] **Step 3: Run the factory test to verify it passes**

Run: `cd esm-backbone-test && python -m pytest tests/test_esm3_stab_scorer.py::test_factory_returns_scorer -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add esm-backbone-test/common/build_scorer.py
git commit -m "feat(build_scorer): add esm3_stab backbone branch"
```

---

### Task 3: Parameter-efficient hooks in `finetune.py` (freeze base, train/save adapters)

**Files:**
- Modify: `esm-backbone-test/run/finetune.py`
- Test: `esm-backbone-test/tests/test_esm3_stab_finetune.py` (Create)

**Interfaces:**
- Consumes: `ESM3StabScorer.{trainable_parameters,freeze_base,save_adapters,load_adapters}`.
- Produces (module-level helpers usable by `eval.py`): `_select_trainable_params(scorer, backbone)->iterable`; `_save_backbone_ckpt(scorer, backbone, path)`; extended `_load_checkpoint_for_esm(scorer, backbone, checkpoint)`. New choices: `--backbone esm3_stab`; `DEFAULT_BATCH_SIZE["esm3_stab"]=1000`.

- [ ] **Step 1: Write the failing test**

Create `esm-backbone-test/tests/test_esm3_stab_finetune.py`:
```python
"""test_esm3_stab_finetune.py — param-efficient training path on CPU.

One optimizer step over the fixture domain must update ONLY LoRA+head, leaving
the frozen ESM3 base unchanged; adapter checkpoints round-trip. ESM3 loads on
CPU (a few minutes per scorer) — these are slow integration gates.
"""
import os
import pickle

import torch

from track_esm3stab.esm3_stab_scorer import ESM3StabScorer

_THIS = os.path.dirname(os.path.abspath(__file__))
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
PDB_DIR = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI2_PDBs")
FIX = os.path.join(_THIS, "fixtures", "one_complex.pkl")


def _load():
    with open(FIX, "rb") as f:
        return pickle.load(f)


def test_one_step_updates_only_adapters():
    sc = ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)
    sc.freeze_base()
    sc.train()
    dom = _load()["binder1"]
    wt = sc.get_wt_seq(dom)
    mut = wt.clone()
    mut[0, 7] = (mut[0, 7] + 1) % 20

    base_name = next(n for n, _ in sc.model.named_parameters() if "lora_" not in n)
    base_before = dict(sc.model.named_parameters())[base_name].detach().clone()
    head_before = [p.detach().clone() for p in sc.head.parameters()]

    opt = torch.optim.Adam(sc.trainable_parameters(), lr=1e-2)
    pred = sc.folding_ddG(dom, mut)
    loss = torch.nn.functional.mse_loss(pred, torch.tensor([1.0]))
    loss.backward()
    opt.step()

    base_after = dict(sc.model.named_parameters())[base_name]
    assert torch.equal(base_before, base_after), "frozen base param changed!"
    head_after = list(sc.head.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(head_before, head_after)), \
        "head did not update"


def test_adapter_ckpt_roundtrip(tmp_path):
    sc = ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)
    dom = _load()["binder1"]
    wt = sc.get_wt_seq(dom)
    before = sc.folding_dG(dom, wt).item()
    path = str(tmp_path / "adapters.pt")
    sc.save_adapters(path)

    sc2 = ESM3StabScorer(device="cpu", pdb_dir=PDB_DIR)
    sc2.load_adapters(path)
    after = sc2.folding_dG(dom, wt).item()
    assert abs(before - after) < 1e-4, (before, after)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd esm-backbone-test && python -m pytest tests/test_esm3_stab_finetune.py -x -q`
Expected: PASS already? No — these exercise only the scorer (built in Task 1), so they may PASS here. That is acceptable: they are the regression gate for the helper refactor below. If they PASS now, proceed to wire the helpers and re-run to confirm they STILL pass. (TDD note: the failing artifact for this task is the *finetune.py* behavior, validated by Step 5's defaults-preserving check.)

- [ ] **Step 3: Add helpers + choices in `finetune.py`**

In `esm-backbone-test/run/finetune.py`:

(a) Add `esm3_stab` to the batch defaults (line ~137):
```python
DEFAULT_BATCH_SIZE = {"mpnn": 10000, "esmc_600m": 2000, "esmc_6b": 2000, "esm3": 500, "esm3_stab": 1000}
```

(b) Add two helpers near `_build_optimizer` (after line ~160):
```python
def _select_trainable_params(scorer, backbone):
    """Params to optimize. For parameter-efficient scorers (esm3_stab) freeze the
    base and return only LoRA+head; else preserve the original full-FT behavior
    (all backbone params, requires_grad_(True))."""
    bb = scorer.backbone_module
    if hasattr(scorer, "trainable_parameters") and hasattr(scorer, "freeze_base"):
        scorer.freeze_base()
        bb.train()
        return scorer.trainable_parameters()
    bb.train()
    bb.requires_grad_(True)
    return bb.parameters()


def _save_backbone_ckpt(scorer, backbone, path):
    """Save adapters-only for parameter-efficient scorers, else full state_dict."""
    if hasattr(scorer, "save_adapters"):
        scorer.save_adapters(path)
    else:
        torch.save(scorer.backbone_module.state_dict(), path)
```

(c) In `finetune()` (SKEMPI loop), replace the block at lines ~327-344:
```python
    backbone = model.scorer.backbone_module

    optimizer = _build_optimizer(backbone.parameters(), args)
```
... and the later `backbone.train(); backbone.requires_grad_(True)` (lines ~343-344) — with:
```python
    scorer = model.scorer
    backbone = scorer.backbone_module
    optimizer = _build_optimizer(_select_trainable_params(scorer, args.backbone), args)
```
and DELETE the standalone `backbone.train()` / `backbone.requires_grad_(True)` lines (now done inside `_select_trainable_params`). Keep `total_steps`/`scheduler`/`ddG_loss_fn` as-is.

(d) In `finetune()`, replace the checkpoint save (line ~408) `torch.save(backbone.state_dict(), ckpt_path)` with:
```python
        _save_backbone_ckpt(scorer, args.backbone, ckpt_path)
```

(e) In `finetune_stability()`, apply the same three edits: replace `optimizer = _build_optimizer(backbone.parameters(), args)` (line ~424) with `optimizer = _build_optimizer(_select_trainable_params(scorer, args.backbone), args)`; DELETE `backbone.train()` / `backbone.requires_grad_(True)` (lines ~444-445); replace `torch.save(backbone.state_dict(), ckpt_path)` (line ~504) with `_save_backbone_ckpt(scorer, args.backbone, ckpt_path)`. (`scorer` and `backbone` are already in scope here.)

(f) Extend `_load_checkpoint_for_esm` (line ~508) to prefer adapters:
```python
def _load_checkpoint_for_esm(scorer, backbone, checkpoint):
    if checkpoint is None or backbone == "mpnn":
        return
    if hasattr(scorer, "load_adapters"):
        scorer.load_adapters(checkpoint)
        print(f"[checkpoint] loaded LoRA+head adapters from {checkpoint}", flush=True)
        return
    sd = torch.load(checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = scorer.backbone_module.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[checkpoint] load_state_dict non-strict: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys", flush=True)
    print(f"[checkpoint] loaded backbone init weights from {checkpoint}", flush=True)
```

(g) Add `esm3_stab` to the `--backbone` choices (line ~533-534):
```python
    ap.add_argument("--backbone", required=True,
                    choices=["mpnn", "esmc_600m", "esmc_6b", "esm3", "esm3_stab"])
```

- [ ] **Step 4: Run the finetune tests to verify they pass**

Run: `cd esm-backbone-test && python -m pytest tests/test_esm3_stab_finetune.py -v`
Expected: both PASS (base frozen, head updates, adapter round-trip).

- [ ] **Step 5: Defaults-preserving regression check (existing backbones unchanged)**

Run: `cd esm-backbone-test && python -m pytest tests/ -q -k "mpnn or eval_format or seq_codec or env"`
Expected: existing tests still PASS (the conditional helpers leave the `mpnn`/`esm*` full-FT path identical — `_select_trainable_params` falls back to `backbone.parameters()` + `requires_grad_(True)`).

- [ ] **Step 6: Commit**

```bash
git add esm-backbone-test/run/finetune.py esm-backbone-test/tests/test_esm3_stab_finetune.py
git commit -m "feat(finetune): conditional LoRA+head param-efficient path for esm3_stab (additive; full-FT path unchanged)"
```

---

### Task 4: Wire `esm3_stab` into `eval.py`

**Files:**
- Modify: `esm-backbone-test/run/eval.py`

**Interfaces:**
- Consumes: `run.finetune._load_checkpoint_for_esm` (already imported; now adapter-aware), `build_scorer("esm3_stab", ...)`.
- Produces: `python -m run.eval --backbone esm3_stab --checkpoint <adapters.pt> --split test` runs end-to-end.

- [ ] **Step 1: Add the choice + eval batch default**

In `esm-backbone-test/run/eval.py`:

(a) `_EVAL_BATCH_SIZE` (line ~159) — add `"esm3_stab": 2000`:
```python
_EVAL_BATCH_SIZE = {"mpnn": 10000, "esmc_600m": 10000, "esmc_6b": 10000, "esm3": 2000, "esm3_stab": 2000}
```

(b) `--backbone` choices (line ~205-206):
```python
    ap.add_argument("--backbone", required=True,
                    choices=["mpnn", "esmc_600m", "esmc_6b", "esm3", "esm3_stab"])
```

- [ ] **Step 2: Smoke the eval CLI on CPU (1 complex) with an untrained adapter ckpt**

Run:
```bash
cd esm-backbone-test
python - <<'PY'
from track_esm3stab.esm3_stab_scorer import ESM3StabScorer
sc = ESM3StabScorer(device="cpu", pdb_dir="../data/SKEMPI2_PDBs")
sc.save_adapters("cache/esm3stab_untrained.pt")
print("saved untrained adapters")
PY
python -m run.eval --backbone esm3_stab --checkpoint cache/esm3stab_untrained.pt \
    --split test --ensemble 1 --limit 1 --device cpu --out cache/eval_esm3stab_smoke.csv
head -3 cache/eval_esm3stab_smoke.csv
```
Expected: prints `[eval] backbone=esm3_stab ...`, writes `cache/eval_esm3stab_smoke.csv` whose header is `,#Pdb,Mutation,ddG,ddG_pred`. (Values are meaningless — head is untrained; this only checks the path runs.)

- [ ] **Step 3: Commit**

```bash
git add esm-backbone-test/run/eval.py
git commit -m "feat(eval): add esm3_stab backbone (adapter-aware checkpoint load)"
```

---

### Task 5: Local end-to-end smoke (Stage-1 + Stage-2 + eval, tiny, CPU)

**Files:** none (validation only).

- [ ] **Step 1: Stage-1 (Megascale folding) micro-smoke**

Run:
```bash
cd esm-backbone-test
python -m run.finetune --backbone esm3_stab --stage stability \
    --epochs 1 --limit 1 --batch_size 1000 --device cpu \
    --out cache/s1_stab_smoke --run_name esm3stab_s1
ls cache/s1_stab_smoke/esm3stab_s1_1.pt
```
Expected: a `*_train_log.csv` with one epoch row and a small checkpoint `esm3stab_s1_1.pt` (only LoRA+head ≈ a few MB, NOT ~5 GB). Verify size: `du -h cache/s1_stab_smoke/esm3stab_s1_1.pt` is small.

- [ ] **Step 2: Stage-2 (SKEMPI binding) micro-smoke chained from Stage-1**

Run:
```bash
cd esm-backbone-test
python -m run.finetune --backbone esm3_stab --stage skempi \
    --checkpoint cache/s1_stab_smoke/esm3stab_s1_1.pt \
    --epochs 1 --limit 1 --batch_size 1000 --device cpu \
    --out cache/s2_skempi_smoke --run_name esm3stab_s2
ls cache/s2_skempi_smoke/esm3stab_s2_1.pt
```
Expected: prints `[checkpoint] loaded LoRA+head adapters ...`, trains one epoch on 1 complex, saves a small ckpt.

- [ ] **Step 3: Eval micro-smoke**

Run:
```bash
cd esm-backbone-test
python -m run.eval --backbone esm3_stab --checkpoint cache/s2_skempi_smoke/esm3stab_s2_1.pt \
    --split test --ensemble 1 --limit 1 --device cpu --out cache/eval_s2_smoke.csv
wc -l cache/eval_s2_smoke.csv
```
Expected: writes a CSV with the right header and ≥1 prediction row, no exceptions.

- [ ] **Step 4: Commit a smoke note (no code)**

```bash
git commit --allow-empty -m "test(esm3_stab): local CPU end-to-end smoke (S1->S2->eval) passes on 1 domain/complex"
```

---

### Task 6: Ibex setup — data locality, env, per-branch sync, smoke job

**Files:** Create `ibex-records/esm-lora-replace/sh/smoke_stage1_esm3lora_<dt>.sh`; `ibex-records/esm-lora-replace/.gitignore`.

**REQUIRED SUB-SKILL for Tasks 6–9:** use the `ibex-usage` skill conventions verbatim.

- [ ] **Step 1: Verify data locality in this worktree**

Run:
```bash
cd /home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace
for d in data/SKEMPI2_PDBs data/AlphaFold_model_PDBs data/Processed_K50_dG_datasets data/SKEMPI data/rocklin; do
  echo "$d: $(ls -1 $d 2>/dev/null | wc -l) entries"
done
```
Expected: each path exists and is non-empty (these are the Stage-1 Megascale + Stage-2 SKEMPI inputs `finetune.py` resolves against the worktree root). If any is empty/missing, copy it into the worktree (gitignored) before syncing — do NOT reference another worktree's data.

- [ ] **Step 2: Create the records dirs + gitignore**

Run:
```bash
cd /home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace
mkdir -p ibex-records/esm-lora-replace/sh ibex-records/esm-lora-replace/results
printf '*.out\n*.err\nresults/*.pkl\n*.pt\n' > ibex-records/esm-lora-replace/.gitignore
ssh guoj0f@glogin.ibex.kaust.edu.sa 'mkdir -p /ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/sh /ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/results'
```

- [ ] **Step 3: Install `peft` on the Ibex `esm-backbone` env + pre-compile**

Run:
```bash
ssh guoj0f@glogin.ibex.kaust.edu.sa 'source $(conda info --base)/etc/profile.d/conda.sh && conda activate esm-backbone && pip install "peft>=0.11,<0.18" && python -c "import peft;print(peft.__version__)"'
```
Expected: prints a peft version. (If an env named `esm-backbone` is absent on Ibex, create it mirroring local, then `compileall` per the ibex skill.)

- [ ] **Step 4: Sync this worktree → per-branch Ibex path**

Run (verify branch first, per ibex-usage §1c):
```bash
cd /home/guoj0f/repos/StaB-ddG/.claude/worktrees/esm-replace
test "$(git rev-parse --abbrev-ref HEAD)" = "esm-replace" || { echo "WRONG BRANCH"; exit 1; }
rsync -av --exclude .git --exclude .claude/worktrees \
    ./ guoj0f@glogin.ibex.kaust.edu.sa:/ibex/user/guoj0f/StaB-ddG/esm-replace/
```

- [ ] **Step 5: Write + submit the Stage-1 smoke sbatch**

Create `ibex-records/esm-lora-replace/sh/smoke_stage1_esm3lora_<dt>.sh` (`<dt>` = `date +%Y%m%d-%H%M%S`):
```bash
#!/bin/bash
#SBATCH --job-name=smoke_s1_esm3lora
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/smoke_s1_esm3lora_%j.out
#SBATCH --error=/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/smoke_s1_esm3lora_%j.err
set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate esm-backbone
cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test
srun python -m run.finetune --backbone esm3_stab --stage stability \
    --epochs 1 --limit 3 --batch_size 2000 \
    --out cache/s1_smoke --run_name esm3stab_s1_smoke
```
Submit + monitor:
```bash
scp ibex-records/esm-lora-replace/sh/smoke_stage1_esm3lora_<dt>.sh guoj0f@glogin.ibex.kaust.edu.sa:/ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/sh/
ssh guoj0f@glogin.ibex.kaust.edu.sa 'cd /ibex/user/guoj0f/StaB-ddG/esm-replace/ibex-records/esm-lora-replace/sh && sbatch smoke_stage1_esm3lora_<dt>.sh'
# poll: squeue -u guoj0f ; on finish check the saved ckpt is small
```
Expected: job COMPLETED; `cache/s1_smoke/esm3stab_s1_smoke_1.pt` exists and is small; no OOM on a100.

- [ ] **Step 6: Commit the smoke script**

```bash
git add ibex-records/esm-lora-replace/sh/smoke_stage1_esm3lora_<dt>.sh ibex-records/esm-lora-replace/.gitignore
git commit -m "ibex(esm-lora-replace): stage-1 smoke sbatch + records scaffold"
git push origin HEAD
```

---

### Task 7: Ibex Stage-1 (Megascale folding ΔΔG) full run

**Files:** Create `ibex-records/esm-lora-replace/sh/megascale_stage1_esm3lora_<dt>.sh`; `ibex-records/esm-lora-replace/megascale_stage1_esm3lora_<dt>.md`.

- [ ] **Step 1: Write the Stage-1 sbatch (full train split)**

Create `ibex-records/esm-lora-replace/sh/megascale_stage1_esm3lora_<dt>.sh` (same SBATCH header style as Task 6; `--time=24:00:00`, job-name `megascale_stage1_esm3lora`):
```bash
cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test
srun python -m run.finetune --backbone esm3_stab --stage stability \
    --epochs 15 --batch_size 2000 \
    --optimizer adamw --weight_decay 0.05 --lr 1e-3 --warmup_frac 0.05 --lr_schedule cosine \
    --out /ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s1_esm3lora --run_name esm3lora_s1
```
(LoRA starting HPs: lr 1e-3 / AdamW wd 0.05 / warmup 0.05 / cosine — the reference repo used lr 1e-3; tune only if Stage-2 underperforms.)

- [ ] **Step 2: Sync code, submit, monitor**

Run: re-rsync (Task 6 Step 4) so the committed code is current, then `sbatch` from the `sh/` dir; poll `squeue`/`sacct` until COMPLETED. Note the **job id, elapsed, GPU**.

- [ ] **Step 3: Record Stage-1 results md**

Write `ibex-records/esm-lora-replace/megascale_stage1_esm3lora_<dt>.md` using the ibex-usage template (中英结合): job id, GPU, elapsed; HPs; the per-epoch train-loss curve (from `esm3lora_s1_train_log.csv`); ckpt path on /ibex (`runs/s1_esm3lora/esm3lora_s1_15.pt`); observations (loss converged? any OOM-skipped long domains?).

- [ ] **Step 4: Commit + push**

```bash
git add ibex-records/esm-lora-replace/sh/megascale_stage1_esm3lora_<dt>.sh ibex-records/esm-lora-replace/megascale_stage1_esm3lora_<dt>.md
git commit -m "ibex(esm-lora-replace): Stage-1 Megascale LoRA+head run + results"
git push origin HEAD
```

---

### Task 8: Ibex Stage-2 (SKEMPI binding ΔΔG) + test eval

**Files:** Create `ibex-records/esm-lora-replace/sh/skempi_stage2_esm3lora_<dt>.sh`; `ibex-records/esm-lora-replace/skempi_twostage_esm3lora_<dt>.md`.

- [ ] **Step 1: Stage-2 sbatch (chained from Stage-1 ckpt) + test eval in one job**

Create `ibex-records/esm-lora-replace/sh/skempi_stage2_esm3lora_<dt>.sh` (`--time=12:00:00`, job-name `skempi_stage2_esm3lora`):
```bash
cd /ibex/user/guoj0f/StaB-ddG/esm-replace/esm-backbone-test
S1=/ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s1_esm3lora/esm3lora_s1_15.pt
OUT=/ibex/user/guoj0f/StaB-ddG/esm-replace/runs/s2_esm3lora
srun python -m run.finetune --backbone esm3_stab --stage skempi \
    --checkpoint $S1 --epochs 15 --batch_size 1000 \
    --optimizer adamw --weight_decay 0.05 --lr 5e-4 --warmup_frac 0.05 --lr_schedule cosine \
    --out $OUT --run_name esm3lora_s2
srun python -m run.eval --backbone esm3_stab --checkpoint $OUT/esm3lora_s2_15.pt \
    --split test --ensemble 1 \
    --out $OUT/eval_esm3lora_twostage_test.csv
```

- [ ] **Step 2: Submit + monitor to COMPLETED.**

- [ ] **Step 3: Compute per-interface Spearman (same definition as the 0.445 baseline)**

Pull the CSV back and compute metrics with the **same script that produced the baseline numbers** (locate it in the repo, e.g. `reproduce/compute_metrics.py`, to guarantee comparability). The metric definition (per-interface Spearman, THRESHOLD=10) for cross-check:
```python
import pandas as pd, numpy as np
from scipy.stats import spearmanr
df = pd.read_csv("eval_esm3lora_twostage_test.csv")  # cols: ,#Pdb,Mutation,ddG,ddG_pred
THRESH = 10
rs = [spearmanr(g["ddG"], g["ddG_pred"]).correlation
      for _, g in df.groupby("#Pdb") if len(g) >= THRESH]
rs = [r for r in rs if np.isfinite(r)]
print("per-interface Spearman:", np.mean(rs), "+/-", np.std(rs), "over", len(rs), "interfaces")
print("overall Spearman:", spearmanr(df["ddG"], df["ddG_pred"]).correlation)
```

- [ ] **Step 4: Record two-stage results md (compact, one summary table)**

Write `ibex-records/esm-lora-replace/skempi_twostage_esm3lora_<dt>.md`: job ids, GPU, elapsed; Stage-2 HPs; ckpt + eval CSV paths on /ibex; headline table:

| config | per-iface Spearman | overall | notes |
|---|---:|---:|---|
| ProteinMPNN two-stage (baseline) | 0.445 | 0.531 | from old RESULTS.md (not re-run) |
| old ESM3 full-FT + likelihood | 0.242 | — | from old RESULTS.md (different method) |
| **ESM3 LoRA+head two-stage (this)** | **<value>** | <value> | esm3_stab |

- [ ] **Step 5: Commit + push** (scripts + md only).

---

### Task 9: Ablation (SKEMPI-only) + final synthesis + memory

**Files:** Create `ibex-records/esm-lora-replace/sh/skempi_only_ablation_esm3lora_<dt>.sh`; update the two-stage md (append ablation row — do NOT open a new file); new memory fact.

- [ ] **Step 1: SKEMPI-only ablation (no Stage-1) sbatch + eval**

Same as Task 8 Step 1 but **omit `--checkpoint`** (train from pretrained ESM3) and write `eval_esm3lora_skempionly_test.csv`. This re-measures Stage-1's contribution under the new recipe (vs old recipe's 0.148→0.242).

- [ ] **Step 2: Submit, monitor, compute per-interface Spearman (Task 8 Step 3 snippet).**

- [ ] **Step 3: Append the ablation row to the SAME two-stage md** (per ibex-usage "compact records — merge variants into one table"):

| ESM3 LoRA+head SKEMPI-only (no Stage-1) | <value> | <value> | ablation |

- [ ] **Step 4: Write a NEW memory fact** at `/home/guoj0f/.claude/projects/-home-guoj0f-repos-StaB-ddG/memory/esm-lora-replace-task.md` (type project), summarizing method (ESM3+LoRA+head, StaB ΔΔG loss, no MGnify/released weights), the headline numbers vs 0.445/0.242, and paths. Add the one-line pointer to `MEMORY.md`. **Link** to (do not edit) `[[esm-backbone-experiment]]`.

- [ ] **Step 5: Commit + push** (scripts + md; memory files are outside the repo).

---

## Self-Review

**1. Spec coverage:**
- Spec §4 architecture (LoRA r=4 + head + sum + naming) → Task 1. ✓
- §5 conditional param-efficient pipeline (freeze/save/load, additive) → Task 3. ✓
- §6 two-stage + eval + ablation → Tasks 7, 8, 9. ✓
- §7 isolation (new dir/backbone key, old artifacts untouched) → Global Constraints + Tasks 1–4 additive; verified by Task 3 Step 5. ✓
- §8 Ibex (per-branch path, ibex-records layout, peft, a100, data locality) → Tasks 6–9. ✓
- §1 decision (ΔΔG-only, no sigmoid) → head has no sigmoid (Task 1), loss unchanged (Task 3 keeps `MSELoss` on ΔΔG). ✓

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to". The Task 8 metric step references reusing the baseline script *and* gives the explicit metric code — not a placeholder. `<value>`/`<dt>` are runtime fill-ins (job outputs / timestamps), not design gaps.

**3. Type consistency:** `folding_dG`/`folding_ddG`/`get_wt_seq`/`backbone_module`/`freeze_base`/`trainable_parameters`/`save_adapters`/`load_adapters` names match across Tasks 1, 3, 4. `_select_trainable_params`/`_save_backbone_ckpt`/`_load_checkpoint_for_esm` defined in Task 3 and consumed by `finetune.py`/`eval.py`. `build_scorer("esm3_stab", ...)` (Task 2) used by Tasks 4, 8, 9. `ESM3StabilityHead(input_dim=1536, output_dim=1)` consistent. ✓
