# SKEMPI finetune smoke (Task 7)

Backbone-agnostic, single-GPU SKEMPI (stage-2) binding finetune harness
(`run/finetune.py`), exercised end-to-end as a TINY training run (`--limit 1`
train complex, `--epochs 2`) for both `esmc_600m` and `esm3`. This is the real
gate for Task 7: it proves the harness trains (gradients flow, weights change,
checkpoints round-trip) — NOT a science run.

## Setup
- Branch: `esm-replace` · env: `esm-backbone` · GPU: **CUDA device 1, NVIDIA RTX A4500 (20 GB)**
  (selected via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`; the other
  GPU is a 12 GB TITAN X Pascal at PCI index 0).
- Train split: `SKEMPIDataset(csv=data/SKEMPI/filtered_skempi.csv,
  split=data/SKEMPI/train_pdb.pkl, pdb_dir=data/SKEMPI2_PDBs,
  pdb_dict_cache=cache/skempi_train_pdb_dict.pkl, af_apo_structures=False)`
  — 120 complexes / 3050 mutants loaded, restricted to the first 1 complex
  (`1A22_A_B`, complex L=372, 99 mutants).
- Command (both backbones, sequential):
  ```
  cd esm-backbone-test
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python -m run.smoke_finetune --backbones esmc_600m esm3
  ```

## Results — ALL SMOKE ASSERTIONS PASSED
| backbone   | token budget | B (1A22, L=372) | init loss (ep1) | final loss (ep2) | finite | ckpts `_1.pt`/`_2.pt` | reload OK | weights changed | changed tensors | max\|Δ\| |
|------------|-------------:|----------------:|----------------:|-----------------:|:------:|:---------------------:|:---------:|:---------------:|----------------:|---------:|
| esmc_600m  | 2000         | 5               | 5.5787          | 4.9427           | yes    | yes                   | yes       | **True**        | 288 / 368       | 4.73e-04 |
| esm3       | 500          | 1               | 3.2666          | 2.6164           | yes    | yes                   | yes       | **True**        | 308 / 539       | 1.00e-03 |

Each backbone's assertions (in `run/smoke_finetune.py`):
1. no crash; per-epoch mean train losses finite (no NaN/Inf) — **pass** (and both
   decrease ep1→ep2, consistent with learning on one complex);
2. `{out}/{run}_1.pt` and `_2.pt` written; each reloads into a FRESH
   `build_scorer(backbone).backbone_module.load_state_dict(...)` — **pass**;
3. at least one backbone parameter CHANGED vs its initial snapshot after 2 epochs
   — **pass** for both. For ESM3 this is the direct proof that removing the
   internal `torch.no_grad()` (see pre-fix below) lets gradients flow: 308/539
   float tensors moved, max\|Δ\| = 1.0e-3.

Per-epoch CSV logs (`{out}/{run}_train_log.csv`):
```
# esmc_600m            # esm3
timestamp,epoch,mean_loss   timestamp,epoch,mean_loss
...,1,5.578744              ...,1,3.266636
...,2,4.942676              ...,2,2.616368
```
Checkpoint sizes: esmc_600m 1.15 GB, esm3 2.80 GB each (gitignored under
`cache/ft_smoke/`, `*.pt` rule).

## Required pre-fixes (training correctness) — done & verified
1. **`backbone_module` accessor.** Added an abstract `@property backbone_module`
   on `common.scorer.SequenceScorer` and concrete overrides:
   `MPNNScorer -> self.pmpnn`, `ESMCScorer -> self.model`, `ESM3Scorer ->
   self.model`. The finetune loop trains `scorer.backbone_module.parameters()`
   and saves `scorer.backbone_module.state_dict()`, decoupling optimizer/ckpt
   from the scorer wrapper so ONE loop drives every backbone.
2. **Removed gradient-blocking `no_grad`s.**
   - `ESM3Scorer.folding_dG` had an internal `with torch.no_grad(), autocast(...)`
     around the model forward — it would give ESM3 finetune ZERO gradient.
     Removed the `no_grad` (kept the autocast); eval wraps forwards in
     `torch.no_grad()` externally (`run/eval.py`), so inference is unaffected.
   - `ESMCScorer.folding_dG` was decorated `@torch.no_grad()` — same problem for
     ESM-C finetune. Removed the decorator.
   - Re-ran the scorer tests after both edits:
     `tests/test_esmc_scorer.py` **4/4 pass**, `tests/test_esm3_scorer.py`
     **4/4 pass** (incl. the multichain alignment-recovery gate).
   - `tests/test_mpnn_scorer_regression.py` still passes (bit-exact vs original),
     confirming the new `backbone_module` property didn't perturb MPNN; and
     `tests/test_eval_format.py` 4/4 pass.

## Memory finding (drove the token-budget defaults)
TRAINING needs FAR smaller batches than eval. `binding_ddG` runs SIX scorer
forwards per step (complex / binder1 / binder2, each mut + wt) whose autograd
graphs are ALL retained until `loss.backward()` — so peak memory is the sum
across all six, vastly larger than the no-grad eval forward at the same budget.
The eval default (10000 tokens) OOMs immediately in training.

Measured on the 20 GB A4500 (1A22, complex L=372), one mut/wt train step:
| backbone   | budget | B  | peak mem | result          |
|------------|-------:|---:|---------:|-----------------|
| esmc_600m  | 10000  | 26 | ~40 GB   | OOM             |
| esmc_600m  | 4000   | 10 | 16.48 GB | OK              |
| esmc_600m  | **2000** | **5** | **12.20 GB** | OK (default) |
| esmc_600m  | 1000   | 2  | 8.26 GB  | OK              |
| esm3       | 4000   | 8  | —        | OOM             |
| esm3       | 2000   | 5  | —        | OOM             |
| esm3       | 1000   | 2  | ~18.6–19.6 GB | OK but FLAKY (intermittent OOM across the 99-batch inner loop as fragmentation accumulates) |
| esm3       | **500**  | **1** | **19.62 GB** | OK, STABLE (full 99-batch loop × 2 epochs, 0 failures) |

Resulting per-backbone TRAINING defaults in `run/finetune.py`
(`DEFAULT_BATCH_SIZE`, a TOKEN budget; `B = max(1, budget // L)`):
`mpnn: 10000` (tiny model, original recipe), `esmc_600m / esmc_6b: 2000`,
`esm3: 500`. Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at
import to curb fragmentation, and made the per-complex `except` handler reclaim
the allocator cache (`optimizer.zero_grad(set_to_none=True)` + `empty_cache()`)
on a failed/OOM complex so one failure does not compound into the next.

**Known limitation (full run, out of scope for this smoke):** ESM3 at B=1 peaks at
~19.6 GB on 1A22 (L=372). The longest train complex is `3VR6_ABCDEF_GH` (L=3397);
at B=1 its O(L²) attention may exceed 20 GB. The loop SKIPS such complexes
(catches OOM, clears cache, continues) rather than crashing — a full ESM3 train
run on the A4500 may therefore drop a few giant complexes. ESM-C (600M) is much
roomier and trains the whole set comfortably at budget 2000.

## Harness notes
- `run/finetune.py` ports `skempi_finetune.py`'s exact inner loop (Adam on the
  backbone, MSELoss on binding ddG, per-complex `randperm` shuffling, token-budget
  batching, per-epoch checkpoint of the backbone state_dict). Data paths resolve
  against the worktree root (cwd-independent), mirroring `run/zeroshot_eval.py`.
  CLI: `--backbone --stage skempi --checkpoint --lr (1e-5) --epochs --batch_size
  --out --run_name --device (cuda) --limit`.
- The Megascale (stage-1) stability finetune is intentionally NOT implemented here
  (data not yet downloaded); `--stage` only accepts `skempi`.
