# Stage-1 Megascale folding-stability finetune smoke

The **Stage-1 (Megascale folding-stability)** finetune path, added to both the
single-GPU entrypoint (`run/finetune.py --stage stability`) and the FSDP harness
(`track_esmc/fsdp_train_esmc6b.py --stage stability`), exercised end-to-end as a
TINY training run (`--limit 3` train domains, `--epochs 2`) for both `esmc_600m`
and `esm3`. This is the gate for the task: it proves the stage-1 loop trains
(gradients flow, weights change, checkpoints round-trip) and that `--checkpoint`
chains a stage-1 backbone into a subsequent finetune — NOT a science run.

## Conceptual difference from Stage-2 (verified in code)
- **Stage-2 (SKEMPI binding, pre-existing):** `StaBddG.binding_ddG(complex,
  binder1, binder2, ...)` = complex − (b1+b2), three `folding_ddG` sub-structures
  per complex.
- **Stage-1 (Megascale folding, this task):** `scorer.folding_ddG(domain, chunk)`
  called DIRECTLY — one AlphaFold structure per domain, NO complex/binder
  decomposition, NO `StaBddG` wrapper.

## How the original Megascale loader was reused
`run.finetune.build_stability_dataset(pdb_dir, stability_csv, mega_splits, limit)`
is a faithful port of `stability_finetune.py`'s data-loading block (repo root):
- reads `data/rocklin/mega_splits.pkl` and uses the **train** split (the finetune
  loop scope; val/test scoring is out of this entrypoint's scope);
- parses each AlphaFold domain PDB via `stabddg.mpnn_utils.parse_PDB` with the
  original name→file mapping (`split off '.pdb'` then `'|' -> ':'`);
- builds `ddG_data[f'{name}.pdb'] = {'ddG'[N], 'mut_seqs'[N,L] int tensor}` from
  the Tsuboyama Dataset2/3 CSV with the **verbatim filters**: keep `ddG_ML != '-'`,
  drop `ins`/`del` mut_types, drop the `wt` row, match on `WT_name == raw split
  name`; sequences featurized with the same `ALPHABET='ACDEFGHIKLMNPQRSTVWYX'`;
- masks each AF domain on chain `'A'` (harmless for ESM-C/ESM3, which read
  seq/coords not masks).
The FSDP harness imports this exact function from `run.finetune` (no duplication).

**Data correctness check** (`--limit 3`, built in **3.8 s** via the chunked
CSV fast path): mutant-sequence length L equals the domain sequence length for all
3 domains (substitutions only), e.g. `r10_437_TrROS_Hall` L=47, N=876 mutants.

## Setup
- Branch: `esm-replace` · env: `esm-backbone` · GPU: **CUDA device 1, NVIDIA RTX
  A4500 (20 GB)** (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`; device 0
  is a 12 GB TITAN X Pascal).
- Data: `data/AlphaFold_model_PDBs/` (862 domain PDBs), `data/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (666 MB),
  `data/rocklin/mega_splits.pkl` (train=239 / val=31 / test=28 domains; all 298
  resolve to AF PDBs, none contain `'|'`).
- Commands:
  ```
  cd esm-backbone-test
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python -m run.finetune --backbone esmc_600m --stage stability \
    --epochs 2 --limit 3 --batch_size 2000 --out cache/s1_smoke_esmc --run_name esmc_s1
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python -m run.finetune --backbone esm3 --stage stability \
    --epochs 2 --limit 3 --batch_size 500 --out cache/s1_smoke_esm3 --run_name esm3_s1
  ```

## Results — single-GPU `run/finetune.py --stage stability` — ALL CHECKS PASSED
| backbone   | token budget | init loss (ep1) | final loss (ep2) | finite | ckpts `_1.pt`/`_2.pt` | strict reload | weights changed | changed tensors | max\|Δ\| | --checkpoint round-trip |
|------------|-------------:|----------------:|-----------------:|:------:|:---------------------:|:-------------:|:---------------:|----------------:|---------:|:-----------------------:|
| esmc_600m  | 2000         | 13.4904         | 6.9239           | yes    | yes                   | 0 missing/0 unexpected | **True** | 287 / 368 | 1.587e-03 | 0 missing/0 unexpected |
| esm3       | 500          | 7.5158          | 2.8498           | yes    | yes                   | 0 missing/0 unexpected | **True** | 309 / 539 | 3.967e-03 | 0 missing/0 unexpected |

Gate conditions (all pass for both backbones):
1. no crash; per-epoch mean train losses finite and **decrease** ep1→ep2;
2. `{out}/{run}_1.pt`, `_2.pt` written; epoch-2 ckpt reloads into a FRESH
   `build_scorer(backbone).backbone_module.load_state_dict(..., strict=True)` with
   **0 missing / 0 unexpected** keys;
3. backbone WEIGHTS CHANGED after 2 epochs vs a fresh untrained backbone (>0 float
   tensors differ → gradients flowed; ESM3 confirms grad flows through the
   structure-conditioned forward);
4. **`--checkpoint` round-trip**: epoch-2 ckpt loaded via the entrypoint's loader
   (`_load_checkpoint_for_esm`, strict=False) reports **0 missing / 0 unexpected**.
   End-to-end CLI chaining also confirmed: re-running `--stage stability
   --checkpoint cache/s1_smoke_esmc/esmc_s1_2.pt` warm-starts cleanly (ep1 loss
   4.96, below the 13.49 fresh-init → the loaded weights took effect).

### ESM3 AF-structure resolution (de-risks the full Ibex run)
ESM3 needs structure tokens of the Megascale AF domains. The stability stage builds
the scorer with `pdb_dir=data/AlphaFold_model_PDBs` (NOT the SKEMPI dir), so the
struct tokenizer resolves `domain['name']` (e.g. `r10_437_TrROS_Hall`, set by
`parse_PDB`) to `{pdb_dir}/{name}.pdb`. **No name↔PDB mismatch** — all 3 smoke
domains tokenized and trained; peak GPU ~14.4 GB at `--batch_size 500` (B=1).

## FSDP harness `track_esmc/fsdp_train_esmc6b.py --stage stability`
Stage-1 added as `train_stability` (mirrors the stage-2 `train`: per-chunk MSE on
`model.scorer.folding_ddG(domain, chunk)`, deterministic per-(epoch,domain)
permutation for lock-step ranks, `reset_fsdp_to_idle` + consolidated FULL_STATE_DICT
save per epoch).

- **Validated under real FSDP + NCCL (single-rank, A4500):**
  ```
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
    -m track_esmc.fsdp_train_esmc6b --backbone esmc_600m --stage stability \
    --epochs 1 --limit 2 --max_steps 3 --batch_size 2000 --use_flash_attn 0 \
    --mixed_precision fp32 --backend nccl --out cache/fsdp_s1_nccl --run_name fs1n
  ```
  → 3 steps, finite losses (17.56 / 20.45 / 29.39), consolidated state_dict saved
  (**368 tensors**, `esmc.` prefix stripped → plain ESM-C keys). The consolidated
  ckpt reloads strict into a fresh `ESMCScorer` (0 missing/0 unexpected) and
  weights changed — so FSDP stage-1 → stage-2 chaining via `--checkpoint` works.
- **gloo/CPU fallback (2-rank local):** fails with
  `FullyShardedDataParallel ... has no attribute '_unshard_stream'` /
  `FSDP-managed module unexpectedly has parameters on cpu`. **This is a
  PRE-EXISTING environment limitation** — `--stage skempi` fails identically on the
  same gloo/CPU setup (verified). The harness is designed for the Ibex 8×a100 NCCL
  pilot (per its docstring); local heterogeneous GPUs (12 GB Pascal + 20 GB A4500)
  preclude a multi-rank NCCL run here. The single-rank NCCL run above exercises the
  real FSDP+CUDA code path and the stage-1 loop end-to-end.

## Concerns for the full (non-limited) Ibex run
- **Megascale dataset build time / CSV read:** the smoke uses a chunked CSV fast
  path (only the limited domains' rows, ~3.8 s). The full run (`--limit` unset)
  falls back to reading the entire **666 MB** CSV once with `usecols` (4 columns) —
  a one-time cost (order minutes + memory for the frame), acceptable but worth
  budgeting. Building the 239 train AF structure dicts via `parse_PDB` is fast.
- **ESM3 AF-structure tokenization cost:** each unique domain is encoded once by
  the ESM3 VQ-VAE structure encoder and cached by name (`ESM3StructTokenizer`), so
  239 domains = 239 one-time encodes; cheap relative to training. Confirmed working
  on AF domains.
- **ESM3 memory O(B·L²):** Megascale domains are short (smoke domains L≈47–49), so
  ESM3 at `--batch_size 500` is comfortable (~14.4 GB peak). The longest train
  domains should still fit, but the loop catches per-domain OOMs and skips
  (mirrors stage-2), so a few giant domains would drop rather than crash.
