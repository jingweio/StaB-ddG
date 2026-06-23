# Local Closed-Loop Gate — ESM-backbone experiment

**Date:** 2026-06-24  
**GPU:** NVIDIA RTX A4500 20 GB (PCI index 1 under `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`)  
**Branch:** `esm-replace`  
**Env:** `esm-backbone` (conda)  
**Env vars:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/home/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1`

Each run used `--epochs 1 --lrs "<single-lr>"` (single-LR sweep as gate driver).

## Results table

| backbone    | finetune loss (ep 1) | eval CSV path                                              | eval CSV rows | per-interface Spearman | overall Spearman | n_qualifying@K=10 | wall-time | CLOSED-LOOP |
|-------------|---------------------:|------------------------------------------------------------|:-------------:|----------------------:|----------------:|:-----------------:|:---------:|:-----------:|
| `mpnn`      | 2.782               | `results/gate_mpnn/eval_mpnn_lr1e-6.csv`                   | 224           | 0.5725                | 0.6629          | 5                 | ~1 min    | OK          |
| `esmc_600m` | 8.509               | `results/gate_esmc600m/eval_esmc_600m_lr1e-5.csv`          | 224           | -0.1627               | -0.0773         | 5                 | ~2 min    | OK          |
| `esm3`      | 5.346               | `results/gate_esm3/eval_esm3_lr1e-5.csv`                  | 192           | 0.0208                | 0.1508          | 3                 | ~4 min    | OK          |

Notes:
- `mpnn`: `--lrs 1e-6 --batch_size 10000 --limit_train 3 --limit_eval 5`. Requires `model_ckpts/stabddg.pt` (symlinked from worktree root into `esm-backbone-test/`).
- `esmc_600m`: `--lrs 1e-5 --batch_size 2000 --limit_train 3 --limit_eval 5`.
- `esm3`: `--lrs 1e-5 --batch_size 500 --limit_train 2 --limit_eval 3`. No OOM; ran cleanly at B=1.
- Metric values are meaningless on these tiny slices (3–5 train complexes, 3–5 eval complexes, 1 epoch). The gate criterion is only that the pipeline runs without error.
- `n_qualifying@K=10` counts complexes with >= 10 mutations in the eval slice. With 3–5 eval complexes the count can be < K=10 threshold for all entries, but `compute_metrics` still produced a per-interface Spearman here because those complexes individually had >= 10 mutations.
- esm3 eval CSV has 192 rows vs 224 for the others because it uses 3 eval complexes (vs 5).

## Gate commands run

```bash
# mpnn
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/home/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1 \
python -m run.lr_sweep --backbone mpnn --lrs "1e-6" --epochs 1 \
    --batch_size 10000 --device cuda --limit_train 3 --limit_eval 5 \
    --out_dir results/gate_mpnn

# esmc_600m
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/home/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1 \
python -m run.lr_sweep --backbone esmc_600m --lrs "1e-5" --epochs 1 \
    --batch_size 2000 --device cuda --limit_train 3 --limit_eval 5 \
    --out_dir results/gate_esmc600m

# esm3
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/home/guoj0f/repos/esm/.hf_cache HF_HUB_OFFLINE=1 \
python -m run.lr_sweep --backbone esm3 --lrs "1e-5" --epochs 1 \
    --batch_size 500 --device cuda --limit_train 2 --limit_eval 3 \
    --out_dir results/gate_esm3
```

## Verdict

**All three backbones passed the local closed loop.** For each backbone, `run_sweep` completed the full pipeline: (1) loaded/built the scorer, (2) ran one epoch of SKEMPI finetune producing a checkpoint, (3) loaded the checkpoint into a fresh model and ran eval producing a baseline-format CSV with `#Pdb/Mutation/ddG/ddG_pred` columns, (4) called `compute_metrics` which produced both a per-interface Spearman and an overall Spearman, and (5) wrote `sweep.csv` and `best.json`. No OOM was observed on any backbone; ESM3 ran stably at `batch_size=500` (B=1) on the 20 GB A4500, consistent with the earlier smoke finding (~19.6 GB peak for L=372). The pipeline is ready for full Ibex GPU jobs.

**Pre-existing fix noted:** The mpnn backbone requires `model_ckpts/stabddg.pt` relative to the working directory. That file lives at the worktree root (`esm-replace/model_ckpts/stabddg.pt`) but not inside `esm-backbone-test/`. A symlink `esm-backbone-test/model_ckpts -> ../model_ckpts` was created to resolve this; Ibex runs should either be launched from the worktree root or use the symlink (already in place).
