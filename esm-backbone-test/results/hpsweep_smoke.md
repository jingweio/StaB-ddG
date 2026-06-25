# ESM3 HP-eval tooling — local A4500 smoke (pipeline gate)

Branch `esm-replace`. Gate for the ESM3 Stage-2 hyperparameter-sweep tooling
(`run/eval.py --split`, `run/finetune.py` recipe options, `track_esm3/hpsweep_esm3.sbatch`).
Values are MEANINGLESS (2-3 complexes, 1 epoch) — this confirms the PIPELINE only.

## Environment

- Host GPU: NVIDIA RTX A4500, 20 GB (`CUDA_VISIBLE_DEVICES=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
- env `esm-backbone`; `cd esm-backbone-test`; `python -m run.{finetune,eval}`.
- No local Stage-1 checkpoint exists (`track_esm3/ckpts/` is empty; the converged
  Stage-1 ckpt lives on Ibex), so the smoke finetunes Stage-2 **from scratch** (no
  `--checkpoint`). Pipeline-only — that is fine for the gate.

## How `--limit` was threaded

`--limit` already exists on both entrypoints, so no new threading was needed:
- finetune: `--limit 2` → first 2 SKEMPI train complexes.
- eval: `--limit 3` (train split) / `--limit 2` (test split) → first K complexes.

## Batch-size note (A4500 vs a100)

The sweep sbatch uses `--batch_size 1500` (the a100 Stage-2 budget). On the 20 GB
A4500 that budget OOMs every complex (the loop catches the OOM, skips, continues —
mean loss = NaN, no optimizer step runs). To exercise the optimizer + new
scheduler step path on the A4500, the smoke finetune used the ESM3 A4500 default
`--batch_size 500`, which fits and produces a finite loss. The eval no-grad path
ran fine at the default eval budget (2000 tokens) for these short complexes.

## Commands run

```
# 1) Stage-2 finetune (defaults-preserving recipe options explicitly passed)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python -m run.finetune --backbone esm3 --stage skempi \
  --lr 1e-5 --epochs 1 --batch_size 500 \
  --optimizer adam --weight_decay 0 --warmup_frac 0 --lr_schedule constant \
  --out track_esm3/ckpts --run_name esm3_smoke --limit 2

# 2) eval TRAIN split
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python -m run.eval --backbone esm3 --checkpoint track_esm3/ckpts/esm3_smoke_1.pt \
  --split train --out results/eval_esm3_smoke_train.csv --ensemble 1 --limit 3

# 3) eval TEST split
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python -m run.eval --backbone esm3 --checkpoint track_esm3/ckpts/esm3_smoke_1.pt \
  --split test --out results/eval_esm3_smoke_test.csv --ensemble 1 --limit 2

# 4) summary (the sbatch's step-4 here-doc, env-var driven) appended a row.
```

## Results

| step | outcome |
|------|---------|
| finetune | exit 0; `[sched] optimizer=adam weight_decay=0.0 warmup_frac=0.0 lr_schedule=constant total_steps~=105`; `[epoch 1/1] mean train loss = 5.190256 lr = 1.000e-05`; ckpt `esm3_smoke_1.pt` saved |
| train-log CSV | new `lr` column written: `...,1,5.190256,1.000000e-05` |
| eval --split train | exit 0; built train dataset (3 complexes); wrote `eval_esm3_smoke_train.csv` (121 rows, baseline format w/ leading index col) |
| eval --split test | exit 0; built test dataset (2 complexes); wrote `eval_esm3_smoke_test.csv` (170 rows) |
| summary row | appended with FINITE values: `final_train_loss=5.190256, train_spearman=0.271955, test_spearman=0.270081` |

Summary row (header + one row, matching the sweep CSV schema):

```
run,lr,warmup,wd,opt,sched,epochs,final_train_loss,train_spearman,test_spearman
smoke,1e-5,0,0,adam,constant,1,5.190256,0.271955,0.270081
```

## Defaults-preserving verification

Unit-checked the new finetune helpers in isolation:
- `_build_optimizer` at defaults returns plain `Adam` with `weight_decay=0`
  (identical to the previous `torch.optim.Adam(params, lr=args.lr)`); `adamw`
  only with `--optimizer adamw`.
- `_build_scheduler` at defaults (warmup_frac=0, lr_schedule=constant) yields a
  `LambdaLR` whose multiplier is a CONSTANT 1.0 — the LR stayed exactly `1e-5`
  across 50 `optimizer.step()/scheduler.step()` iterations. So the per-step LR
  the optimizer sees is bit-for-bit the old behavior even though we now call
  `scheduler.step()`.
- Warmup + cosine engage correctly only when requested: linear ramp to peak 1.0
  at the end of warmup, cosine decay to ~0 at the final step.

## Notes / caveats

- A train-split eval at the a100 budget on the A4500 would OOM-skip the longest
  complexes (caught/skipped, dataset continues) — anticipated and harmless. On the
  a100 (the sweep target) the 1500-token budget is the intended setting.
- Smoke artifacts (`track_esm3/ckpts/esm3_smoke*`, `results/eval_esm3_smoke_*.csv`,
  `results/esm3_hp_sweep_smoke.csv`) are intermediate and NOT committed; only this
  markdown is.
```
