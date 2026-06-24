# Two-stage pipeline local end-to-end smoke (ESM3)

Date: 2026-06-24
Branch: `esm-replace`
Env: `esm-backbone`
GPU: RTX A4500 20GB (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`)

Goal: prove the full **stage1 (Megascale folding) -> stage2 (SKEMPI binding,
chained from the stage-1 ckpt) -> eval (new `run.eval` CLI)** chain works
end-to-end on the fastest backbone (ESM3), and that the new eval CLI loads a
finetuned ESM checkpoint and writes the baseline-format CSV. This is the GATE
for the multi-hour Ibex runs; the metric value is meaningless on tiny data.

## Commands & results

### Stage 1 — Megascale folding stability (`scorer.folding_ddG`)
```
python -m run.finetune --backbone esm3 --stage stability \
  --lr 1e-5 --epochs 1 --batch_size 2000 --limit 2 \
  --out /tmp/ts_smoke --run_name e3s1
```
- OK. `[epoch 1/1] mean train loss = 13.369986`
- Saved `/tmp/ts_smoke/e3s1_1.pt` (backbone state_dict).

### Stage 2 — SKEMPI binding (`StaBddG.binding_ddG`), chained
```
python -m run.finetune --backbone esm3 --stage skempi \
  --checkpoint /tmp/ts_smoke/e3s1_1.pt \
  --lr 1e-5 --epochs 1 --batch_size 500 --limit 2 \
  --out /tmp/ts_smoke --run_name e3s2
```
- OK. `[checkpoint] loaded backbone init weights from /tmp/ts_smoke/e3s1_1.pt`
  -> the stage-1 -> stage-2 chain works (`_load_checkpoint_for_esm`).
- `[epoch 1/1] mean train loss = 5.619264` (real gradient steps).
- Saved `/tmp/ts_smoke/e3s2_1.pt`.

NOTE on batch_size: the task's suggested `--batch_size 2000` for stage-2 OOMs on
the 20GB A4500 (both complexes skipped via the OOM-catch path -> loss=NaN, ckpt
still saved). This is expected — finetune.py's ESM3 SKEMPI default is **500
tokens** (six retained scorer graphs per binding step at the 20GB ceiling). I
re-ran stage-2 at `--batch_size 500`, which trains cleanly (loss 5.62). The
sbatch scripts target a100 (40/80GB), where the larger budget fits; the A4500
limit is a local-smoke artifact only.

### Eval — new `run.eval` CLI on the stage-2 checkpoint
```
python -m run.eval --backbone esm3 \
  --checkpoint /tmp/ts_smoke/e3s2_1.pt \
  --out /tmp/ts_smoke/eval.csv --ensemble 1 --limit 3
```
- OK. `[checkpoint] loaded backbone init weights from /tmp/ts_smoke/e3s2_1.pt`
  -> the eval CLI loads a FINETUNED ESM checkpoint (via `_load_checkpoint_for_esm`,
  same path lr_sweep uses; eval batch_size auto = 2000 for esm3, no-grad, no OOM).
- Wrote `/tmp/ts_smoke/eval.csv` (192 rows over 3 interfaces).

## Assertions (all PASS)
- Each stage completed; per-epoch ckpts saved (`e3s1_1.pt`, `e3s2_1.pt`).
- `eval.csv` header is EXACTLY the baseline format: `,#Pdb,Mutation,ddG,ddG_pred`
  (leading unnamed index column).
- columns = `['#Pdb', 'Mutation', 'ddG', 'ddG_pred']`, all `ddG_pred` finite.
- `baselines.eval_utils.compute_metrics(df, bootstrap=False)` RUNS:
  - Spearman = 0.2680, Per Structure Spearman = 0.0834,
    Pearson = 0.3426, RMSE = 2.3485
  - (meaningless on 3-complex tiny data — pipeline validation only.)

## Conclusion
The full **stage1 -> stage2 -> eval** chain + the new `run.eval` CLI are
validated end-to-end. The eval CLI correctly loads a finetuned ESM checkpoint
(build_scorer's `--checkpoint` is honoured only for `mpnn`; for esm3/esmc the
finetuned weights are loaded into `scorer.backbone_module` via
`_load_checkpoint_for_esm`, exactly as `run/finetune.py` and `run/lr_sweep.py`
do). Ready for the Ibex two-stage runs. FSDP path NOT re-smoked here (the pilot
already validated stage1+stage2 mechanics); the esmc sbatch references the
correct consolidated `{run_name}_{epoch}.pt` filenames (verified against
`fsdp_train_esmc6b.py:472`).
