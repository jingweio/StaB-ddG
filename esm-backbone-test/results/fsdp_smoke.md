# ESMC-6B FSDP/ZeRO-3 harness — local mechanics smoke (Task 9)

Harness: `track_esmc/fsdp_train_esmc6b.py` (`torchrun`-launched FSDP / ZeRO-3
full-fine-tune). The DEFINITIVE multi-shard 6B validation is the Ibex 8×a100
pilot (`track_esmc/pilot_esmc6b.sbatch`, human-gated, not run here). This file
records the strongest LOCAL mechanics check that the harness code is correct and
runs without API errors.

## Local hardware
With `CUDA_DEVICE_ORDER=PCI_BUS_ID`:
- `cuda:0` = NVIDIA RTX A4500, 21 GB, Ampere (cc 8.6) — good bf16.
- `cuda:1` = NVIDIA TITAN X (Pascal), 12.8 GB, Pascal (cc 6.1) — poor bf16.

Heterogeneous + Pascal → the local smoke runs in **fp32** and **without
flash-attn** (flash-attn not installed locally; ESM-C only needs a bool
`sequence_id` mask when flash is ON, so with it OFF none is passed). The Pascal
card's 12 GB ceiling forces a tiny token budget locally (see "OOM" below).

## ROOT-CAUSE bug found and fixed (the load-bearing result)
Naively FSDP-wrapping ESM-C and running `backward()` ALWAYS failed — even with
`world_size=1` (NO_SHARD), a SINGLE forward, and activation checkpointing OFF:

```
_post_backward_hook: expected to be in states [FORWARD_BACKWARD] but current
state is TrainingState.IDLE
```

Bisected cause: **ESM-C's `forward` returns an `ESMCOutput` dataclass**. FSDP1
registers its pre-backward hook by walking the forward output for tensors but
does NOT recurse into a custom dataclass, so the hook is never placed on
`sequence_logits`. The FSDP training-state machine never transitions to
FORWARD_BACKWARD and the first `_post_backward_hook` asserts. (Verified: a
synthetic transformer that returns a bare tensor passes the identical
1-forward and 6-forward FSDP patterns; ESM-C fails them; wrapping ESM-C so its
forward returns the bare `sequence_logits` tensor makes both pass.)

**Fix in the harness:** FSDP-wrap a thin `_LogitsAdapter(esmc)` whose forward
returns the plain `sequence_logits` tensor, and assign `scorer.model` a
`_FSDPModelShim` (nn.Module) that calls the FSDP adapter and re-wraps the tensor
in a `.sequence_logits`-bearing object — so `ESMCScorer.folding_dG` works
unchanged. The consolidated checkpoint save strips the `_LogitsAdapter.esmc.`
key prefix to recover plain ESM-C key names.

This bug is backbone-intrinsic, so it WILL bite the 8-GPU 6B Ibex run too; the
adapter fix is what makes the whole harness viable, not just a local nicety.

## What was validated (2-GPU `torchrun`, the PREFERRED path)

### A. FSDP 6-forward / 1-backward pattern via the harness wrap — OK (AC off AND on)
With the `_LogitsAdapter` fix, the StaB 6-forward step (complex/binder1/binder2
× mut/wt → one combined loss → one backward) runs cleanly through the
FSDP-wrapped ESM-C across 2 GPUs, with `--activation_checkpointing` both 0 and 1.
(The earlier AC-on failure had the SAME dataclass root cause — AC itself is NOT
the problem; activation checkpointing works once the adapter is in place.)

### B. End-to-end harness run (real SKEMPI 1A22, fp32, B=1, AC off, 3 steps)
Command:
```
PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID torchrun --nproc_per_node=2 \
  -m track_esmc.fsdp_train_esmc6b --backbone esmc_600m --stage skempi \
  --lr 1e-5 --epochs 1 --batch_size 300 --limit 1 --max_steps 3 \
  --use_flash_attn 0 --mixed_precision fp32 --activation_checkpointing 0 \
  --out cache/fsdp_smoke --run_name s
```
Result (PASS, exit 0):
```
[fsdp] backbone=esmc_600m world_size=2 backend=nccl mixed_precision=fp32 flash_attn=False ...
[fsdp] backbone wrapped (FULL_SHARD, activation_ckpt=False, mp=fp32); ESM-C blocks sharded across 2 rank(s)
[fsdp] train complexes = 1
[step 1] loss = 17.596437
[step 2] loss = 0.404981
[step 3] loss = 28.089405
[epoch 1/1] mean train loss = 15.363608
[ckpt] saved consolidated state_dict -> cache/fsdp_smoke/s_1.pt (368 tensors)
[fsdp] done.
```
All per-step losses finite (they vary because B=1 per-chunk MSE on single
mutations). Checkpoint saved + reloaded into a plain `ESMCScorer` (below).

- `init_process_group("nccl")` over 2 heterogeneous GPUs: OK.
- `FSDP(FULL_SHARD, use_orig_params=True)` auto-wrap on `{UnifiedTransformerBlock}`
  (36 ESM-C-600M blocks) with `device_id=local_rank`, `MixedPrecision=None` (fp32):
  OK — ESM-C sharded across both ranks.
- `StaBddG(scorer).forward(...)` drives the 6 scorer forwards through the
  FSDP-wrapped backbone (via `_FSDPModelShim`): OK — no API conflict with
  `ESMCScorer.folding_dG`.
- `loss.backward(); AdamW.step(); zero_grad()` per chunk: OK, finite losses.

### C. Consolidated (FULL_STATE_DICT) save + reload into a plain ESMCScorer — OK
`FSDP.state_dict_type(..., FULL_STATE_DICT, FullStateDictConfig(
offload_to_cpu=True, rank0_only=True))` on rank0 gathers ONE consolidated `.pt`
of the ESM-C weights (1.15 GB for 600M fp32, **368 tensors**) after stripping the
`esmc.` adapter prefix. Reload check (1 GPU, plain scorer):
```python
sd = torch.load("cache/fsdp_smoke/s_1.pt", map_location="cpu")   # 368 tensors
sc = ESMCScorer("esmc_600m", device="cpu", use_flash_attn=False)
sc.backbone_module.load_state_dict(sd, strict=True)   # <All keys matched successfully>
```
Keys are PLAIN ESM-C names (`embed.weight`, `transformer.blocks.0.attn...`) — NO
`_fsdp_wrapped_module.` / `_checkpoint_wrapped_module.` / `esmc.` prefixes — so
the consolidated checkpoint is loadable by a plain `ESMCScorer` for 1-GPU eval,
exactly as the two-stage flow (T11) needs.

GOTCHA also fixed: after the last per-chunk `backward()`, the FSDP root handle is
left in `BACKWARD_POST`; gathering FULL_STATE_DICT then asserts "Expects the
handle training to be IDLE". The harness runs one dummy ESM-C forward under
`eval()/no_grad` (`reset_fsdp_to_idle`), on ALL ranks in lock-step, to return
every handle to IDLE before the gather.

### Heterogeneous-GPU OOM (hardware artifact, not a harness bug)
At `--batch_size 2000` (B≈5 at L≈372), rank0 on the 12 GB Pascal card OOMs:
`GPU 0 has a total capacity of 11.90 GiB`. The six retained fp32 scorer graphs
of the 600M model exceed 12 GB. Dropping to `--batch_size 300` (B=1) fits. On
8×a100-80GB with bf16 this budget is far larger; the local OOM is purely the
12 GB Pascal card.

## Which validation path worked
PREFERRED (2-GPU `torchrun`, fp32, flash off, B=1): the harness mechanics — NCCL
init, FULL_SHARD wrap, the 6-forward StaBddG step (AC off and on),
backward/step, consolidated save, plain-scorer reload — all run without API
errors. The DEFINITIVE multi-shard 6B validation remains the Ibex 8×a100 pilot.

## Concerns for the 8×a100 6B Ibex pilot
- **`_LogitsAdapter` is REQUIRED** — without it FSDP+ESM-C cannot backward at all
  (the ESMCOutput-dataclass bug), independent of GPU count. It is wired into the
  harness; the pilot exercises it on flash-attn + bf16 + 8 shards (re-confirm no
  flash-attn-specific output-structure surprise).
- **Activation checkpointing**: needed for 6B memory (80 blocks). It works
  locally with the adapter; the `--activation_checkpointing {0,1}` flag is
  exposed. The pilot uses it ON (`pilot_esmc6b.sbatch`).
- **Consolidated-ckpt size**: 6B fp32 FULL_STATE_DICT ≈ 24 GB on disk, CPU-
  offloaded on rank0 (`offload_to_cpu=True`) → fits the 480 GB node RAM, but the
  gather is a real all-gather of the whole model; budget time + host RAM
  headroom. (600M was 1.15 GB / 368 tensors.)
- **Optimizer state**: with `use_orig_params=True` + FULL_SHARD, AdamW state is
  sharded (true ZeRO-3); peak ≈ 2×(params/N) per rank. Not separately
  checkpointed (the consolidated weight `.pt` is what eval needs); add
  `FSDP.optim_state_dict` if mid-run resume is wanted for the long run (T11).
- **The 6 retained graphs/step**: peak activation memory ≈ SUM of all six scorer
  graphs (complex/binder1/binder2 × mut/wt) until backward. Keep `--batch_size`
  (token budget) SMALL on the pilot. The pilot uses `--limit 5 --batch_size 8000`.
- **All ranks process identical data** (model-parallel ZeRO-3, not data-parallel
  over distinct shards). Correct and simplest for the tiny SKEMPI set; gradients
  are still all-reduced (reduce-scatter) by FSDP. A `DistributedSampler` could be
  layered on later for data-parallel throughput.
