"""
fsdp_train_esmc6b.py — FSDP / ZeRO-3 full-fine-tune harness for ESM-C (6B / 600M).

The repo's single-GPU ``run/finetune.py`` cannot hold the 6B ESM-C backbone (the
StaB binding loss runs SIX scorer forwards per step — complex/binder1/binder2 ×
mut/wt — whose activation graphs are ALL retained until ``backward``). This
script sharded-trains the SAME SKEMPI recipe across N GPUs with PyTorch FSDP
(``FULL_SHARD`` == ZeRO-3): parameters, gradients and optimizer state are sharded
across ranks, gradients are all-reduced (reduce-scatter) by FSDP, and bf16 mixed
precision keeps the working set small.

Launched with ``torchrun`` (reads ``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` from
the environment). Every rank runs the SAME data through the SAME forward — this
is valid ZeRO-3: the model (not the data) is what FSDP shards, so each rank holds
1/N of the params and FSDP gathers full layers on the fly per forward/backward.
(Data parallelism over distinct shards could be layered on later via a
DistributedSampler; for the SKEMPI binding finetune the dataset is tiny and the
bottleneck is the 6B model, so keeping the data identical across ranks is the
simplest correct ZeRO-3 setup.)

The harness mechanics (block-class auto-wrap, activation checkpointing,
bf16 mixed precision, consolidated FULL_STATE_DICT save) are validated locally on
ESM-C-600M across 2 GPUs; the definitive multi-shard 6B validation is the Ibex
8×a100 pilot (``pilot_esmc6b.sbatch``).

Usage (local 2-GPU smoke, 600M, fp32 for the Pascal card)
---------------------------------------------------------
    CUDA_DEVICE_ORDER=PCI_BUS_ID torchrun --nproc_per_node=2 \
        -m track_esmc.fsdp_train_esmc6b \
        --backbone esmc_600m --stage skempi --lr 1e-5 --epochs 1 \
        --batch_size 2000 --limit 1 --use_flash_attn 0 --mixed_precision fp32 \
        --out cache/fsdp_smoke --run_name s

Ibex 8×a100 pilot (6B, bf16, flash-attn) — see pilot_esmc6b.sbatch.
"""

import argparse
import datetime
import functools
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path / env bootstrap (mirrors run/finetune.py for the `-m track_esmc....`     #
# module launch under torchrun). Adds the worktree root (for `stabddg`) and     #
# esm-backbone-test/ (for `common`, `track_*`) to sys.path, and sets offline-HF #
# env BEFORE any esm import.                                                    #
# --------------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parent.parent                      # .../esm-backbone-test
_WORKTREE_ROOT = _PKG_ROOT.parent                    # .../esm-replace
for _p in (str(_WORKTREE_ROOT), str(_PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from common.build_scorer import build_scorer
from common.stab_model import StaBddG
from stabddg.ppi_dataset import SKEMPIDataset
from esm.layers.blocks import UnifiedTransformerBlock

# SKEMPI (stage-2) inputs, resolved against the worktree root so the entrypoint
# is cwd-independent.
SKEMPI_CSV = str(_WORKTREE_ROOT / "data/SKEMPI/filtered_skempi.csv")
SKEMPI_TRAIN_SPLIT = str(_WORKTREE_ROOT / "data/SKEMPI/train_pdb.pkl")
SKEMPI_PDB_DIR = str(_WORKTREE_ROOT / "data/SKEMPI2_PDBs")


# --------------------------------------------------------------------------- #
# Distributed setup helpers                                                     #
# --------------------------------------------------------------------------- #
def setup_dist(backend):
    """init_process_group from torchrun env vars; return (rank, local_rank, world)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank):
    return rank == 0


def rank0_print(rank, *a, **kw):
    if is_main(rank):
        print(*a, **kw, flush=True)


# --------------------------------------------------------------------------- #
# FSDP wrapping                                                                 #
# --------------------------------------------------------------------------- #
class _LogitsAdapter(torch.nn.Module):
    """Thin wrapper whose forward returns the PLAIN ``sequence_logits`` tensor.

    WHY THIS EXISTS (verified root cause): ESM-C's ``forward`` returns an
    ``ESMCOutput`` dataclass. FSDP1 registers the pre-backward hook by walking the
    forward output for tensors, but it does NOT recurse into a custom dataclass —
    so when the FSDP root returns ``ESMCOutput`` the pre-backward hook is never
    placed on ``sequence_logits``. The FSDP training-state machine then never
    transitions to FORWARD_BACKWARD, and the FIRST backward trips
    ``_post_backward_hook: expected FORWARD_BACKWARD but current state is IDLE``
    (reproduces even with world_size=1, a single forward, AC off — i.e. it is the
    dataclass output, not sharding/multi-forward/activation-checkpointing).

    Having the FSDP root return a bare tensor lets FSDP register the hook
    correctly. The scorer reads only ``out.sequence_logits``, so a tiny shim
    (:class:`_FSDPModelShim`) re-wraps the tensor in an object exposing that one
    attribute — no scorer code changes needed.
    """

    def __init__(self, esmc):
        super().__init__()
        self.esmc = esmc

    def forward(self, sequence_tokens=None, sequence_id=None):
        return self.esmc(
            sequence_tokens=sequence_tokens, sequence_id=sequence_id
        ).sequence_logits


class _LogitsOut:
    """Minimal stand-in for ``ESMCOutput`` exposing only ``.sequence_logits``."""

    __slots__ = ("sequence_logits",)

    def __init__(self, sequence_logits):
        self.sequence_logits = sequence_logits


class _FSDPModelShim(torch.nn.Module):
    """``nn.Module`` assigned to ``scorer.model`` (a registered submodule slot).

    Forwards ``model(sequence_tokens=...)`` through the FSDP-wrapped
    :class:`_LogitsAdapter` (which returns a bare tensor so FSDP's backward hooks
    register) and re-wraps the result so ``out.sequence_logits`` keeps working in
    the scorer. It is an ``nn.Module`` holding the FSDP module as its child
    ``fsdp_module``, so it can be assigned to the ``ESMCScorer.model`` submodule
    slot and ``.train()``/``.eval()`` propagate to the FSDP params normally.
    """

    def __init__(self, fsdp_module):
        super().__init__()
        self.fsdp_module = fsdp_module

    def forward(self, sequence_tokens=None, sequence_id=None):
        logits = self.fsdp_module(sequence_tokens=sequence_tokens, sequence_id=sequence_id)
        return _LogitsOut(logits)


def wrap_backbone_fsdp(model_to_wrap, local_rank, mixed_precision, backend,
                       activation_checkpointing=True):
    """Wrap ESM-C in a logits-only adapter, activation-checkpoint every
    UnifiedTransformerBlock, THEN FSDP-wrap.

    ``model_to_wrap`` is the bare ESM-C model. We first wrap it in
    :class:`_LogitsAdapter` so the FSDP root returns a plain tensor (see that
    class for the verified reason). Activation checkpointing is then applied to
    the (nested) native blocks, and FSDP wraps the adapter, using the
    transformer-auto-wrap policy keyed on UnifiedTransformerBlock — which recurses
    through the adapter to find the 36/80 blocks.
    """
    # 0) Logits-only adapter so the FSDP root returns a bare tensor (FSDP backward
    #    hooks don't register through ESM-C's ESMCOutput dataclass otherwise).
    adapter = _LogitsAdapter(model_to_wrap)

    # 1) Activation checkpointing (NO_REENTRANT) on every native ESM-C block.
    if activation_checkpointing:
        non_reentrant = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        apply_activation_checkpointing(
            adapter,
            checkpoint_wrapper_fn=non_reentrant,
            check_fn=lambda m: isinstance(m, UnifiedTransformerBlock),
        )

    # 2) FSDP auto-wrap policy: shard at the UnifiedTransformerBlock granularity.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={UnifiedTransformerBlock},
    )

    # 3) Mixed precision: bf16 params/reduce/buffers, or None (pure fp32).
    if mixed_precision == "bf16":
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp = None

    # device_id is meaningless under the gloo/CPU fallback.
    device_id = local_rank if backend == "nccl" else None

    wrapped = FSDP(
        adapter,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device_id,
        use_orig_params=True,
    )
    return wrapped


# --------------------------------------------------------------------------- #
# Data                                                                          #
# --------------------------------------------------------------------------- #
def build_train_dataset(pdb_dict_cache_path, limit=None):
    dataset = SKEMPIDataset(
        csv_path=SKEMPI_CSV,
        split_path=SKEMPI_TRAIN_SPLIT,
        pdb_dir=SKEMPI_PDB_DIR,
        pdb_dict_cache_path=pdb_dict_cache_path,
        af_apo_structures=False,
    )
    if limit is not None:
        dataset.data = dataset.data[:limit]
    return dataset


# --------------------------------------------------------------------------- #
# Training loop (ported from run/finetune.py, FSDP-aware)                        #
# --------------------------------------------------------------------------- #
def train(model, train_dataset, args, device, rank):
    """SKEMPI binding finetune over the FSDP-wrapped backbone.

    Same per-complex token-budget batching / per-chunk backward+step+zero_grad as
    run/finetune.py. The optimizer is built over the FSDP-wrapped module's
    parameters (with use_orig_params=True these are the flat sharded params, so
    optimizer state is sharded too — true ZeRO-3). Every rank runs the same
    samples; FSDP all-reduces gradients across ranks.
    """
    # The trainable params live in the FSDP-wrapped adapter behind the shim
    # (scorer.model is a _FSDPModelShim, NOT an nn.Module child of StaBddG, so
    # model.parameters() / model.train() do NOT reach it). Optimize and toggle
    # train()/eval() on the FSDP module directly.
    fsdp_module = model.scorer.model.fsdp_module
    optimizer = torch.optim.AdamW(fsdp_module.parameters(), lr=args.lr)
    ddG_loss_fn = torch.nn.MSELoss()

    fsdp_module.train()

    if is_main(rank):
        os.makedirs(args.out, exist_ok=True)
        log_path = os.path.join(args.out, f"{args.run_name}_train_log.csv")
        with open(log_path, "w") as lf:
            lf.write("timestamp,epoch,mean_loss\n")
        rank0_print(rank, f"[train log] per-epoch mean loss -> {log_path}")

    global_step = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        epoch_losses = []
        for sample in train_dataset:
            if stop:
                break
            try:
                complex, binder1, binder2 = (
                    sample["complex"], sample["binder1"], sample["binder2"]
                )
                cms = sample["complex_mut_seqs"]
                b1ms = sample["binder1_mut_seqs"]
                b2ms = sample["binder2_mut_seqs"]
                ddG = sample["ddG"].float().to(device)

                N = cms.shape[0]
                M = max(1, args.batch_size // cms.shape[1])

                # Per-complex mutant shuffle. All ranks must agree on the order so
                # the per-chunk forwards stay in lock-step (FSDP collectives are
                # ordered) — seed the permutation deterministically per
                # (epoch, complex).
                g = torch.Generator()
                g.manual_seed(args.seed + epoch * 100003 + abs(hash(sample["complex"]["name"])) % 100003)
                perm = torch.randperm(N, generator=g)
                ddG = ddG[perm]
                cms, b1ms, b2ms = cms[perm], b1ms[perm], b2ms[perm]

                for i in range(0, N, M):
                    B = min(N - i, M)
                    pred = model(
                        complex, binder1, binder2,
                        cms[i:i + B], b1ms[i:i + B], b2ms[i:i + B],
                    )
                    loss = ddG_loss_fn(pred, ddG[i:i + B])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_losses.append(loss.item())
                    global_step += 1
                    rank0_print(rank, f"[step {global_step}] loss = {loss.item():.6f}")
                    # --max_steps caps total optimizer steps (smoke / pilot). All
                    # ranks share the same data/order, so they break together.
                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop = True
                        break
            except Exception as e:
                rank0_print(rank, "Failed on", sample["complex"]["name"], repr(e))
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        mean_loss = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
        rank0_print(rank, f"[epoch {epoch + 1}/{args.epochs}] mean train loss = {mean_loss:.6f}")
        if is_main(rank):
            with open(log_path, "a") as lf:
                lf.write(f"{datetime.datetime.now().isoformat(timespec='seconds')},"
                         f"{epoch + 1},{mean_loss:.6f}\n")

        # After the last per-chunk backward the FSDP root handle is left in the
        # BACKWARD_POST training state; gathering a FULL_STATE_DICT then trips
        # `Expects the handle training to be IDLE`. A single forward under
        # eval()/no_grad transitions every handle back to IDLE (post-forward
        # reset when grad isn't required), so do a tiny dummy forward first.
        reset_fsdp_to_idle(model, device)
        save_consolidated_checkpoint(model, args, epoch, rank)
        fsdp_module.train()


# --------------------------------------------------------------------------- #
# Sharded -> consolidated checkpoint save                                       #
# --------------------------------------------------------------------------- #
def reset_fsdp_to_idle(model, device):
    """Run one dummy ESM-C forward under eval()/no_grad so every FSDP handle
    transitions back to the IDLE training state (required before a
    FULL_STATE_DICT gather; otherwise the gather asserts on BACKWARD_POST).

    All ranks must run this in lock-step (it triggers FSDP all-gather
    collectives), so it is called outside the rank0-only save block.
    """
    fsdp_module = model.scorer.model.fsdp_module
    was_training = fsdp_module.training
    fsdp_module.eval()
    with torch.no_grad():
        dummy = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)  # [cls?,…]
        fsdp_module(sequence_tokens=dummy)
    if was_training:
        fsdp_module.train()


def save_consolidated_checkpoint(model, args, epoch, rank):
    """Gather the sharded ESM-C weights into ONE consolidated state_dict and save
    it (rank0 only), CPU-offloaded so the full 6B fp32 dict never lives on a GPU.

    The saved file is a plain ESM-C ``state_dict`` — loadable later on 1 GPU by
    ``ESMCScorer(...).backbone_module.load_state_dict(...)`` for eval — NOT a
    FSDP-specific format. ``model.scorer.model.fsdp_module`` is the FSDP-wrapped
    ``_LogitsAdapter(esmc)``; its keys are prefixed ``esmc.`` (the adapter
    attribute), so we strip that prefix to recover plain ESM-C key names.
    """
    fsdp_module = model.scorer.model.fsdp_module
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_module, StateDictType.FULL_STATE_DICT, save_policy):
        sd = fsdp_module.state_dict()
    if is_main(rank):
        # Strip the `_LogitsAdapter.esmc.` prefix so the consolidated checkpoint
        # has plain ESM-C key names and loads into a vanilla ESMCScorer.
        prefix = "esmc."
        sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
        os.makedirs(args.out, exist_ok=True)
        ckpt_path = os.path.join(args.out, f"{args.run_name}_{epoch + 1}.pt")
        torch.save(sd, ckpt_path)
        rank0_print(rank, f"[ckpt] saved consolidated state_dict -> {ckpt_path} "
                          f"({len(sd)} tensors)")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=["esmc_600m", "esmc_6b"])
    ap.add_argument("--stage", default="skempi", choices=["skempi"],
                    help="only the SKEMPI (stage-2) binding finetune is implemented")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="optional init weights (a consolidated ESM-C state_dict)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2000,
                    help="TOKEN budget per forward (B = max(1, batch_size // L))")
    ap.add_argument("--out", type=str, required=True, help="checkpoint/log output dir")
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first K train complexes")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="cap on total optimizer steps (smoke / pilot); None = full epochs")
    ap.add_argument("--use_flash_attn", type=int, default=1, choices=[0, 1],
                    help="1 on Ibex (flash-attn installed); 0 locally")
    ap.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--activation_checkpointing", type=int, default=1, choices=[0, 1])
    ap.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"],
                    help="gloo = CPU fallback for heterogeneous-GPU local smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdb_dict_cache_path", type=str,
                    default="cache/skempi_train_pdb_dict.pkl")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # 1) Distributed init.
    rank, local_rank, world_size = setup_dist(args.backend)
    if args.backend == "nccl":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    rank0_print(rank, f"[fsdp] backbone={args.backbone} world_size={world_size} "
                      f"backend={args.backend} mixed_precision={args.mixed_precision} "
                      f"flash_attn={bool(args.use_flash_attn)} lr={args.lr} "
                      f"epochs={args.epochs} batch_size(tokens)={args.batch_size}")

    # 2) Build the scorer (loads ESM-C onto this rank's device), grab the bare
    #    backbone module to wrap. build_scorer() defaults flash-attn ON, which
    #    crashes locally (flash-attn not installed) — so for the local smoke
    #    (--use_flash_attn 0) build the ESMCScorer directly with flash OFF.
    if bool(args.use_flash_attn):
        scorer = build_scorer(args.backbone, device=str(device), pdb_dir=SKEMPI_PDB_DIR)
    else:
        from track_esmc.esmc_scorer import ESMCScorer
        scorer = ESMCScorer(args.backbone, device=str(device), use_flash_attn=False)

    model_to_wrap = scorer.backbone_module  # ESM-C `model` (has the 36/80 blocks)

    # Optional init weights (resume / stage-1 -> stage-2) BEFORE wrapping.
    if args.checkpoint is not None:
        sd = torch.load(args.checkpoint, map_location="cpu")
        model_to_wrap.load_state_dict(sd)
        rank0_print(rank, f"[fsdp] loaded init checkpoint {args.checkpoint}")

    # 3) Activation checkpointing + FSDP wrap (of a logits-only adapter), then
    #    reassign a shim to scorer.model so StaBddG forwards THROUGH the FSDP
    #    module and ESMCScorer still reads `out.sequence_logits`.
    wrapped = wrap_backbone_fsdp(model_to_wrap, local_rank, args.mixed_precision,
                                 args.backend, bool(args.activation_checkpointing))
    scorer.model = _FSDPModelShim(wrapped)
    rank0_print(rank, f"[fsdp] backbone wrapped (FULL_SHARD, "
                      f"activation_ckpt={bool(args.activation_checkpointing)}, "
                      f"mp={args.mixed_precision}); ESM-C blocks sharded across "
                      f"{world_size} rank(s)")

    # 4) StaBddG + train.
    model = StaBddG(scorer)
    train_dataset = build_train_dataset(args.pdb_dict_cache_path, limit=args.limit)
    rank0_print(rank, f"[fsdp] train complexes = {len(train_dataset)}")

    train(model, train_dataset, args, device, rank)

    # 6) Clean shutdown.
    dist.barrier()
    dist.destroy_process_group()
    rank0_print(rank, "[fsdp] done.")


if __name__ == "__main__":
    main()
