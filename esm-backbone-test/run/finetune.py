"""
finetune.py — backbone-agnostic, single-GPU StaB-ddG finetune (two stages).

Drives any backbone behind the :class:`common.scorer.SequenceScorer` interface
(ProteinMPNN / ESM-C / ESM3) through one of two training recipes:

  - ``--stage skempi``    : SKEMPI binding ddG (stage-2). Ported verbatim from
    ``skempi_finetune.py``'s inner loop. Each sample is a complex decomposed into
    complex/binder1/binder2 sub-structures, scored through ``StaBddG.binding_ddG``.
  - ``--stage stability`` : Megascale single-domain folding-stability ddG
    (stage-1). Ported from the original ``stability_finetune.py``: one AlphaFold
    structure per domain, scored DIRECTLY via ``scorer.folding_ddG(domain, chunk)``
    (NO complex/binder decomposition, NO StaBddG wrapper).

The optimizer trains ``scorer.backbone_module.parameters()`` and each epoch saves
``scorer.backbone_module.state_dict()`` — so the same loop works for every
backbone without knowing its internals. The two-stage pipeline (Megascale ->
SKEMPI) is chained by passing the stage-1 checkpoint to stage-2 via
``--checkpoint``.

Usage
-----
    # SKEMPI (stage-2) smoke (one train complex, two epochs) on the A4500:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python -m run.finetune --backbone esmc_600m --stage skempi \
        --epochs 2 --limit 1 --out cache/ft_smoke --run_name esmc_smoke

    # Megascale stability (stage-1) smoke (three train domains, two epochs):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python -m run.finetune --backbone esmc_600m --stage stability \
        --epochs 2 --limit 3 --out cache/s1_smoke --run_name esmc_s1

CLI
---
    --backbone {mpnn,esmc_600m,esmc_6b,esm3}   required
    --stage {skempi,stability}                 stage-2 (binding) or stage-1 (folding)
    --checkpoint <path>                        optional init weights for the backbone
                                               (chains stage-1 -> stage-2)
    --lr <float>                               default 1e-5
    --epochs <int>                             required
    --batch_size <int>                         TOKEN budget per forward (NOT #seqs)
    --out <dir>                                checkpoint + log output dir
    --run_name <str>                           prefix for checkpoints/log
    --device cuda                              torch device
    --limit K                                  first K train complexes/domains (smoke)
    --pdb_dir <dir>                            (stability) AlphaFold domain PDB dir
    --stability_csv <path>                     (stability) Tsuboyama Dataset2/3 CSV
    --mega_splits <path>                       (stability) mega_splits.pkl

Memory note
-----------
``--batch_size`` is a TOKEN budget; the number of sequences per forward is
``B = max(1, batch_size // L)`` where ``L`` is the per-row sequence length.

TRAINING needs much smaller batches than eval: ``binding_ddG`` runs SIX scorer
forwards per step (complex / binder1 / binder2, each mut + wt) whose activation
graphs are ALL retained until ``loss.backward()``, so the peak memory is the sum
across all six — far larger than the no-grad eval forward at the same budget.

Backbone-specific TRAINING defaults (measured on the 20GB A4500, complex L=372):
  - ``esmc_600m`` : 2000 tokens (B~=5 at L=372 -> ~12 GB peak; 10000 OOMs at ~40 GB).
  - ``esm3``      : 500 tokens (forces B=1 for any L>=500 -> a stable ~19.6 GB
                    peak that runs the full 99-batch 1A22 inner loop with 0
                    failures over 2 epochs). The 1.4B model + six retained graphs
                    + O(B*L^2) attention puts B=2/1000 at ~18.6-19.6 GB, which is
                    too close to the 20 GB ceiling and OOMs intermittently as
                    allocator fragmentation accumulates across the inner loop.
  - ``mpnn``      : 10000 tokens (tiny model, matches the original recipe).
Long complexes collapse to B=1 under these budgets (the intended token-budget
behaviour). NOTE: the very longest train complex (3VR6, L=3397) at B=1 may still
exceed 20 GB for ESM3 because attention is O(L^2); the loop SKIPS such complexes
(catches the OOM, clears the allocator cache, continues) rather than crashing.
A full ESM3 train run on the A4500 may therefore drop a few giant complexes —
out of scope for this smoke, which uses 1A22 (L=372).

``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is set at import to reduce
allocator fragmentation under the large, variable-size training activations.
"""

import argparse
import datetime
import math
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path / env bootstrap (mirrors run/zeroshot_eval.py for standalone runs).     #
# Adds the worktree root (for `stabddg`) and esm-backbone-test/ (for `common`, #
# `track_*`, `run`) to sys.path, and sets offline-HF env BEFORE any import.    #
# --------------------------------------------------------------------------- #
_RUN_DIR = Path(__file__).resolve().parent          # .../esm-backbone-test/run
_PKG_ROOT = _RUN_DIR.parent                          # .../esm-backbone-test
_WORKTREE_ROOT = _PKG_ROOT.parent                    # .../esm-replace
for _p in (str(_WORKTREE_ROOT), str(_PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Reduce CUDA allocator fragmentation under large, variable-size training
# activations (the six retained scorer graphs per ddG step).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pickle

import numpy as np
import pandas as pd
import torch

from common.build_scorer import build_scorer
from common.stab_model import StaBddG
from stabddg.ppi_dataset import SKEMPIDataset
from stabddg.mpnn_utils import StructureDataset, parse_PDB

# SKEMPI (stage-2) inputs, resolved against the worktree root so the entrypoint
# is cwd-independent (data/ lives at the worktree root alongside stabddg/).
SKEMPI_CSV = str(_WORKTREE_ROOT / "data/SKEMPI/filtered_skempi.csv")
SKEMPI_TRAIN_SPLIT = str(_WORKTREE_ROOT / "data/SKEMPI/train_pdb.pkl")
SKEMPI_PDB_DIR = str(_WORKTREE_ROOT / "data/SKEMPI2_PDBs")

# Megascale (stage-1, folding-stability) inputs, also resolved against the
# worktree root. These mirror stability_finetune.py's argparse defaults.
MEGA_PDB_DIR = str(_WORKTREE_ROOT / "data/AlphaFold_model_PDBs")
MEGA_STABILITY_CSV = str(
    _WORKTREE_ROOT
    / "data/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
)
MEGA_SPLITS = str(_WORKTREE_ROOT / "data/rocklin/mega_splits.pkl")

# StaB alphabet (mpnn_utils.py:211); ALPHABET in stability_finetune.py.
_STAB_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Per-backbone default TOKEN budgets for TRAINING (smaller than eval: six
# retained scorer graphs per step). Measured on the 20GB A4500 — see module
# docstring. ESM-C training OOMs at the eval budget of 10000; ESM3 attention is
# O(B*L^2). MPNN is tiny and keeps the original recipe's 10000.
DEFAULT_BATCH_SIZE = {"mpnn": 10000, "esmc_600m": 2000, "esmc_6b": 2000, "esm3": 500, "esm3_stab": 1000}


# --------------------------------------------------------------------------- #
# Large-model full-finetune recipe: optimizer + LR schedule helpers.          #
#                                                                              #
# These are DEFAULTS-PRESERVING. At the original defaults                      #
# (--optimizer adam, --weight_decay 0.0, --warmup_frac 0.0,                    #
# --lr_schedule constant) ``_build_optimizer`` returns exactly the previous    #
# ``torch.optim.Adam(params, lr=args.lr)`` and ``_build_scheduler`` returns a  #
# LambdaLR whose multiplier is a CONSTANT 1.0 for every step — so the loss     #
# path / LR trajectory is bit-for-bit identical to the old loop (verified by   #
# the 2-step defaults check). AdamW / weight_decay / warmup / cosine only      #
# engage when explicitly requested.                                            #
# --------------------------------------------------------------------------- #
def _build_optimizer(params, args):
    """Adam (default) or AdamW with weight decay, on the backbone parameters.

    At defaults (optimizer=adam, weight_decay=0.0) this is identical to the
    original ``torch.optim.Adam(params, lr=args.lr)``.
    """
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(params, lr=args.lr)


def _select_trainable_params(scorer, backbone):
    """Params to optimize. For parameter-efficient scorers (esm3_stab) freeze the
    base and return only LoRA+head; else preserve the original full-FT behavior
    (all backbone params, requires_grad_(True)).

    ``backbone`` is accepted for signature symmetry but not dispatched on (selection is via ``hasattr`` on the scorer).
    """
    bb = scorer.backbone_module
    if hasattr(scorer, "trainable_parameters") and hasattr(scorer, "freeze_base"):
        scorer.freeze_base()
        bb.train()
        return scorer.trainable_parameters()
    bb.train()
    bb.requires_grad_(True)
    return bb.parameters()


def _save_backbone_ckpt(scorer, backbone, path):
    """Save adapters-only for parameter-efficient scorers, else full state_dict.

    ``backbone`` is accepted for signature symmetry but not dispatched on (selection is via ``hasattr`` on the scorer).
    """
    if hasattr(scorer, "save_adapters"):
        scorer.save_adapters(path)
    else:
        torch.save(scorer.backbone_module.state_dict(), path)


def _estimate_total_steps(args, *, complex_lengths_and_counts):
    """Estimate total optimizer steps = epochs * (per-epoch chunk count).

    ``complex_lengths_and_counts`` is an iterable of ``(L, N)`` per domain/complex
    where ``L`` is the per-row sequence length used for the token-budget split and
    ``N`` is the number of mutant rows. Per the inner loops, each domain/complex
    contributes ``ceil(N / M)`` steps with ``M = max(1, batch_size // L)``.
    """
    per_epoch = 0
    for L, N in complex_lengths_and_counts:
        if N <= 0:
            continue
        M = max(1, args.batch_size // max(1, L))
        per_epoch += math.ceil(N / M)
    return max(1, args.epochs * per_epoch)


def _build_scheduler(optimizer, args, total_steps):
    """LambdaLR: linear warmup over ``warmup_frac*total_steps`` then constant or
    cosine decay to ~0 over the remainder.

    At defaults (warmup_frac=0.0, lr_schedule=constant) the multiplier is a
    CONSTANT 1.0 at every step, so ``scheduler.step()`` is a no-op on the LR —
    bit-for-bit identical to having no scheduler at all.
    """
    warmup_steps = int(args.warmup_frac * total_steps)

    def lr_lambda(step):
        # step is the number of completed scheduler.step() calls (0-indexed).
        if warmup_steps > 0 and step < warmup_steps:
            # Linear warmup from ~0 up to 1.0 (peak LR) at the end of warmup.
            return float(step + 1) / float(warmup_steps)
        if args.lr_schedule == "cosine":
            denom = max(1, total_steps - warmup_steps)
            progress = float(step - warmup_steps) / float(denom)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        # constant
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_train_dataset(pdb_dict_cache_path, limit=None):
    """Build the SKEMPI train-split dataset (optionally first K complexes)."""
    dataset = SKEMPIDataset(
        csv_path=SKEMPI_CSV,
        split_path=SKEMPI_TRAIN_SPLIT,
        pdb_dir=SKEMPI_PDB_DIR,
        pdb_dict_cache_path=pdb_dict_cache_path,
        af_apo_structures=False,
    )
    if limit is not None:
        # SKEMPIDataset is list-backed (self.data); restrict to first K complexes.
        dataset.data = dataset.data[:limit]
    return dataset


def build_stability_dataset(pdb_dir, stability_csv, mega_splits, limit=None):
    """Build the Megascale (stage-1) folding-stability train set.

    Faithful port of ``stability_finetune.py``'s data-loading code (read
    ``mega_splits.pkl`` -> parse the train-split AlphaFold domain PDBs via
    ``parse_PDB`` -> build ``ddG_data[f'{name}.pdb'] = {'ddG', 'mut_seqs'}`` from
    the Tsuboyama Dataset2/3 CSV). Only the TRAIN split is used (finetuning loops
    over train domains; val/test scoring is out of scope for this entrypoint).

    Returns ``(dataset_train, ddG_data)`` where ``dataset_train`` is a
    ``StructureDataset`` of parsed AF domains (each masked on chain 'A', matching
    the original) and ``ddG_data`` maps ``f'{name}.pdb' -> {'ddG'[N], 'mut_seqs'
    [N, L] int tensor}``. The training loop reads ``sample['name']`` to index
    ``ddG_data`` exactly as the original does.

    ``limit`` restricts to the first K train domains (numeric/list order of the
    split). When set, the CSV read is restricted to just those domains' rows via a
    chunked filter so the 666 MB file does not have to be fully materialised — a
    smoke-only fast path. With ``limit=None`` the full CSV is read (one-time, slow:
    see the module concerns / results note).
    """
    # 1) Read split file; use the TRAIN split (matches the finetune loop scope).
    with open(mega_splits, "rb") as f:
        splits = pickle.load(f)
    train_names = list(splits["train"])
    if limit is not None:
        train_names = train_names[:limit]

    # 2) Parse AF domain structures. The original maps a split name (e.g.
    #    'r10_437_TrROS_Hall.pdb') to a file by stripping after '.pdb' and
    #    replacing '|' -> ':' (a handful of WT_names embed '|', though none in the
    #    current train/val/test splits do).
    pdb_dict_train = []
    for name in train_names:
        fname = name.split(".pdb", 1)[0] + ".pdb"
        fname = fname.replace("|", ":")
        path = os.path.join(pdb_dir, fname)
        pdb_dict_train.append(parse_PDB(path)[0])

    # 3) Build ddG_data from the Tsuboyama CSV (verbatim filters from the original:
    #    keep ddG_ML != '-', drop ins/del mutations, drop the wt row, match on
    #    WT_name == the RAW split name).
    keep_cols = ["WT_name", "mut_type", "aa_seq", "ddG_ML"]
    wt_name_set = set(train_names)

    if limit is not None:
        # Smoke fast path: stream the CSV in chunks and keep only the rows whose
        # WT_name is one of the (few) limited domains, so we never hold the full
        # 666 MB frame in memory. Equivalent result to the full read + filter.
        kept = []
        for chunk in pd.read_csv(
            stability_csv, usecols=keep_cols, low_memory=True, chunksize=200_000
        ):
            kept.append(chunk[chunk["WT_name"].isin(wt_name_set)])
        df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=keep_cols)
    else:
        df = pd.read_csv(stability_csv, usecols=keep_cols, low_memory=False)

    dataset_3 = df[df["ddG_ML"] != "-"]
    dataset_3_noindel = dataset_3.loc[
        ~dataset_3.mut_type.str.contains("ins")
        & ~dataset_3.mut_type.str.contains("del"),
        :,
    ].reset_index(drop=True)

    ddG_data = {}
    for name in train_names:
        cleaned_name = name.split(".pdb", 1)[0] + ".pdb"
        cleaned_name = cleaned_name.replace("|", ":")
        mut_df = dataset_3_noindel[
            (dataset_3_noindel["WT_name"] == name)
            & (dataset_3_noindel["mut_type"] != "wt")
        ]
        ddG_data[cleaned_name] = {
            "mut_seqs": mut_df["aa_seq"].to_list(),
            "ddG": mut_df["ddG_ML"].to_numpy(dtype=np.float32),
        }

    # 4) Featurize mutant sequences as StaB-alphabet index tensors (verbatim).
    for name, entry in ddG_data.items():
        index_matrix = []
        for s in entry["mut_seqs"]:
            indices = np.asarray([_STAB_ALPHABET.index(a) for a in s], dtype=np.int64)
            index_matrix.append(indices)
        index_matrix = np.vstack(index_matrix)
        ddG_data[name]["mut_seqs"] = torch.from_numpy(index_matrix)
        ddG_data[name]["ddG"] = torch.tensor(entry["ddG"])

    # 5) Mask all input chains (single-chain AF domains -> chain 'A'), matching
    #    the original. Harmless for ESM-C/ESM3 (they read seq/coords, not masks).
    for d in pdb_dict_train:
        d["masked_list"] = ["A"]
        d["visible_list"] = []

    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=3000)
    return dataset_train, ddG_data


def finetune(model, train_dataset, args, device):
    """Port of ``skempi_finetune.py``'s inner SKEMPI training loop.

    Trains ``model.scorer.backbone_module`` with Adam + MSE on binding ddG,
    using the original token-budget batching and per-complex shuffling. Logs
    per-epoch mean train loss to stdout and a small CSV, and saves a checkpoint
    (the backbone state_dict) after every epoch.
    """
    scorer = model.scorer
    optimizer = _build_optimizer(_select_trainable_params(scorer, args.backbone), args)
    # total_steps = epochs * sum over complexes of ceil(N_muts / M); used for the
    # warmup / cosine schedule. At defaults the schedule multiplier is a constant
    # 1.0, so this estimate has no effect on the LR.
    total_steps = _estimate_total_steps(
        args,
        complex_lengths_and_counts=(
            (s["complex_mut_seqs"].shape[1], s["complex_mut_seqs"].shape[0])
            for s in train_dataset
        ),
    )
    scheduler = _build_scheduler(optimizer, args, total_steps)
    ddG_loss_fn = torch.nn.MSELoss()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"{args.run_name}_train_log.csv")
    with open(log_path, "w") as lf:
        lf.write("timestamp,epoch,mean_loss,lr\n")
    print(f"[train log] per-epoch mean loss -> {log_path}", flush=True)
    print(f"[sched] optimizer={args.optimizer} weight_decay={args.weight_decay} "
          f"warmup_frac={args.warmup_frac} lr_schedule={args.lr_schedule} "
          f"total_steps~={total_steps}", flush=True)

    for epoch in range(args.epochs):
        epoch_losses = []
        for sample in train_dataset:
            try:
                complex, binder1, binder2 = (
                    sample["complex"], sample["binder1"], sample["binder2"]
                )
                cms = sample["complex_mut_seqs"]
                b1ms = sample["binder1_mut_seqs"]
                b2ms = sample["binder2_mut_seqs"]
                ddG = sample["ddG"].float().to(device)

                N = cms.shape[0]
                # token budget -> number of sequences per forward batch
                M = max(1, args.batch_size // cms.shape[1])

                # shuffle mutants within this complex (matches original recipe)
                perm = torch.randperm(N)
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
                    scheduler.step()
                    optimizer.zero_grad()
                    epoch_losses.append(loss.item())
            except Exception as e:
                # Original recipe: skip a failed complex and continue. On a CUDA
                # OOM (heavy ESM3 backbone near the 20 GB ceiling) we additionally
                # drop the half-built graph and reclaim the allocator's cached
                # blocks so fragmentation from one failure does not compound into
                # the next complex/epoch.
                print("Failed on", sample["complex"]["name"], e, flush=True)
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        mean_loss = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch + 1}/{args.epochs}] mean train loss = {mean_loss:.6f} "
              f"lr = {cur_lr:.3e}", flush=True)
        with open(log_path, "a") as lf:
            lf.write(f"{datetime.datetime.now().isoformat(timespec='seconds')},"
                     f"{epoch + 1},{mean_loss:.6f},{cur_lr:.6e}\n")

        ckpt_path = os.path.join(args.out, f"{args.run_name}_{epoch + 1}.pt")
        _save_backbone_ckpt(scorer, args.backbone, ckpt_path)
        print(f"[ckpt] saved {ckpt_path}", flush=True)


def finetune_stability(scorer, dataset_train, ddG_data, args, device):
    """Port of ``stability_finetune.py``'s inner Megascale training loop.

    Single-domain FOLDING ddG: each ``sample`` is ONE AlphaFold structure and the
    loss is MSE on ``scorer.folding_ddG(sample, chunk)`` directly — NO complex /
    binder decomposition and NO ``StaBddG`` wrapper (that is the stage-2 binding
    path). Trains ``scorer.backbone_module`` with Adam + per-chunk token-budget
    batching and per-domain mutant shuffling, mirroring the original recipe and
    the stage-2 loop's logging / per-epoch checkpoint save.
    """
    optimizer = _build_optimizer(_select_trainable_params(scorer, args.backbone), args)
    # total_steps = epochs * sum over domains of ceil(N_muts / M). Domains map to
    # ddG_data[f'{name}.pdb']['mut_seqs'] (shape [N, L]); domains absent from
    # ddG_data (or with N==0) contribute 0 steps, matching the inner loop.
    def _stability_lengths_and_counts():
        for s in dataset_train:
            entry = ddG_data.get(f"{s['name']}.pdb")
            if entry is None:
                continue
            ms = entry["mut_seqs"]
            if ms.shape[0] == 0:
                continue
            yield ms.shape[1], ms.shape[0]

    total_steps = _estimate_total_steps(
        args, complex_lengths_and_counts=_stability_lengths_and_counts()
    )
    scheduler = _build_scheduler(optimizer, args, total_steps)
    ddG_loss_fn = torch.nn.MSELoss()

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"{args.run_name}_train_log.csv")
    with open(log_path, "w") as lf:
        lf.write("timestamp,epoch,mean_loss,lr\n")
    print(f"[train log] per-epoch mean loss -> {log_path}", flush=True)
    print(f"[sched] optimizer={args.optimizer} weight_decay={args.weight_decay} "
          f"warmup_frac={args.warmup_frac} lr_schedule={args.lr_schedule} "
          f"total_steps~={total_steps}", flush=True)

    for epoch in range(args.epochs):
        epoch_losses = []
        for sample in dataset_train:
            try:
                pdb_name = sample["name"]
                key = f"{pdb_name}.pdb"
                ddG = ddG_data[key]["ddG"].float().to(device)
                mut_seqs = ddG_data[key]["mut_seqs"]

                N = mut_seqs.shape[0]
                if N == 0:
                    continue
                # token budget -> number of sequences per forward batch
                M = max(1, args.batch_size // mut_seqs.shape[1])

                # shuffle mutants within this domain (matches original recipe)
                perm = torch.randperm(N)
                ddG = ddG[perm]
                mut_seqs = mut_seqs[perm]

                for i in range(0, N, M):
                    B = min(N - i, M)
                    pred = scorer.folding_ddG(sample, mut_seqs[i:i + B])
                    loss = ddG_loss_fn(pred, ddG[i:i + B])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    epoch_losses.append(loss.item())
            except Exception as e:
                # Mirror the stage-2 loop: skip a failed domain, drop the half-built
                # graph and reclaim cached allocator blocks (heavy ESM3 backbone
                # near the 20 GB ceiling on long domains) so one failure does not
                # compound into the next.
                print("Failed on", sample.get("name"), repr(e), flush=True)
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        mean_loss = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("nan")
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch + 1}/{args.epochs}] mean train loss = {mean_loss:.6f} "
              f"lr = {cur_lr:.3e}", flush=True)
        with open(log_path, "a") as lf:
            lf.write(f"{datetime.datetime.now().isoformat(timespec='seconds')},"
                     f"{epoch + 1},{mean_loss:.6f},{cur_lr:.6e}\n")

        ckpt_path = os.path.join(args.out, f"{args.run_name}_{epoch + 1}.pt")
        _save_backbone_ckpt(scorer, args.backbone, ckpt_path)
        print(f"[ckpt] saved {ckpt_path}", flush=True)


def _load_checkpoint_for_esm(scorer, backbone, checkpoint):
    """Load a prior-stage backbone state_dict into an ESM-C / ESM3 scorer.

    ``build_scorer`` honours ``--checkpoint`` only for the ``mpnn`` backbone (it
    loads the ProteinMPNN weights there). For ESM-C / ESM3 the factory ignores it,
    so to chain stage-1 -> stage-2 (or resume) we load the consolidated backbone
    state_dict into ``scorer.backbone_module`` AFTER the scorer is built. No-op for
    mpnn (already handled) or when ``checkpoint`` is None.

    For parameter-efficient scorers exposing ``load_adapters`` (e.g. esm3_stab), this loads only the LoRA+head adapters and returns early.
    """
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


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backbone", required=True,
                    choices=["mpnn", "esmc_600m", "esmc_6b", "esm3", "esm3_stab"])
    ap.add_argument("--stage", default="skempi", choices=["skempi", "stability"],
                    help="skempi = SKEMPI binding (stage-2); "
                         "stability = Megascale folding stability (stage-1)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="optional init weights for the backbone "
                         "(mpnn: StaB-ddG ckpt; esm*: backbone state_dict). "
                         "Chains stage-1 -> stage-2.")
    ap.add_argument("--lr", type=float, default=1e-5)
    # Large-model full-FT recipe options. Defaults preserve the original behavior
    # (Adam, no weight decay, no warmup, constant LR).
    ap.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                    help="adam (default, original behavior) or adamw (decoupled "
                         "weight decay)")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="weight decay (only meaningful with --optimizer adamw)")
    ap.add_argument("--warmup_frac", type=float, default=0.0,
                    help="fraction of total optimizer steps used for linear LR "
                         "warmup (0.0 = no warmup, original behavior)")
    ap.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"],
                    help="LR schedule after warmup: constant (default) or cosine "
                         "decay to ~0")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="TOKEN budget per forward batch "
                         "(default: 500 esm3, 2000 esmc, 10000 mpnn)")
    ap.add_argument("--out", type=str, required=True, help="checkpoint/log output dir")
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="first K train complexes (skempi) / domains (stability)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdb_dict_cache_path", type=str,
                    default="cache/skempi_train_pdb_dict.pkl",
                    help="(skempi) structure-dict cache (built on first run)")
    # Stage-1 (stability) inputs — default to the Megascale paths under data/.
    ap.add_argument("--pdb_dir", type=str, default=MEGA_PDB_DIR,
                    help="(stability) AlphaFold domain PDB directory")
    ap.add_argument("--stability_csv", type=str, default=MEGA_STABILITY_CSV,
                    help="(stability) Tsuboyama Dataset2/3 ddG CSV")
    ap.add_argument("--mega_splits", type=str, default=MEGA_SPLITS,
                    help="(stability) mega_splits.pkl train/val/test domain lists")
    args = ap.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.backbone]

    torch.manual_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "cuda" else torch.device("cpu"))

    if args.stage == "stability":
        # Stage-1: Megascale single-domain folding stability. The ESM3 scorer must
        # resolve domain['name'] to the AlphaFold PDBs (NOT the SKEMPI dir), so the
        # scorer is built with the AF pdb_dir.
        dataset_train, ddG_data = build_stability_dataset(
            args.pdb_dir, args.stability_csv, args.mega_splits, limit=args.limit
        )
        scorer = build_scorer(
            args.backbone,
            device=str(device),
            checkpoint=args.checkpoint,
            pdb_dir=args.pdb_dir,
        )
        _load_checkpoint_for_esm(scorer, args.backbone, args.checkpoint)
        scorer.to(device)

        print(f"[finetune] backbone={args.backbone} stage=stability "
              f"lr={args.lr} epochs={args.epochs} "
              f"batch_size(tokens)={args.batch_size} "
              f"domains={len(dataset_train)} device={device}", flush=True)

        finetune_stability(scorer, dataset_train, ddG_data, args, device)
        return

    # Stage-2: SKEMPI binding.
    train_dataset = build_train_dataset(args.pdb_dict_cache_path, limit=args.limit)

    scorer = build_scorer(
        args.backbone,
        device=str(device),
        checkpoint=args.checkpoint,
        pdb_dir=SKEMPI_PDB_DIR,
    )
    _load_checkpoint_for_esm(scorer, args.backbone, args.checkpoint)
    model = StaBddG(scorer).to(device)

    print(f"[finetune] backbone={args.backbone} stage={args.stage} "
          f"lr={args.lr} epochs={args.epochs} batch_size(tokens)={args.batch_size} "
          f"complexes={len(train_dataset)} device={device}", flush=True)

    finetune(model, train_dataset, args, device)


if __name__ == "__main__":
    main()
