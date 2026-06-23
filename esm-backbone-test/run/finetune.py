"""
finetune.py — backbone-agnostic, single-GPU SKEMPI binding finetune.

Reuses StaB-ddG's EXACT SKEMPI (stage-2) training recipe (ported verbatim from
``skempi_finetune.py``'s inner loop) but drives any backbone behind the
:class:`common.scorer.SequenceScorer` interface (ProteinMPNN / ESM-C / ESM3).

The optimizer trains ``scorer.backbone_module.parameters()`` and each epoch saves
``scorer.backbone_module.state_dict()`` — so the same loop works for every
backbone without knowing its internals.

Usage
-----
    # smoke (one train complex, two epochs) on the A4500:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python -m run.finetune --backbone esmc_600m --stage skempi \
        --epochs 2 --limit 1 --out cache/ft_smoke --run_name esmc_smoke

CLI
---
    --backbone {mpnn,esmc_600m,esmc_6b,esm3}   required
    --stage skempi                             only SKEMPI (stage-2) implemented
    --checkpoint <path>                        optional init weights for the backbone
    --lr <float>                               default 1e-5
    --epochs <int>                             required
    --batch_size <int>                         TOKEN budget per forward (NOT #seqs)
    --out <dir>                                checkpoint + log output dir
    --run_name <str>                           prefix for checkpoints/log
    --device cuda                              torch device
    --limit K                                  first K train complexes (smoke)

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
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Reduce CUDA allocator fragmentation under large, variable-size training
# activations (the six retained scorer graphs per ddG step).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from common.build_scorer import build_scorer
from common.stab_model import StaBddG
from stabddg.ppi_dataset import SKEMPIDataset

# SKEMPI (stage-2) inputs, resolved against the worktree root so the entrypoint
# is cwd-independent (data/ lives at the worktree root alongside stabddg/).
SKEMPI_CSV = str(_WORKTREE_ROOT / "data/SKEMPI/filtered_skempi.csv")
SKEMPI_TRAIN_SPLIT = str(_WORKTREE_ROOT / "data/SKEMPI/train_pdb.pkl")
SKEMPI_PDB_DIR = str(_WORKTREE_ROOT / "data/SKEMPI2_PDBs")

# Per-backbone default TOKEN budgets for TRAINING (smaller than eval: six
# retained scorer graphs per step). Measured on the 20GB A4500 — see module
# docstring. ESM-C training OOMs at the eval budget of 10000; ESM3 attention is
# O(B*L^2). MPNN is tiny and keeps the original recipe's 10000.
DEFAULT_BATCH_SIZE = {"mpnn": 10000, "esmc_600m": 2000, "esmc_6b": 2000, "esm3": 500}


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


def finetune(model, train_dataset, args, device):
    """Port of ``skempi_finetune.py``'s inner SKEMPI training loop.

    Trains ``model.scorer.backbone_module`` with Adam + MSE on binding ddG,
    using the original token-budget batching and per-complex shuffling. Logs
    per-epoch mean train loss to stdout and a small CSV, and saves a checkpoint
    (the backbone state_dict) after every epoch.
    """
    backbone = model.scorer.backbone_module

    optimizer = torch.optim.Adam(backbone.parameters(), lr=args.lr)
    ddG_loss_fn = torch.nn.MSELoss()

    backbone.train()
    backbone.requires_grad_(True)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"{args.run_name}_train_log.csv")
    with open(log_path, "w") as lf:
        lf.write("timestamp,epoch,mean_loss\n")
    print(f"[train log] per-epoch mean loss -> {log_path}", flush=True)

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
        print(f"[epoch {epoch + 1}/{args.epochs}] mean train loss = {mean_loss:.6f}",
              flush=True)
        with open(log_path, "a") as lf:
            lf.write(f"{datetime.datetime.now().isoformat(timespec='seconds')},"
                     f"{epoch + 1},{mean_loss:.6f}\n")

        ckpt_path = os.path.join(args.out, f"{args.run_name}_{epoch + 1}.pt")
        torch.save(backbone.state_dict(), ckpt_path)
        print(f"[ckpt] saved {ckpt_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backbone", required=True,
                    choices=["mpnn", "esmc_600m", "esmc_6b", "esm3"])
    ap.add_argument("--stage", default="skempi", choices=["skempi"],
                    help="only the SKEMPI (stage-2) binding finetune is implemented")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="optional init weights for the backbone "
                         "(mpnn: StaB-ddG ckpt; esm*: backbone state_dict)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="TOKEN budget per forward batch "
                         "(default: 500 esm3, 2000 esmc, 10000 mpnn)")
    ap.add_argument("--out", type=str, required=True, help="checkpoint/log output dir")
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="first K train complexes (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdb_dict_cache_path", type=str,
                    default="cache/skempi_train_pdb_dict.pkl",
                    help="structure-dict cache (built on first run)")
    args = ap.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.backbone]

    torch.manual_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "cuda" else torch.device("cpu"))

    train_dataset = build_train_dataset(args.pdb_dict_cache_path, limit=args.limit)

    scorer = build_scorer(
        args.backbone,
        device=str(device),
        checkpoint=args.checkpoint,
        pdb_dir=SKEMPI_PDB_DIR,
    )
    model = StaBddG(scorer).to(device)

    print(f"[finetune] backbone={args.backbone} stage={args.stage} "
          f"lr={args.lr} epochs={args.epochs} batch_size(tokens)={args.batch_size} "
          f"complexes={len(train_dataset)} device={device}", flush=True)

    finetune(model, train_dataset, args, device)


if __name__ == "__main__":
    main()
