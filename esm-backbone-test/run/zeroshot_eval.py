"""
zeroshot_eval.py — zero-shot (no-finetune) SKEMPI evaluation entrypoint.

Builds the SKEMPI test dataset, a backbone scorer via
:func:`common.build_scorer.build_scorer`, wraps it in
:class:`common.stab_model.StaBddG`, runs the backbone-agnostic eval loop
(:func:`run.eval.eval_dataset`), and writes the prediction CSV in the EXACT
original ``skempi_eval.py`` format (:func:`run.eval.write_csv`).

No finetuning is performed: the scorers are used zero-shot.  The ``mpnn``
backbone still loads its StaB-ddG checkpoint (it is not a "from-scratch" model),
but no training happens here.

Usage
-----
    python -m run.zeroshot_eval --backbone esmc_600m --out results/zs.csv [--limit 2]

CLI
---
    --backbone {mpnn,esmc_600m,esmc_6b,esm3}   required
    --out <csv>                                required output path
    --ensemble N                               MC ensemble size (default: backbone-dependent)
    --limit K                                  restrict to the first K complexes (smoke)
    --device cuda                              torch device
    --checkpoint <path>                        mpnn checkpoint (default model_ckpts/stabddg.pt)
"""

import argparse
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path / env bootstrap (mirrors conftest.py for standalone `python -m` runs).  #
# conftest.py only runs under pytest, so this entrypoint sets up the same      #
# sys.path (worktree root for `stabddg`, esm-backbone-test/ for `common` etc.) #
# and offline-HF env BEFORE importing any package.                             #
# --------------------------------------------------------------------------- #
_RUN_DIR = Path(__file__).resolve().parent          # .../esm-backbone-test/run
_PKG_ROOT = _RUN_DIR.parent                          # .../esm-backbone-test
_WORKTREE_ROOT = _PKG_ROOT.parent                    # .../esm-replace
for _p in (str(_WORKTREE_ROOT), str(_PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from common.build_scorer import build_scorer
from common.stab_model import StaBddG
from run.eval import eval_dataset, write_csv
from stabddg.ppi_dataset import SKEMPIDataset

# Default SKEMPI test-split inputs.  Resolved against the worktree root so the
# entrypoint is cwd-independent (data/ lives at the worktree root, alongside
# stabddg/), matching the original skempi_eval.py defaults.
SKEMPI_CSV = str(_WORKTREE_ROOT / "data/SKEMPI/filtered_skempi.csv")
SKEMPI_SPLIT = str(_WORKTREE_ROOT / "data/SKEMPI/test_pdb.pkl")
SKEMPI_PDB_DIR = str(_WORKTREE_ROOT / "data/SKEMPI2_PDBs")

# Ensemble defaults: ESM-C is deterministic (sequence-only), MPNN here also runs
# without re-seeding per row; ESM3 benefits from a small structure ensemble.
DEFAULT_ENSEMBLE = {"mpnn": 1, "esmc_600m": 1, "esmc_6b": 1, "esm3": 5}


def build_dataset(pdb_dict_cache_path, limit=None):
    dataset = SKEMPIDataset(
        csv_path=SKEMPI_CSV,
        split_path=SKEMPI_SPLIT,
        pdb_dir=SKEMPI_PDB_DIR,
        pdb_dict_cache_path=pdb_dict_cache_path,
        af_apo_structures=False,
    )
    if limit is not None:
        # SKEMPIDataset is list-backed (self.data); restrict to the first K complexes.
        dataset.data = dataset.data[:limit]
    return dataset


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backbone", required=True,
                    choices=["mpnn", "esmc_600m", "esmc_6b", "esm3"])
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--ensemble", type=int, default=None,
                    help="MC ensemble size (default: 1 for mpnn/esmc, 5 for esm3)")
    ap.add_argument("--limit", type=int, default=None,
                    help="restrict to the first K complexes (smoke)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="mpnn checkpoint path (default model_ckpts/stabddg.pt)")
    ap.add_argument("--batch_size", type=int, default=10000,
                    help="token budget per forward batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdb_dict_cache_path", type=str,
                    default="cache/skempi_test_pdb_dict.pkl",
                    help="structure-dict cache (built on first run)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "cuda" else torch.device("cpu"))

    ensemble = args.ensemble if args.ensemble is not None else DEFAULT_ENSEMBLE[args.backbone]

    dataset = build_dataset(args.pdb_dict_cache_path, limit=args.limit)

    scorer = build_scorer(
        args.backbone,
        device=str(device),
        checkpoint=args.checkpoint,
        pdb_dir=SKEMPI_PDB_DIR,
    )
    model = StaBddG(scorer).to(device).eval()

    with torch.no_grad():
        pred_df = eval_dataset(model, dataset, ensemble=ensemble,
                               batch_size=args.batch_size, device=str(device))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_csv(pred_df, args.out)
    print(f"Saved {len(pred_df)} predictions to {args.out} "
          f"(backbone={args.backbone}, ensemble={ensemble}, "
          f"complexes={len(dataset)})")


if __name__ == "__main__":
    main()
