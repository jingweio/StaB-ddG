"""
lr_sweep.py — learning-rate sweep driver (SKEMPI-only, cheap).

For each LR in ``--lrs``:
  1. Finetune the backbone from pretrained weights (SKEMPI-only, stage-2).
  2. Evaluate on the SKEMPI test split.
  3. Compute per-interface Spearman via ``baselines.eval_utils.compute_metrics``.
  4. Append a row to ``{out_dir}/sweep.csv``.

Print and persist the argmax-LR to ``{out_dir}/best.json``.

If a single LR run errors (finetune crash, eval crash, metric crash), the error
is logged and the loop continues with the remaining LRs — the sweep never aborts.

Usage
-----
    # 2-LR smoke on the A4500:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
        python -m run.lr_sweep \\
            --backbone esmc_600m --lrs "1e-6,5e-6" \\
            --epochs 1 --batch_size 2000 --device cuda \\
            --limit_train 2 --limit_eval 3

    # Full sweep (4 LRs, ibex):
    python -m run.lr_sweep \\
        --backbone esmc_600m --lrs "1e-6,5e-6,1e-5,3e-5" \\
        --epochs 5 --batch_size 25000 --device cuda

CLI
---
    --backbone      {mpnn,esmc_600m,esmc_6b,esm3}     required
    --lrs           comma-separated floats             default "1e-6,5e-6,1e-5,3e-5"
    --epochs        int                                required
    --batch_size    int (TOKEN budget)                 default: backbone default
    --device        str                                default "cuda"
    --limit_train   K  (first K train complexes)       default None (all)
    --limit_eval    K  (first K eval complexes)        default None (all)
    --out_dir       path                               default results/sweep_{backbone}
    --ensemble      int                                default 1
    --seed          int                                default 0
"""

import argparse
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path

# --------------------------------------------------------------------------- #
# Path / env bootstrap (mirrors run/finetune.py for standalone runs).          #
# --------------------------------------------------------------------------- #
_RUN_DIR = Path(__file__).resolve().parent          # .../esm-backbone-test/run
_PKG_ROOT = _RUN_DIR.parent                          # .../esm-backbone-test
_WORKTREE_ROOT = _PKG_ROOT.parent                    # .../esm-replace
for _p in (str(_WORKTREE_ROOT), str(_PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from common.build_scorer import build_scorer
from common.stab_model import StaBddG
from run.eval import eval_dataset, write_csv
from run.finetune import (
    DEFAULT_BATCH_SIZE,
    SKEMPI_PDB_DIR,
    build_train_dataset,
    finetune,
)
from stabddg.ppi_dataset import SKEMPIDataset

# SKEMPI test split inputs (resolved against worktree root, cwd-independent)
SKEMPI_CSV = str(_WORKTREE_ROOT / "data/SKEMPI/filtered_skempi.csv")
SKEMPI_TEST_SPLIT = str(_WORKTREE_ROOT / "data/SKEMPI/test_pdb.pkl")

# Evaluation token budget per backbone (no-grad, so larger than finetune)
EVAL_BATCH_SIZE = {
    "mpnn": 10000,
    "esmc_600m": 10000,
    "esmc_6b": 10000,
    "esm3": 2000,
}


def _build_eval_dataset(limit=None):
    """Build the SKEMPI test-split dataset (optionally first K complexes)."""
    dataset = SKEMPIDataset(
        csv_path=SKEMPI_CSV,
        split_path=SKEMPI_TEST_SPLIT,
        pdb_dir=SKEMPI_PDB_DIR,
        pdb_dict_cache_path="cache/skempi_test_pdb_dict.pkl",
        af_apo_structures=False,
    )
    if limit is not None:
        dataset.data = dataset.data[:limit]
    return dataset


class _FinetuneArgs:
    """Minimal args namespace for run.finetune.finetune()."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _safe_metric(metrics, key, fallback=float("nan")):
    """Return metrics[key], replacing ZeroDivisionError / missing → fallback."""
    try:
        v = metrics.get(key, fallback)
        return float("nan") if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
    except Exception:
        return fallback


def run_one_lr(
    backbone,
    lr,
    epochs,
    batch_size,
    device,
    out_dir,
    limit_train,
    limit_eval,
    ensemble,
    seed,
):
    """Run finetune + eval + metrics for a single LR.

    Returns a dict with keys:
        lr, epochs, per_interface_spearman, overall_spearman, n_qualifying_complexes
    Raises on unrecoverable errors (caller catches and logs).
    """
    lr_tag = f"{lr:.0e}".replace("-0", "-").replace("+0", "")  # e.g. "1e-6"
    run_name = f"{backbone}_lr{lr_tag}"
    ckpt_dir = os.path.join(out_dir, "ckpts", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Build model + train dataset                                       #
    # ------------------------------------------------------------------ #
    torch.manual_seed(seed)
    train_dataset = build_train_dataset(
        pdb_dict_cache_path="cache/skempi_train_pdb_dict.pkl",
        limit=limit_train,
    )
    scorer = build_scorer(backbone, device=str(device), checkpoint=None,
                          pdb_dir=SKEMPI_PDB_DIR)
    model = StaBddG(scorer).to(device)

    print(f"\n[sweep] backbone={backbone} lr={lr:.2e} epochs={epochs} "
          f"batch_size={batch_size} limit_train={limit_train} "
          f"complexes={len(train_dataset)}", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Finetune                                                          #
    # ------------------------------------------------------------------ #
    ft_args = _FinetuneArgs(
        backbone=backbone,
        stage="skempi",
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        out=ckpt_dir,
        run_name=run_name,
        device=str(device),
        limit=limit_train,
    )
    finetune(model, train_dataset, ft_args, device)

    # checkpoint for the final epoch
    final_ckpt = os.path.join(ckpt_dir, f"{run_name}_{epochs}.pt")
    if not os.path.exists(final_ckpt):
        raise FileNotFoundError(f"Expected checkpoint not found: {final_ckpt}")

    # ------------------------------------------------------------------ #
    # 3. Build fresh model from the final checkpoint and eval             #
    # ------------------------------------------------------------------ #
    del model, scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_scorer = build_scorer(backbone, device=str(device), checkpoint=None,
                               pdb_dir=SKEMPI_PDB_DIR)
    # Load the finetuned backbone weights
    ckpt_sd = torch.load(final_ckpt, map_location=str(device))
    eval_scorer.backbone_module.load_state_dict(ckpt_sd)

    eval_model = StaBddG(eval_scorer).to(device).eval()
    eval_dataset_obj = _build_eval_dataset(limit=limit_eval)

    eval_bs = EVAL_BATCH_SIZE.get(backbone, 10000)
    print(f"[sweep] eval backbone={backbone} lr={lr:.2e} "
          f"complexes={len(eval_dataset_obj)} ensemble={ensemble}", flush=True)

    with torch.no_grad():
        pred_df = eval_dataset(eval_model, eval_dataset_obj,
                               ensemble=ensemble,
                               batch_size=eval_bs,
                               device=str(device))

    # Write eval CSV
    eval_csv = os.path.join(out_dir, f"eval_{run_name}.csv")
    write_csv(pred_df, eval_csv)
    print(f"[sweep] wrote eval CSV: {eval_csv} ({len(pred_df)} rows)", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Compute per-interface Spearman                                   #
    # ------------------------------------------------------------------ #
    # compute_metrics expects a DataFrame with columns #Pdb, ddG, ddG_pred.
    # write_csv returns combined_df which has exactly those columns.
    import pandas as pd
    from baselines.eval_utils import compute_metrics

    eval_df = pd.read_csv(eval_csv, index_col=0)

    # Guard: compute_metrics does bootstrap=True by default (300 iters),
    # which is wasteful in a sweep; use bootstrap=False for speed.
    # Also guard against num_ppis==0 (all complexes have <10 mutations)
    # which would cause ZeroDivisionError in compute_metrics.
    try:
        metrics = compute_metrics(eval_df, bootstrap=False)
        per_iface_sp = _safe_metric(metrics, "Per Structure Spearman")
        overall_sp = _safe_metric(metrics, "Spearman")

        # Count qualifying complexes (groups with >= THRESHOLD rows)
        from baselines.eval_utils import THRESHOLD
        n_qualifying = int(
            (eval_df.groupby("#Pdb").size() >= THRESHOLD).sum()
        )
    except ZeroDivisionError:
        # No qualifying complexes (all have < 10 mutations in the eval slice)
        print(f"[sweep] WARNING: num_ppis=0 for lr={lr:.2e} "
              f"(no complex has >= {THRESHOLD} mutations in the eval slice); "
              "per_interface_spearman=NaN", flush=True)
        per_iface_sp = float("nan")
        overall_sp = eval_df["ddG"].corr(eval_df["ddG_pred"], method="spearman")
        overall_sp = float("nan") if overall_sp is None else float(overall_sp)
        n_qualifying = 0

    print(f"[sweep] lr={lr:.2e} per_iface_sp={per_iface_sp:.4f} "
          f"overall_sp={overall_sp:.4f} n_qualifying={n_qualifying}", flush=True)

    # cleanup
    del eval_model, eval_scorer, eval_dataset_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "lr": lr,
        "epochs": epochs,
        "per_interface_spearman": per_iface_sp,
        "overall_spearman": overall_sp,
        "n_qualifying_complexes": n_qualifying,
    }


def run_sweep(
    backbone,
    lrs,
    epochs,
    batch_size=None,
    device="cuda",
    limit_train=None,
    limit_eval=None,
    out_dir=None,
    ensemble=1,
    seed=0,
):
    """Run a full LR sweep and persist results.

    Parameters
    ----------
    backbone : str
        One of ``{"mpnn", "esmc_600m", "esmc_6b", "esm3"}``.
    lrs : list[float]
        Learning rates to sweep over.
    epochs : int
        Finetune epochs per LR.
    batch_size : int | None
        Token budget per forward (default: backbone default).
    device : str
        Torch device string.
    limit_train : int | None
        Restrict to first K train complexes (smoke).
    limit_eval : int | None
        Restrict to first K eval complexes (smoke).
    out_dir : str | None
        Output directory (default: ``results/sweep_{backbone}``).
    ensemble : int
        Eval ensemble size (default: 1).
    seed : int
        RNG seed.

    Returns
    -------
    rows : list[dict]
        One dict per LR with keys lr, epochs, per_interface_spearman,
        overall_spearman, n_qualifying_complexes.
    best_lr : float | None
        LR with the highest per_interface_spearman (NaN rows are ignored for
        argmax; if all rows are NaN, falls back to highest overall_spearman).
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[backbone]
    if out_dir is None:
        out_dir = f"results/sweep_{backbone}"

    os.makedirs(out_dir, exist_ok=True)
    sweep_csv = os.path.join(out_dir, "sweep.csv")

    # Write / append header if file doesn't exist yet
    csv_exists = os.path.exists(sweep_csv)
    sweep_fields = ["lr", "epochs", "per_interface_spearman",
                    "overall_spearman", "n_qualifying_complexes"]

    device_obj = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" else torch.device(device)
    )

    # Write header once before the loop (truncating any existing file so each
    # run_sweep call produces a clean sweep.csv for that backbone/config).
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_fields)
        writer.writeheader()

    rows = []
    for lr in lrs:
        print(f"\n{'=' * 70}\n[sweep] === LR={lr:.2e} ===\n{'=' * 70}", flush=True)
        try:
            row = run_one_lr(
                backbone=backbone,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                device=device_obj,
                out_dir=out_dir,
                limit_train=limit_train,
                limit_eval=limit_eval,
                ensemble=ensemble,
                seed=seed,
            )
        except Exception:
            tb = traceback.format_exc()
            print(f"[sweep] ERROR on lr={lr:.2e}:\n{tb}", flush=True)
            row = {
                "lr": lr,
                "epochs": epochs,
                "per_interface_spearman": float("nan"),
                "overall_spearman": float("nan"),
                "n_qualifying_complexes": 0,
            }

        rows.append(row)

        # Append row immediately so partial results survive a mid-sweep crash
        with open(sweep_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_fields)
            writer.writerow(row)

    # ------------------------------------------------------------------ #
    # Determine best LR                                                    #
    # ------------------------------------------------------------------ #
    def _finite_or_neg_inf(v):
        return v if (v is not None and math.isfinite(v)) else float("-inf")

    # Primary: per_interface_spearman; fallback: overall_spearman
    best_row = None
    best_val = float("-inf")
    for r in rows:
        val = _finite_or_neg_inf(r["per_interface_spearman"])
        if val == float("-inf"):
            val = _finite_or_neg_inf(r["overall_spearman"])
        if val > best_val:
            best_val = val
            best_row = r

    best_lr = best_row["lr"] if best_row is not None else None

    best_json_path = os.path.join(out_dir, "best.json")
    best_payload = {
        "best_lr": best_lr,
        "backbone": backbone,
        "epochs": epochs,
        "metric_used": "per_interface_spearman",
        "best_per_interface_spearman": (
            best_row["per_interface_spearman"] if best_row else None
        ),
        "best_overall_spearman": (
            best_row["overall_spearman"] if best_row else None
        ),
        "all_rows": rows,
    }
    with open(best_json_path, "w") as f:
        json.dump(best_payload, f, indent=2)

    print(f"\n[sweep] DONE. Best LR={best_lr:.2e} "
          f"(per_iface_sp={best_row['per_interface_spearman'] if best_row else 'N/A'})",
          flush=True)
    print(f"[sweep] Results: {sweep_csv}", flush=True)
    print(f"[sweep] Best:    {best_json_path}", flush=True)

    return rows, best_lr


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backbone", required=True,
                    choices=["mpnn", "esmc_600m", "esmc_6b", "esm3"])
    ap.add_argument("--lrs", type=str, default="1e-6,5e-6,1e-5,3e-5",
                    help="comma-separated learning rates")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="TOKEN budget per forward (default: backbone default)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit_train", type=int, default=None,
                    help="first K train complexes (smoke)")
    ap.add_argument("--limit_eval", type=int, default=None,
                    help="first K eval complexes (smoke)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="output directory (default: results/sweep_{backbone})")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="eval ensemble size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]
    out_dir = args.out_dir or f"results/sweep_{args.backbone}"

    run_sweep(
        backbone=args.backbone,
        lrs=lrs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        out_dir=out_dir,
        ensemble=args.ensemble,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
