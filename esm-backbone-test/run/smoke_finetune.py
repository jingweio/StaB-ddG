"""
smoke_finetune.py — GPU smoke for the SKEMPI finetune harness (Task 7).

Runs a TINY finetune (``--limit 1`` train complex, 2 epochs) for one or more
backbones and asserts the harness actually trains:

  1. no crash; per-epoch mean train loss values are finite (no NaN/Inf);
  2. checkpoints ``{out}/{run}_1.pt`` and ``_2.pt`` are written, and each
     reloads into a FRESH ``build_scorer(backbone).backbone_module.load_state_dict``;
  3. gradients flowed: at least one backbone parameter CHANGED vs its initial
     value after the 2 epochs (the proof training works — especially for ESM3
     after removing its internal ``no_grad``).

This is the real gate for Task 7.  Loading ESM3 + a couple of optimizer steps on
GPU takes a few minutes, so this is a standalone script (run on the A4500) whose
output is captured into ``results/finetune_smoke.md`` rather than a pytest test.

Run on the A4500 (CUDA index 1 under PCI_BUS_ID ordering):
    cd esm-backbone-test
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
        python -m run.smoke_finetune --backbones esmc_600m esm3
"""

import argparse
import copy
import os
import sys
from pathlib import Path

_RUN_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _RUN_DIR.parent
_WORKTREE_ROOT = _PKG_ROOT.parent
for _p in (str(_WORKTREE_ROOT), str(_PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("HF_HOME", "/home/guoj0f/repos/esm/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from common.build_scorer import build_scorer
from common.stab_model import StaBddG
from run.finetune import (
    SKEMPI_PDB_DIR,
    build_train_dataset,
    finetune,
)


class _Args:
    """Minimal args object to drive run.finetune.finetune()."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def smoke_one(backbone, device, epochs=2, limit=1, out_root="cache/ft_smoke"):
    """Run a tiny finetune for ``backbone`` and return a result dict."""
    print(f"\n{'=' * 70}\n[smoke] backbone={backbone} device={device} "
          f"epochs={epochs} limit={limit}\n{'=' * 70}", flush=True)

    run_name = f"{backbone}_smoke"
    out = os.path.join(out_root, backbone)
    # backbone-default token budget (4000 esm3, 10000 otherwise)
    from run.finetune import DEFAULT_BATCH_SIZE
    batch_size = DEFAULT_BATCH_SIZE[backbone]

    dataset = build_train_dataset(
        f"cache/skempi_train_pdb_dict.pkl", limit=limit
    )

    scorer = build_scorer(backbone, device=str(device), checkpoint=None,
                          pdb_dir=SKEMPI_PDB_DIR)
    model = StaBddG(scorer).to(device)

    # snapshot initial backbone weights (deep copy on CPU) to compare later
    init_state = copy.deepcopy(
        {k: v.detach().cpu().clone() for k, v in
         model.scorer.backbone_module.state_dict().items()}
    )

    args = _Args(backbone=backbone, stage="skempi", lr=1e-5, epochs=epochs,
                 batch_size=batch_size, out=out, run_name=run_name,
                 device=str(device), limit=limit)

    finetune(model, dataset, args, device)

    # --- parse per-epoch mean losses from the CSV log ---
    log_path = os.path.join(out, f"{run_name}_train_log.csv")
    losses = []
    with open(log_path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                losses.append(float(parts[2]))
    import math
    losses_finite = all(math.isfinite(x) for x in losses)

    # --- checkpoints written? ---
    ckpts = [os.path.join(out, f"{run_name}_{e}.pt") for e in range(1, epochs + 1)]
    ckpts_exist = all(os.path.exists(p) for p in ckpts)

    # --- each checkpoint reloads into a FRESH scorer of the same backbone? ---
    fresh = build_scorer(backbone, device="cpu", checkpoint=None,
                         pdb_dir=SKEMPI_PDB_DIR)
    reload_ok = True
    reload_err = None
    try:
        for p in ckpts:
            sd = torch.load(p, map_location="cpu")
            fresh.backbone_module.load_state_dict(sd)
    except Exception as e:  # pragma: no cover - reported in md
        reload_ok = False
        reload_err = repr(e)

    # --- did weights change vs initial? (proof gradients flowed) ---
    final_state = {k: v.detach().cpu().clone()
                   for k, v in model.scorer.backbone_module.state_dict().items()}
    changed_params = []
    max_abs_delta = 0.0
    for k, v0 in init_state.items():
        v1 = final_state[k]
        if v0.shape == v1.shape and v0.dtype.is_floating_point:
            delta = (v1.float() - v0.float()).abs().max().item()
            max_abs_delta = max(max_abs_delta, delta)
            if delta > 0:
                changed_params.append(k)
    weights_changed = len(changed_params) > 0

    res = {
        "backbone": backbone,
        "batch_size": batch_size,
        "losses": losses,
        "losses_finite": losses_finite,
        "ckpts_exist": ckpts_exist,
        "ckpt_paths": ckpts,
        "reload_ok": reload_ok,
        "reload_err": reload_err,
        "weights_changed": weights_changed,
        "n_changed_params": len(changed_params),
        "n_total_float_params": sum(
            1 for v in init_state.values() if v.dtype.is_floating_point),
        "max_abs_delta": max_abs_delta,
    }
    print(f"[smoke:{backbone}] losses={losses} finite={losses_finite} "
          f"ckpts={ckpts_exist} reload_ok={reload_ok} "
          f"weights_changed={weights_changed} "
          f"(changed {len(changed_params)}/{res['n_total_float_params']} tensors, "
          f"max|delta|={max_abs_delta:.3e})", flush=True)

    # hard asserts (the gate)
    assert losses_finite, f"{backbone}: non-finite loss {losses}"
    assert ckpts_exist, f"{backbone}: missing checkpoints {ckpts}"
    assert reload_ok, f"{backbone}: checkpoint reload failed: {reload_err}"
    assert weights_changed, f"{backbone}: NO weights changed (grad blocked?)"

    # free GPU before the next backbone
    del model, scorer, fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["esmc_600m", "esm3"],
                    choices=["mpnn", "esmc_600m", "esmc_6b", "esm3"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "cuda" else torch.device("cpu"))
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device} ({gpu_name})", flush=True)

    results = []
    for bb in args.backbones:
        results.append(smoke_one(bb, device, epochs=args.epochs, limit=args.limit))

    print("\n" + "=" * 70)
    print("SMOKE SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['backbone']:10s}  init_loss={r['losses'][0]:.4f}  "
              f"final_loss={r['losses'][-1]:.4f}  "
              f"weights_changed={r['weights_changed']}  "
              f"reload_ok={r['reload_ok']}  "
              f"max|delta|={r['max_abs_delta']:.3e}")
    print("ALL SMOKE ASSERTIONS PASSED", flush=True)


if __name__ == "__main__":
    main()
