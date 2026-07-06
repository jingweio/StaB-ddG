#!/usr/bin/env python3
"""LoRA fine-tune ESM3ΔG on binding ΔΔG (SKEMPI) or folding ΔΔG (Megascale).

Faithful ESM3ΔG: only LoRA + stability head + output scaling train; ESM3 trunk frozen.
  --stage skempi    : binding ddG = complex - binder1 - binder2  (SKEMPIDataset)
  --stage megascale : folding ddG = dG(mut) - dG(wt)             (Tsuboyama per-domain)
Stage chaining: pass a prior run's --resume to continue (megascale stage1 -> skempi stage2).
"""
import os, sys, argparse, time
os.environ.setdefault("HF_HOME", "/home/guoj0f/share/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import torch
from scipy.stats import spearmanr
from scorer import ESM3dGScorer


def save_adapters(scorer, path):
    sd = scorer.adapter_state_dict()          # name-based: only LoRA + head + scaling
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(sd, path)

def load_adapters(scorer, path):
    sd = torch.load(path, map_location=scorer.device)
    msg = scorer.model.load_state_dict(sd, strict=False)
    loaded = len(sd)
    print(f"  resumed {loaded} adapter tensors from {path}")


def _batch_M(L, batch_tokens, max_batch):
    """#sequences per batch: token budget capped by a hard sequence cap (ESM3-1.4B
    backward is memory-heavy per sequence — the ProteinMPNN token heuristic over-batches)."""
    return max(1, min(batch_tokens // max(1, L), max_batch))

def chunked_binding_ddG(scorer, item, batch_tokens, max_batch=8):
    L = item["complex"]["seq"].shape[0]
    M = _batch_M(L, batch_tokens, max_batch)
    n = item["ddG"].shape[0]
    preds = []
    for s in range(0, n, M):
        e = min(n, s + M)
        p = scorer.binding_ddG(item["complex"], item["binder1"], item["binder2"],
                               item["complex_mut"][s:e], item["binder1_mut"][s:e], item["binder2_mut"][s:e])
        preds.append(p)
    return torch.cat(preds)

def chunked_folding_ddG(scorer, item, batch_tokens, max_batch=8):
    L = item["enc"]["seq"].shape[0]
    M = _batch_M(L, batch_tokens, max_batch)
    n = item["ddG"].shape[0]
    preds = []
    for s in range(0, n, M):
        e = min(n, s + M)
        preds.append(scorer.folding_ddG(item["enc"], item["mut"][s:e]))
    return torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["skempi", "megascale"], required=True)
    ap.add_argument("--lora_ckpt", required=True, help="ESM3ΔG init ckpt (LoRA+head+scaling)")
    ap.add_argument("--resume", default="", help="prior fine-tuned adapters (stage chaining)")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--lr", type=float, default=6e-5)        # esm-replace best for ESM3
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_tokens", type=int, default=10000,
                    help="StaB token budget: optimizer-step WINDOW = batch_tokens//L mutants (grad-accumulated)")
    ap.add_argument("--max_batch", type=int, default=4,
                    help="physical micro-batch cap /forward (ESM3 memory); grads accumulated up to the window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single_batch", action="store_true",
                    help="sample one batch of mutants per domain/epoch (needed for Megascale: ~1460 muts/domain)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_freq", type=int, default=0, help="also save adapters every K epochs (0=only final)")
    ap.add_argument("--run_name", default="run")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.run_name}] stage={args.stage} lr={args.lr} {args.optimizer} wd={args.weight_decay} "
          f"epochs={args.epochs} batch_tokens={args.batch_tokens} dev={dev}")

    scorer = ESM3dGScorer(args.lora_ckpt, device=dev, trainable=True)
    if args.resume:
        load_adapters(scorer, args.resume)
    n_train = sum(p.numel() for p in scorer.trainable_parameters())
    print(f"  trainable params: {n_train:,}")

    # ---- data ----
    D = args.data_dir
    if args.stage == "skempi":
        from skempi_data import build_skempi
        items = build_skempi(scorer,
            csv_path=os.path.join(D, "SKEMPI", "filtered_skempi.csv"),
            split_path=os.path.join(D, "SKEMPI", f"{args.split}_pdb.pkl"),
            pdb_dir=os.path.join(D, "SKEMPI2_PDBs"),
            pdb_dict_cache_path=os.path.join(HERE, "..", "cache", f"skempi_esm3_{args.split}_pdb_dict.pkl"),
            limit=args.limit)
        predict = chunked_binding_ddG
    else:
        from megascale_data import build_megascale
        items = build_megascale(scorer, data_dir=D, split=args.split, limit=args.limit)
        predict = chunked_folding_ddG

    # re-freeze AFTER encoding (the VQ-VAE structure encoder loads lazily during the
    # first encode and would otherwise stay trainable → corrupts/bloats). Only LoRA+head+scaling train.
    n_after = scorer.set_trainable(True)
    print(f"  trainable params after data build (re-frozen encoder): {n_after:,}")

    opt_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = opt_cls(scorer.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    for ep in range(args.epochs):
        t_ep = time.time()
        scorer.model.train()
        order = np.random.permutation(len(items))
        losses, sps, oom = [], [], 0
        for idx in order:
            it = items[idx]
            label = it["ddG"].to(dev)
            L = (it["complex"]["seq"].shape[0] if args.stage == "skempi" else it["enc"]["seq"].shape[0])
            n = label.shape[0]
            perm = torch.randperm(n)                       # shuffle mutants within complex each epoch
            label = label[perm.to(dev)]
            # StaB alignment: one optimizer step per WINDOW of Mwin = batch_tokens//L mutants
            # (StaB does exactly this; its tiny model fits Mwin in one forward). ESM3-1.4B can't,
            # so we GRAD-ACCUMULATE physical micro-batches (<= max_batch, OOM-fallback ->2->1) up
            # to the window, then step ONCE. => same #mutants averaged per gradient & same #steps
            # per complex as StaB. Loss scaled by (chunk/window) so accumulated grad == mean over window.
            Mwin = max(1, args.batch_tokens // max(1, L))
            n_use = min(n, Mwin) if args.single_batch else n   # single_batch: one window/complex/epoch (Megascale)
            ps, failed = [], False
            for ws in range(0, n_use, Mwin):
                we = min(n_use, ws + Mwin); win = we - ws
                ps_mark = len(ps); win_loss, ok = 0.0, False
                for mbtry in sorted({args.max_batch, 2, 1}, reverse=True):
                    optimizer.zero_grad(set_to_none=True)
                    del ps[ps_mark:]                        # drop partial preds from a failed OOM attempt
                    win_loss = 0.0
                    try:
                        for s in range(ws, we, mbtry):
                            e = min(we, s + mbtry)
                            pi = perm[s:e]
                            if args.stage == "skempi":
                                pred = scorer.binding_ddG(it["complex"], it["binder1"], it["binder2"],
                                                          it["complex_mut"][pi], it["binder1_mut"][pi], it["binder2_mut"][pi])
                            else:
                                pred = scorer.folding_ddG(it["enc"], it["mut"][pi])
                            loss = loss_fn(pred, label[s:e]) * ((e - s) / win)   # accumulate -> mean over window
                            loss.backward()
                            win_loss += loss.item()
                            ps.append(pred.detach().cpu())
                        ok = True; break
                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
                if not ok:                                  # even micro-batch=1 OOMs -> drop this complex
                    failed = True; break
                optimizer.step()
                losses.append(win_loss)
            if failed:
                oom += 1; optimizer.zero_grad(set_to_none=True); del ps[:]; continue
            if ps:
                ps = torch.cat(ps)
                if ps.shape[0] >= 3:
                    sp, _ = spearmanr(ps.numpy(), label[:ps.shape[0]].cpu().numpy())
                    if np.isfinite(sp): sps.append(sp)
        print(f"  epoch {ep+1}/{args.epochs}  loss={np.mean(losses):.4f}  "
              f"train_spearman={np.mean(sps):.3f}  oom_skipped_complexes={oom}  "
              f"time={time.time()-t_ep:.0f}s", flush=True)
        if args.save_freq and (ep + 1) % args.save_freq == 0 and (ep + 1) < args.epochs:
            save_adapters(scorer, args.out.replace(".pt", f"_ep{ep+1}.pt"))

    save_adapters(scorer, args.out)
    print(f"saved adapters -> {args.out}")

if __name__ == "__main__":
    main()
