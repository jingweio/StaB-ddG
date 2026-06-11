"""Probe the largest --batch_size (in tokens) that lets a real training step
(forward + backward) fit in GPU memory for EVERY SKEMPI train complex.

Replicates the per-chunk train step of skempi_finetune.py. Reuses the cached
structure dict so loading is fast. Run on the TITAN X (cuda:0 under
CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=0).
"""
import os
import torch
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.model import StaBddG
from stabddg.ppi_dataset import SKEMPIDataset

CKPT = "model_ckpts/stability_finetuned.pt"
PDB_DIR = "data/SKEMPI2_PDBs"
CSV = "data/SKEMPI/filtered_skempi.csv"
SPLIT = "data/SKEMPI/train_pdb.pkl"
CACHE = "output/2026-06-01/skempi_train_pdb_dict.pkl"
CANDIDATES = [8000, 6000, 5000, 4000, 3000, 2500, 2000, 1500]
LOSS = torch.nn.MSELoss()


def main():
    device = torch.device("cuda:0")
    print("device:", torch.cuda.get_device_name(0))

    ds = SKEMPIDataset(split_path=SPLIT, pdb_dir=PDB_DIR, csv_path=CSV,
                       pdb_dict_cache_path=CACHE)

    pmpnn = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3,
                        k_neighbors=48, dropout=0.0, augment_eps=0.0)
    ckpt = torch.load(CKPT)
    pmpnn.load_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
    model = StaBddG(pmpnn=pmpnn, noise_level=0.2, use_antithetic_variates=True, device=device)
    model.to(device)
    model.train()

    # order complexes largest-first (largest complex seq length = worst case)
    samples = sorted(ds, key=lambda s: s["complex_mut_seqs"].shape[1], reverse=True)
    print(f"{len(samples)} complexes; largest complex length = "
          f"{samples[0]['complex_mut_seqs'].shape[1]} residues "
          f"({samples[0]['name']})")

    def one_step(sample, bs):
        L = sample["complex_mut_seqs"].shape[1]
        N = sample["complex_mut_seqs"].shape[0]
        M = max(1, bs // L)
        B = min(N, M)
        c = sample["complex_mut_seqs"][:B].to(device)
        b1 = sample["binder1_mut_seqs"][:B].to(device)
        b2 = sample["binder2_mut_seqs"][:B].to(device)
        ddG = sample["ddG"][:B].float().to(device)
        pred = model(sample["complex"], sample["binder1"], sample["binder2"], c, b1, b2)
        loss = LOSS(pred, ddG)
        loss.backward()
        model.zero_grad(set_to_none=True)

    chosen = None
    for bs in CANDIDATES:
        failed = []
        peak = 0
        for s in samples:
            try:
                one_step(s, bs)
                peak = max(peak, torch.cuda.max_memory_allocated() // (1024**2))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    failed.append((s["name"], s["complex_mut_seqs"].shape[1]))
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                else:
                    raise
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        status = "OK (0 OOM)" if not failed else f"{len(failed)} OOM: {failed[:3]}"
        print(f"  batch_size={bs:5d} -> {status} | peak~{peak} MiB")
        if not failed:
            chosen = bs
            break

    print("\n==> RECOMMENDED batch_size:", chosen if chosen else "none of the candidates fit (try <1500)")


if __name__ == "__main__":
    main()
