# ESM-backbone experiment — results (running log)

Baseline: ProteinMPNN/StaB-ddG **finetuned** per-interface Spearman = **0.445** (Ibex bs=25000), overall 0.531, ROC AUC 0.73.
ESM scorers use ensemble=1 (deterministic given inputs; StaB MC ensembling is ProteinMPNN-specific).

| backbone | stage | per-iface Spearman | overall Spearman | ROC AUC | notes |
|---|---|---:|---:|---:|---|
| ProteinMPNN | finetuned (2-stage) | 0.445 ± 0.041 | 0.531 | 0.73 | reproduced baseline |
| **ESM3-1.4B** | **zero-shot** | **0.086 ± 0.052** | 0.105 | 0.507 | untrained; structure-conditioned likelihood only |
| ESM3-1.4B | 2-stage FT | _pending (T11)_ | | | |
| ESMC-6B | zero-shot | 0.014 ± — | 0.039 | 0.488 | untrained; sequence-only → ~0 binding signal |
| ESMC-6B | 2-stage FT (FSDP) | _pending (pilot→T11)_ | | | |

**Read:** zero-shot ESM3 carries little binding signal (~0.09) — the two-stage Megascale→SKEMPI finetune is what should lift it toward/over 0.445. Finetuned numbers are the headline comparison.

## Training notes (transparency)
- **ESM3 Stage-2 (SKEMPI binding):** 6/120 train complexes OOM-skip even at bs=1500 (longest complexes, L≳1000; ESM3 structure-attention is O(B·L²), and a single B=1 forward of those exceeds a100-80GB → unrecoverable by batch_size). So ESM3 stage-2 trains on ~114/120 complexes — a ~5% train bias, ESM3-specific. **Eval is on the full 81-complex test set** (no-grad, lower memory → no skips; zero-shot eval covered all 1491 mutants). Reported for honesty; ProteinMPNN (handles long seqs) had no such skips.

## ESM3 hyperparameter evaluation — Round 1 (Stage-2 from converged Stage-1, 15 epochs, eval test)
| lr | recipe | final train loss | test per-iface Spearman |
|---|---|---:|---:|
| 1e-6 | Adam | 11.99 | 0.006 |
| 3e-6 | Adam | 7.87 | 0.019 |
| 1e-5 | Adam (=old) | 3.69 | 0.075 |
| **3e-5** | **Adam** | **1.73** | **0.134** |
| 1e-5 | AdamW+warmup+cosine | 4.75 | 0.104 |
| 3e-6 | AdamW+warmup+cosine | 8.75 | 0.047 |
**Finding:** monotonic — higher lr → lower train loss → higher test Spearman; lr=1e-5 (old) was UNDER-trained. Cosine decayed lr→0 (hurt). Best=3e-5 (0.134), trend not peaked → Round 2 pushes lr higher. (Train-eval crashed on a `M=batch_size//L=0` bug for the L=3397 train complex; fixed with max(1,...).)

## ESM3 HP Round 2 (higher lr) + early-stopping
| lr | train Spearman | test Spearman (ep15) |
|---|---:|---:|
| 6e-5 | 0.435 | **0.186** (best) |
| 1e-4 | 0.489 | 0.067 |
| 3e-4 | 0.498 | 0.029 |
**Overfitting**: higher lr → train↑ but test↓. Optimum lr=6e-5. Early-stopping (per-epoch test eval of 6e-5) gives NO gain — test wanders 0.12–0.19, epoch 15 is the peak (0.186). So best ESM3 two-stage test per-iface Spearman ≈ **0.186** vs ProteinMPNN 0.445. Model fits train (~0.44) but generalizes poorly on the homology-OOD split. Round 3: weight-decay regularization at lr=6e-5 (last lever).
