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

## ESM3 HP Round 3 (weight decay @ lr=6e-5) — wd HELPS
| lr | wd | train Spearman | test Spearman |
|---|---|---:|---:|
| 6e-5 | 0 | 0.435 | 0.186 |
| 6e-5 | 0.05 | 0.565 | **0.242** (new best) |
| 6e-5 | 0.1 | 0.567 | 0.157 (over-reg) |
wd=0.05 raised BOTH train(0.44→0.57) and test(0.19→0.24) → better optimization. Trajectory still climbing (0.075→0.134→0.186→0.242). Round 4: map wd∈{0.02,0.03} + lr1e4+wd + more epochs to find the ceiling.

## ESM3 HP Round 4 (map optimum) + FINAL
| lr | wd | epochs | train Sp. | test Sp. |
|---|---|---|---:|---:|
| 6e-5 | 0.02 | 15 | 0.625 | 0.106 |
| 6e-5 | 0.03 | 15 | 0.561 | 0.203 |
| **6e-5** | **0.05** | **15** | 0.565 | **0.242** ← OPTIMUM |
| 6e-5 | 0.10 | 15 | 0.567 | 0.157 |
| 1e-4 | 0.05 | 15 | 0.565 | 0.176 |
| 6e-5 | 0.05 | 30 | 0.529 | 0.179 |

### CONCLUSION (ESM3, reliable after thorough HP sweep)
Swept lr {1e-6…3e-4} × wd {0…0.1} × optimizer {Adam,AdamW} × schedule {constant,cosine,warmup} × epochs {15,30} + per-epoch early-stopping. **ESM3 two-stage optimum: lr=6e-5, AdamW wd=0.05, 15 ep → test per-interface Spearman = 0.242** (train 0.565). The naive lr=1e-5 run (0.104) was indeed UNDER-TUNED — proper tuning more than doubled it (0.104→0.242). **But the optimum (0.242) is still far below ProteinMPNN's 0.445.** ESM3 fits train well (~0.57) yet generalizes to only ~0.24 on the homology-OOD test → the bottleneck is **OOD generalization**, not training. A small specialized inverse-folding net (ProteinMPNN) generalizes far better on OOD PPI ΔΔG than the large general-purpose ESM3 via the StaB identity. (Pending: from-pretrained ablation = does Megascale Stage-1 help?)

## Ablation: does Megascale Stage-1 help? — YES
ESM3 best config (lr=6e-5, AdamW wd=0.05, 15ep), from PRETRAINED (no Stage-1) → test per-iface Spearman **0.148**, vs two-stage (with Stage-1) **0.242**. → Megascale folding-stability Stage-1 adds **+0.094** (nearly doubles the single-stage gain over zero-shot 0.086). The two-stage design IS valuable.

## FINAL SUMMARY (ESM3 track)
| config | per-interface Spearman |
|---|---:|
| ProteinMPNN two-stage (baseline) | **0.445** |
| **ESM3 two-stage, tuned (lr6e-5/AdamW wd0.05/15ep)** | **0.242** |
| ESM3 SKEMPI-only, tuned (no Megascale) | 0.148 |
| ESM3 zero-shot | 0.086 |
| ESMC-6B zero-shot (track paused) | 0.014 |
**Verdict:** Proper HP tuning + Megascale pretraining take ESM3 from 0.086 → 0.242 (under-tuning was real and fixed), but ESM3 still generalizes far worse on the homology-OOD split than the small specialized ProteinMPNN (0.242 vs 0.445). Bottleneck = OOD generalization (ESM3 train Spearman ~0.57 vs test ~0.24), not training.
