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
