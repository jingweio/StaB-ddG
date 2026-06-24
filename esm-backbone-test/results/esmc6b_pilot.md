# ESMC-6B FSDP pilot — result

- Job 47757476 | State **COMPLETED** | 4×a100 | elapsed 5:25 | peak host RAM ~127 GB (of 240 G) | consolidated ckpt 12 GB (808 tensors).
- Validated end-to-end: load_esmc 6B (correct weights) → activation-checkpoint + FSDP/ZeRO-3 wrap of UnifiedTransformerBlock → `_LogitsAdapter` shim (fixes ESMCOutput-dataclass FSDP backward) → 6-graph StaB binding loss (10 steps, finite) → `FSDP.state_dict_type(FULL_STATE_DICT)` consolidated save (loadable by a plain ESMCScorer for eval). NO OOM.
- Throughput: 5 train complexes + 1 epoch + setup = 5.4 min (≈3 min train). Extrapolated full SKEMPI stage-2 (120 complexes) ≈ ~70 min/epoch on 4×a100 (faster on 8).
- torchrun must use `--standalone` (auto free port); default 29500 collided on shared 8-GPU nodes (EADDRINUSE).
