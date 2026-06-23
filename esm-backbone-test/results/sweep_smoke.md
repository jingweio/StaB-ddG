# LR Sweep Driver Smoke Test — esmc_600m

**Date:** 2026-06-24  
**Backbone:** esmc_600m  
**Command:**
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python -m run.lr_sweep \
    --backbone esmc_600m \
    --lrs "1e-6,5e-6" \
    --epochs 1 \
    --batch_size 2000 \
    --device cuda \
    --limit_train 2 \
    --limit_eval 3 \
    --out_dir results/sweep_esmc_600m
```
**GPU:** NVIDIA RTX A4500 (20 GB, CUDA index 1 under PCI_BUS_ID ordering)

## results/sweep_esmc_600m/sweep.csv

```
lr,epochs,per_interface_spearman,overall_spearman,n_qualifying_complexes
1e-06,1,-0.0062584354682832894,0.003128144569457893,3
5e-06,1,0.11824964830527233,0.09341330309762526,3
```

## results/sweep_esmc_600m/best.json

```json
{
  "best_lr": 5e-06,
  "backbone": "esmc_600m",
  "epochs": 1,
  "metric_used": "per_interface_spearman",
  "best_per_interface_spearman": 0.11824964830527233,
  "best_overall_spearman": 0.09341330309762526,
  "all_rows": [
    {
      "lr": 1e-06,
      "epochs": 1,
      "per_interface_spearman": -0.0062584354682832894,
      "overall_spearman": 0.003128144569457893,
      "n_qualifying_complexes": 3
    },
    {
      "lr": 5e-06,
      "epochs": 1,
      "per_interface_spearman": 0.11824964830527233,
      "overall_spearman": 0.09341330309762526,
      "n_qualifying_complexes": 3
    }
  ]
}
```

## Driver gate verification

- [x] `sweep.csv` has 2 rows (one per LR)
- [x] `per_interface_spearman` and `overall_spearman` columns contain finite floats (NaN was NOT needed; all 3 eval complexes had >= 10 mutations and qualified at THRESHOLD=10)
- [x] `n_qualifying_complexes=3` for both rows (all 3 eval complexes passed the K>=10 gate)
- [x] `best.json` written; best_lr=5e-06 selected by argmax per_interface_spearman
- [x] Full pipeline ran end-to-end: finetune → checkpoint → eval → compute_metrics → record → argmax

## Concerns / notes

- **NaN not triggered on this smoke:** With `--limit_eval 3`, all 3 complexes happen to have >= 10 mutations in the test split (1A4Y: 192 rows, 1AO7 and 1B41 combined cover the rest), so `n_qualifying_complexes=3` and `per_interface_spearman` is finite. If eval is restricted to 1 complex with < 10 mutations, `num_ppis=0` would cause `ZeroDivisionError` in `compute_metrics`; the driver catches this and logs a WARNING with `per_interface_spearman=NaN`, falling back to `overall_spearman` for argmax.
- **Spearman values are low/noisy** due to 1-epoch training on 2 complexes only — this is expected for a driver mechanics smoke, not a quality signal.
- **FutureWarning on torch.load:** `weights_only=False` used for checkpoint loading (same as the rest of the codebase); harmless for now.
