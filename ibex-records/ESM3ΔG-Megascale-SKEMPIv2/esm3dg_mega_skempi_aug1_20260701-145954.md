# ESM3ΔG-Megascale-SKEMPIv2 — experiment record  (created 2026-07-01; status: PLANNED)

> **REDO**:第一次 run(2026-06-30)跑在**已污染并删除的** `/home/guoj0f/repos/esm` 上,结果作废(旧 test per-structure 0.008;stage-2 曾几乎没学动)。本次在干净的 `share/esm` + `share/hf_cache` 上重跑。

## 1. Goal / hypothesis
在 ESM3ΔG(MGnify 预训练)之上**再加 Megascale folding-ddG stage-1**,再微调 SKEMPI binding ΔΔG,是否比 task1(无 Megascale stage)更好。**task2 vs task1** 隔离"加 Megascale stage 的增量"。

## 2. Design & decision points  (写于提交前)
- **架构**:同 task1(ESM3ΔG 冻结 + LoRA r=4 + head;纯 LoRA+head;binding 分解)。
- **两阶段**:stage-1 = Megascale folding ddG(单链域,`--single_batch`:每域每 epoch 采 1 batch,因每域 ~1460 突变);stage-2 = `--resume` stage-1 → SKEMPI binding ddG。
- **KEY DECISIONS**:
  - 模型版本 = `ESM3dG_weights_augmented_1`(单成员;⚠ 用户保留 ensemble/base 决策权)。
  - 超参:两阶段 lr 6e-5 / AdamW / wd 0.05 / 15 ep / seed 0;stage-1 `--max_batch 8`(小域),stage-2 `--max_batch 4` + `--save_freq 5`。
  - 数据:stage-1 Megascale(`data/megascale/Tsuboyama…csv` + `AlphaFold_model_PDBs` + `mega_splits.pkl` train,~239 域);stage-2 同 task1 SKEMPI。
- **对标**:ProteinMPNN 0.448;task1 结果。
- **指标**:test per-structure Spearman + overall。

## 3. Run config
- SLURM:a100 ×1(A100-SXM4-80GB),walltime 23h,cpus 12,mem 110G。
- sbatch:`ibex-records/ESM3ΔG-Megascale-SKEMPIv2/sh/esm3dg_mega_skempi_aug1_20260701-145954.sh`。
- env / HF_HOME / 代码路径:同 task1(`esm-backbone` py3.12 + share/)。

## 4. Change log  (LIVE)
- 2026-07-01:依赖迁 share/esm;旧记录已清;plan 写于重跑前。job 未提交(待 env 重建 + smoke)。

## 5. Results  (jobs 完成后填)
- _待写入:stage-1/stage-2 曲线 / test per-structure / overall / 与 task1、0.448 对比 / 与旧(作废)0.008 对照。_
