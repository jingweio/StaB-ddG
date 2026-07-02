# ESM3ΔG-Megascale-SKEMPIv2 — experiment record  (created 2026-07-01; status: RUNNING)

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
- SLURM:a100 ×1(A100-SXM4-80GB),walltime **10h**(据上一次相同 run 实测 7h10m + ~40% buffer;从 23h 下调以改善 backfill/缩短排队),cpus 12,mem 110G。
- sbatch:`ibex-records/ESM3ΔG-Megascale-SKEMPIv2/sh/esm3dg_mega_skempi_aug1_20260701-145954.sh`。
- env / HF_HOME / 代码路径:同 task1(`esm3dg` py3.12 + share/)。

## 4. Change log  (LIVE)
- 2026-07-01:依赖迁 share/esm;旧记录已清;plan 写于重跑前。job 未提交(待 env 重建 + smoke)。

- 2026-07-01(cont.):env 明确为**独立 `esm3dg`**(py3.12,torch2.6+cu124,esm 3.3.0 editable from share/esm;从本地打的 per-branch wheelhouse 离线装 + `--no-build-isolation`)。**不用 esm-backbone**(那是别 task 的 env)。本地 + Ibex a100 smoke 均通过(ESM3ΔG 加载/预测 ΔG 一致、scorer OK)。**首次提交误用 esm-backbone 已取消**,改用 esm3dg 重新提交。

- 2026-07-01(cont.):**task2 (ESM3ΔG→Megascale→SKEMPI) 提交,job 47936772**(a100-80GB,esm3dg env)。

- 2026-07-01(cont.):job PENDING(等 a100)。据 sacct 上一次相同 run 实测 7h10m,把 walltime 23h→**10h**(`scontrol update` 改 live job + 同步 sbatch 源文件),`--start` 估计从 07-04 提前到 07-03。

## 5. Results  (COMPLETED 2026-07-02;job 47936772,a100,elapsed 7h10m)

**SKEMPI test(81 complexes / 1491 mutants;OOM-skip 6;两阶段:stage-1 Megascale folding → stage-2 SKEMPI binding `--resume`):**

| 指标 | **task2** (Mega→SKEMPI, clean) | **task1** (SKEMPI-only, clean) | 旧 task2(作废)| ProteinMPNN |
|---|---|---|---|---|
| **per-structure Spearman** | **0.1089** | 0.1584 | 0.008 | **0.448** |
| overall Spearman | 0.2542 | 0.2763 | — | 0.531 |
| overall Pearson | 0.2132 | 0.2615 | — | — |

- stage-1 Megascale 训完存 `stage1.pt`;stage-2 `--resume` 续上,train ep1 spr0.107 → ep15 **0.536**(中途 ep9 单点抖动 0.252 已回弹,纯噪声)。
- csv 留证:`results/eval_esm3dg_mega_skempi_aug1.csv`;adapter:`cache/esm3dg_mega_skempi_aug1.pt`。

**判读(task2 vs task1 —— 本实验的核心对照):**
1. **加 Megascale folding stage-1 = 净负**:task2 在 per-structure(0.109 < 0.158)和 overall(0.254 < 0.276)上**都低于** task1。原假设"ESM3ΔG 之上再叠 Megascale folding stage 有助 binding ΔΔG"**不成立**。
2. **可能原因**:(a) ESM3ΔG 本就已在 MGnify stability 上预训练,再加 Megascale(又一个 monomer folding-stability 数据集)是**冗余的 folding 信号**,不带来 binding 相关信息;(b) 中间 folding stage 把 LoRA/head 推向 monomer-folding basin,stage-2 需要"再适配"回 binding,反而略损。这与 ProteinMPNN(StaB-ddG)中 Megascale→SKEMPI 两阶段**有效**的结论相反 —— 说明该 recipe **不随 backbone 迁移**。
3. 与 task1 同样的两个 gap:train 0.536 ≫ test per-structure 0.109(泛化 gap);overall > per-structure(复合物内排序弱)。
4. clean 依赖把 task2 从 0.008 → 0.109(旧值近乎零) —— 污染依赖对 task2 影响尤其大。

**结论**:两个任务都清干净依赖后,**per-structure 仍远低于 ProteinMPNN 0.448**,且 Megascale stage 有害。低分是真实方法性 gap,不是依赖问题。诊断/改进候选见 task1 §5(早停 / SUM 聚合 / 分解数值噪声)——属改设计,待用户拍板。
