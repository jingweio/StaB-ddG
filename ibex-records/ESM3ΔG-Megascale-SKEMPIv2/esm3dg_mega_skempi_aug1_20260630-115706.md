# ESM3ΔG-Megascale-SKEMPIv2 — experiment record  (created 2026-06-30; status: RUNNING)

## 1. Goal / hypothesis
测试:在 ESM3ΔG(MGnify 预训练)之上**再加 Megascale folding-ddG 预训练阶段**,然后微调 SKEMPI binding ΔΔG,是否比 task1(无 Megascale stage)更好。
本任务 = "MGnify + Megascale + SKEMPI"(两阶段)。与 task1(`ESM3ΔG-SKEMPIv2`)配对;**task2 vs task1** 隔离"加 Megascale stage 的增量",**task2 vs plain-ESM3 两阶段 0.242** 隔离"MGnify+LoRA 相对 full-FT 的效果"。

## 2. Design & decision points  (写于提交前)
- **架构**:同 task1(ESM3ΔG 冻结 trunk + LoRA r=4 + head;纯 LoRA+head 训练;binding 分解 `complex − binder1 − binder2`)。
- **两阶段流程**:
  - **stage-1**:ESM3ΔG → 在 **Megascale (Tsuboyama) folding ddG** 上微调(单链 domain,`folding_ddG=dG(mut)−dG(wt)`,MSE)。
  - **stage-2**:`--resume` stage-1 权重 → 在 **SKEMPI binding ddG** 上微调。
- **KEY DECISIONS(选项 + 理由)**:
  - **模型版本 = `ESM3dG_weights_augmented_1`**(单成员;同 task1,⚠ 用户保留 ensemble/base 的最终决策权;本次先 augmented_1)。
  - **stage-1 用 `--single_batch`**:Megascale 每域 ~1460 个突变,全跑则一个 epoch ~87k 步、ESM3 上不可行;`--single_batch` = 每域每 epoch 只采 1 个 batch 的突变(对齐原 `stability_finetune.py`)。
  - **超参**:两阶段均 lr 6e-5 / AdamW / wd 0.05 / 15 epochs / seed 0;stage-1 `--max_batch 8`(小域),stage-2 `--max_batch 4`(大复合物,OOM 降批重试)。
  - **数据**:stage-1 Megascale = `data/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` + AF2 结构 `AlphaFold_model_PDBs` + `mega_splits.pkl` train split(实测编码 **239 domains**);stage-2 同 task1 SKEMPI。
- **对标**:同 task1(0.448 / 0.148),外加 **plain-ESM3 两阶段 0.242** 与 **task1 结果**。
- **评测**:SKEMPI test per-structure Spearman + overall。

## 3. Run config
- SLURM:a100 ×1(A100-SXM4-80GB),walltime 23h,`--cpus-per-task 12 --mem 110G`。
- sbatch:`ibex-records/ESM3ΔG-Megascale-SKEMPIv2/sh/esm3dg_mega_skempi_aug1_20260630-115706.sh`。
- env / 权重 / 代码路径:同 task1。
- job id:**47891585**(节点 gpu101-16-l)。

## 4. Change log  (LIVE)
- 2026-06-30 ~11:5x:提交 job 47891585(已含 OOM 降批重试修复)。
- 2026-06-30:**stage-1 (Megascale) 训完 15 ep**,末 train_spearman **0.81**(loss 0.26,0 OOM-skip),已存 `..._stage1.pt`;进入 **stage-2 SKEMPI**(编码 120 complexes,resume stage-1,训练中)。RUNNING,GPU 100% util。

- 2026-06-30 ~18:5x:**job 47891585 COMPLETED**(~7h)。stage-1 收敛 0.81,但 **stage-2 几乎没学动**(loss 卡 ~4),**test per-structure 仅 0.008**(overall 0.225)→ 比 task1 更差,Megascale stage 有害。详见 §5。

## 5. Results  (job 47891585 COMPLETED, elapsed ~07:0x)
- **stage-1 (Megascale folding ddG)**:train_spearman →**0.81**(15 ep,学得很好;loss→0.26)。
- **stage-2 (SKEMPI binding ddG, resume stage-1)**:**几乎没学动** —— train loss 卡在 ~4.0–4.4,train_spearman 在 0.04–0.17 间抖(对比 task1 同期 train_sp 0.57)。
- **Test(SKEMPI test_pdb.pkl,81 complex / 1491 mutants)**:
  - **per-structure Spearman = 0.0083** ← 近 0
  - overall Spearman = 0.225;overall Pearson = 0.129
- 预测 csv:`ibex-records/ESM3ΔG-Megascale-SKEMPIv2/results/eval_esm3dg_mega_skempi_aug1.csv`;stage-1 ckpt:`cache/..._stage1.pt`。

| 方法(同 test split)| per-structure | overall |
|---|---|---|
| ProteinMPNN (StaB-ddG baseline) | **0.448** | 0.531 |
| task1: ESM3ΔG + LoRA → SKEMPI | 0.071 | 0.256 |
| **task2: ESM3ΔG → Megascale → SKEMPI(本run)** | **0.008** | 0.225 |

**结论 / 观察**:
- **加 Megascale folding-ddG 的 stage-1 反而有害**:task2 < task1(per-structure 0.008 < 0.071)。stage-1 在 folding ddG 上收敛到 0.81,但把模型推到了一个**对 SKEMPI binding-ddG 几乎训不动**的初始点(stage-2 loss 卡 4)。folding ddG(单域、逐残基均值)与 binding ddG(复合物分解)目标不一致,stage-1 似乎造成了负迁移 / catastrophic 干扰。
- 与 task1 共同指向:**"忠实微调 ESM3ΔG(逐残基均值 ΔG)→ binding ddG" 在 per-structure(界面内排序)上迁移很差**。两者 overall(~0.22–0.26)≫ per-structure(~0–0.07),说明只学到跨复合物粗幅值,学不到界面内排序。
- ⚠ 待确认:stage-2 训不动是否部分由 `--resume` 加载问题导致(需诊断);但 task1(无 resume)同样差,核心问题更可能在**聚合方式 / 任务迁移**而非 resume。
