# 自主推进决策日志 — 2026-07-03(~10h 无人值守窗口)

> 用户 2026-07-03 离开约 10h、期间无法做决策,授权我自主推进两个 Ibex 验证任务。
> **本文件记录我在此窗口内每一个"自主拍板"的点(理由 + 影响)**,供用户回来排查。
> 用户已明确的决策会标注 `[USER]`;我自己拍板的标注 `[AUTO]`。

## 任务范围(用户下达)
- **Task 1 `megascale-evaluation-comparison`**:(1a) 最原始 pretrained ESM3dG 在 Megascale **test split** 上 eval;(1b) 完成 stage-1 finetuning 后的 stab-ddG(model_ckpts/ 下)在 Megascale test 上 eval;对比两者。
- **Task 2 `ESM3ΔG-reproduce-mgnify`**:最原始 pretrained ESM3dG 在 MGnify Stability dataset 上 eval,尝试复现论文结果(test 3283 条,paper ESM3ΔG Spearman≈0.87 / RMSE 0.80)。脚本放 `esm3dg-reproduce/`;mgnify 数据/结构按 §1c-3 copy 进本 branch。
- 共同目标:验证我们当前 pretrained ESM3dG 的**部署是否正确**。

## 决策清单

### [USER] D1 — 模型型号 = base 三成员 ensemble
"最原始 pretrained ESM3dG" 用 **base weights_1/2/3 三成员 ensemble**(逐成员预测后平均),非 augmented。两个任务都用它。(用户 AskUserQuestion 明确选择。)

### [AUTO] D2 — Task1b 的模型 = `model_ckpts/stability_finetuned.pt`
核实:`model_ckpts/` 三个 ckpt 全是 ProteinMPNN(各 1.66M)——`proteinmpnn.pt`(base)、`stability_finetuned.pt`(Megascale stage-1 后)、`stabddg.pt`(再过 SKEMPI 的两阶段)。用户说"完成 stage-1 finetuning 后的 stab-ddG @ model_ckpts/",对应的就是 **`stability_finetuned.pt`**。
- **影响**:1b 评的是 ProteinMPNN(不是我们的 ESM3dG-stage1;后者在 Ibex `cache/`)。若用户本意是"我们的 ESM3dG stage-1",需回来纠正重跑。

### [AUTO] D3 — Task1b ProteinMPNN eval = 复用 StaB `validation_step` + 20× MC ensemble
StaB 无独立 megascale-eval;`stability_finetune.py:validation_step` 正是逐 domain `folding_ddG`→per-domain Spearman + overall。ProteinMPNN 稳定性模型本身随机(随机 decoding order + backbone noise),StaB 报告级数字用 `run_stabddg.py` 的 **20× MC ensemble**。故 1b 用 **20× MC**(matching StaB 默认)。
- **影响**:1a(ESM3dG)是确定性 3 成员 ensemble,1b(ProteinMPNN)是 20× 随机 MC ——两种 ensemble 风格不同,但各自代表其"报告级"配置。对比时会明确标注这个不对称。

### [AUTO] D4 — Task2 报 scaled(校准)绝对 dG 为主,raw 作交叉核对
paper 报 RMSE 0.80 kcal/mol → 预测须在物理量程 → 用 **SigmoidScaling 校准后的 scaled 输出**(`sigmoid_on=True` 等价:逐残基 sigmoid 后再 masked-mean)。Spearman 对 raw/scaled 都近似,但为对齐 RMSE 用 scaled;同时也报 raw-mean 的 Spearman 交叉核对。
- **影响**:与我们 ddG pipeline(用 raw)不同,但那是 ddG;此处是复现绝对 dG,scaled 才对得上 paper 口径。

### [AUTO] D5 — 只 copy test 需要的 1862 个结构进 branch(非全量 6.9G)
test split 3283 行(1862 WT + 1421 mutant)只需 **1862 个唯一 scaffold 结构**(mutant thread 到母体),共 22.2MB,全部命中 0 缺失。按 §1c-3 copy 进 `data/mgnify/structures_test/`(gitignored)。
- **影响**:自包含且省盘;若后续要评 train/val 需再 copy。

### [AUTO] D6 — 指标口径 = per-domain/structure Spearman(均值)+ overall Spearman/Pearson
与我们既有 eval + StaB `validation_step` 一致。Task2 另加 RMSE(kcal/mol)。

### [AUTO] D7 — mutant 序列 thread 到 scaffold 结构(both tasks)
Megascale/MGnify 的 mutant 都是 point mutation、与 scaffold 等长,按数据集设计 thread 到母体 AF2/ESMFold 结构(见结构说明文档 §7)。encode(CSV seq, scaffold coords)。加等长断言。

_(后续新决策继续往下追加)_
