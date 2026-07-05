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

### [AUTO] D4 — Task2 报 scaled + raw 双份 ⚠️⚠️ 口径二次纠正(2026-07-05,以 paper Methods 为准 = scaled)
**最终认定(权威来源 MGnify.pdf Methods "Fine-Tuning with a Sigmoid-Corrected Stability Head"):** sigmoid correction 层**按数据集类型开关**——**cDNA 数据集(=MGnify)推理时 sigmoid 开 = scaled**;非 cDNA(量热/宽量程)才 bypass=raw。故 **MGnify-test 复现口径 = SCALED**。→ **我最初 D4(scaled)正确;中间基于 notebook 默认 sigmoid_on=False 改成 raw 是过度纠正、已作废。** 对齐 paper 用 **scaled:Spearman 0.8718≈0.87 ✓**;RMSE 1.58(scaled)/1.45(raw)都远于 0.80 → 与口径无关,结构来源(ESMFold2 vs AF2)假设成立。
**训练 loss(paper 明确)**:`L = 0.3·(ΔG_pred,mut−ΔG_true,mut)² + 0.3·(ΔG_pred,WT−ΔG_true,WT)² + 1.0·(ΔΔG_pred−ΔΔG_true)²`,ΔΔG_pred=ΔG_mut−ΔG_WT;在 MGnify(cDNA)上训 → sigmoid 开 → 三项都建在 scaled 输出上。base=3 次独立训练 ensemble(对上 D1)。
--- 以下为历史记录(已被上面覆盖)---
**原 D4(有误)**:我以为复现绝对 dG 要用 scaled(校准)输出为主。
**纠正**:核实 absolute-stability-predictor 的 notebook(`ESM3dG.ipynb`)后确认——**paper 的 MGnify-test dG-prediction 口径是 RAW**:`ESM3dG_predict(...)` 默认 `sigmoid_on=False` → 用 `pred_dg_avg`(raw masked-mean),不是 scaled。(ddG-scanning 才用 scaled。)
- **幸好我 scaled+raw 都报了**,正确对齐数(raw):**Spearman 0.8713 ≈ paper 0.87(更干净)**、RMSE 1.447(仍 ~1.8× 偏高)。
- **结论不变**:部署正确(Spearman);RMSE 偏差与 scaled/raw 无关(SigmoidScaling 在 [0,4] 恒等,scaled≈raw),**结构来源假设 ESMFold2 vs AF2 依旧成立**。
- **训练口径(dG/ddG loss 用 scaled 还是 raw)无法代码确认**:absolute-stability-predictor 未发布 training loop;仅从 checkpoint 命名 `..dg_and_ddg_sigmoid..ddg_bins..`(dG+ddG 联合、含 sigmoid、ddG 疑分箱)推断 dG loss 很可能建在 scaled 上、ddG 处理不明——待原作者确认。

### [AUTO] D5 — 只 copy test 需要的 1862 个结构进 branch(非全量 6.9G)
test split 3283 行(1862 WT + 1421 mutant)只需 **1862 个唯一 scaffold 结构**(mutant thread 到母体),共 22.2MB,全部命中 0 缺失。按 §1c-3 copy 进 `data/mgnify/structures_test/`(gitignored)。
- **影响**:自包含且省盘;若后续要评 train/val 需再 copy。

### [AUTO] D6 — 指标口径 = per-domain/structure Spearman(均值)+ overall Spearman/Pearson
与我们既有 eval + StaB `validation_step` 一致。Task2 另加 RMSE(kcal/mol)。

### [AUTO] D7 — mutant 序列 thread 到 scaffold 结构(both tasks)
Megascale/MGnify 的 mutant 都是 point mutation、与 scaffold 等长,按数据集设计 thread 到母体 AF2/ESMFold 结构(见结构说明文档 §7)。encode(CSV seq, scaffold coords)。加等长断言。

### [AUTO] D8 — task1 的 1a 与 1b 合并到同一个 job(串行)
1a(ESM3dG)+1b(ProteinMPNN)都在 Megascale test、数据小,放同一 sbatch 串行跑,省一次排队。本地已验证 `esm3dg` env 能同时 import `stabddg`(ProteinMPNN 依赖轻),故 1b 无需单独 env。

### [AUTO] D9 — max_batch(1a)=32、MC(1b)=20
1a ESM3dG:a100-80GB 上 small domain 用 max_batch 32(带 OOM→减半 fallback);1b:mc=20(matching run_stabddg 默认)。

## 进度快照(2026-07-03 03:0x)
- 结构 copy(1862/22MB)、代码、sbatch 全部本地验证 + 同步 Ibex。
- **提交:Task2 job 47982065(mgnify 复现)、Task1 job 47982066(megascale 对比)——均 RUNNING(秒级 backfill)。**
- 本地 smoke 预览数(非最终):task2 scaled Spearman 0.93/RMSE 0.51(n=8);task1 ProteinMPNN per-domain 0.755(mc=1 全 test)。等 Ibex 全量结果。

## ✅ 最终结果(2026-07-03,两任务均 COMPLETED)

**Task2 — MGnify 复现(job 47982065,20min):**
- scaled **Spearman 0.8718 ≈ paper 0.87** → **部署正确(排序侧)已验证**。
- RMSE 1.58 vs paper 0.80(~2×,offset-removed 1.55)→ 疑因**结构来源 ESMFold2(ours) vs AF2(paper)**,尺度漂移、排序稳健。列为待验证假设(手上无 MGnify AF2 结构对照)。

**Task1 — Megascale test 对比(job 47982066,43min):**
| | 1a ESM3dG(zero-shot) | 1b ProteinMPNN-stage1(Megascale-finetuned) |
|---|---|---|
| per-domain Spearman | **0.7714**(scaled)| 0.7690 |
| overall Spearman | 0.6077 | 0.6983 |
- per-domain **打平**(ESM3dG zero-shot 竟≈ Megascale 专家)→ 部署 sane + 迁移强;overall ProteinMPNN 占优(per-domain强/overall弱,同 SKEMPI pattern)。
- **2026-07-05 口径修订**:1a 首跑用 raw;经 paper 确认 Megascale=cDNA 应 scaled,job **48064110** scaled 重跑补齐(per-domain 0.7714≈raw 0.7715,结论不变)。Task1 md §5 主口径已改 scaled、raw 挪 §6。

**总结论:两个独立验证都表明 pretrained ESM3dG 部署正确**(Task2 精准复现 paper Spearman;Task1a zero-shot 迁移与 Megascale 专家持平)。

## 过程备注(供排查)
- [PROC] monitor 首次因 sbatch 内 1a/1b **各打印一次 `===== DONE =====`** 被首个 DONE 提前触发退出;已重挂"job 离队(sacct COMPLETED)"判定的 monitor 补捕 1b 结果,无数据损失。教训:多步 sbatch 的 monitor 退出条件应用 job-gone,而非中间 DONE 字符串。
- 全程未碰 esm-replace / 别 task 的 env/job;结构仅 copy test 需要的 1862 个进 branch;所有代码本地 smoke 后才上 Ibex。

_(后续新决策/结果继续往下追加)_
