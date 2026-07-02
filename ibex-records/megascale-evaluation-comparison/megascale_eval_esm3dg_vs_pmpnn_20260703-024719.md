# megascale-evaluation-comparison — experiment record  (created 2026-07-03 02:47; status: PLANNED)

> 自主推进窗口(2026-07-03)产出。自主决策见 `ibex-records/autonomous-session-2026-07-03-decisions.md`。

## 1. Goal / hypothesis
在 **Megascale test split(28 domains)** 上,对比:
- **(1a)** 最原始 pretrained ESM3dG(base 三成员 ensemble,MGnify 训练、对 Megascale 是 zero-shot transfer)
- **(1b)** StaB 完成 stage-1 finetuning 后的 ProteinMPNN(`model_ckpts/stability_finetuned.pt`,在 Megascale train 上 finetune 过)

两者的 folding-ddG 预测能力。用于**验证 ESM3dG 部署**(1a 给出 sane 数字?)+ 以 StaB 的 known-good 模型(1b)作为 benchmark 锚点。

## 2. Design & decision points
- **数据**:Megascale `data/megascale/Tsuboyama…csv` + `AlphaFold_model_PDBs` + `rocklin/mega_splits.pkl` 的 **test split(28 domains)**。folding ddG = `ddG_ML`(stabilizing=正,原样)。
- **(1a) ESM3dG eval**:base 三成员 ensemble `[USER D1]`;每 domain 每 mutant thread 到 AF2 结构 `[AUTO D7]` → `folding_ddG = dG(mut) − dG(wt)`(raw masked-mean,与我们 ddG pipeline 一致);3 成员平均。代码:`esm3dg_stab/eval_megascale.py`(新)。
- **(1b) ProteinMPNN eval**:`stability_finetuned.pt` `[AUTO D2]`;复用 StaB `stability_finetune.py:validation_step` 的逐 domain `folding_ddG`,**20× MC ensemble**(matching `run_stabddg.py` 默认)`[AUTO D3]`。代码:`esm3dg_stab/eval_megascale_pmpnn.py`(新,import StaB 的 model/data)。
- **指标**:per-domain Spearman(均值)+ overall Spearman/Pearson `[AUTO D6]`。
- **⚠ 不对称说明**:1a 确定性 3 成员 ensemble;1b 随机 20× MC ensemble —— 各自代表其报告级配置,对比时标注。

## 3. Run config
- SLURM:a100 ×1;1a 与 1b 可**同一个 job 串行**(都在 Megascale test,数据小);walltime 申 4h;cpus 8-12,mem 64-96G。
- env:1a 用 `esm3dg`;1b 用 ProteinMPNN(StaB 代码,纯 torch,可同 env——StaB 依赖轻)。**待本地确认 esm3dg env 能否 import stabddg**(若不行则 1b 单独用 StaB 的 env/依赖)。
- sbatch:`ibex-records/megascale-evaluation-comparison/sh/…_20260703-024719.sh`。

## 4. Change log (LIVE)
- 2026-07-03 02:47:plan 写入;mega_splits test=28、数据在 branch 已确认;待写两个 eval 代码 + 本地 smoke。

## 5. Results (jobs 完成后填)
- _待填:1a vs 1b 的 per-domain / overall Spearman 对比表;结论:ESM3dG 部署是否 sane、zero-shot transfer 到 Megascale 的能力 vs Megascale-finetuned ProteinMPNN。_
