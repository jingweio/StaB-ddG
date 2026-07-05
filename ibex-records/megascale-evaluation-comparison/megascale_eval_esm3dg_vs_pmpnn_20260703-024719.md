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
- 2026-07-03 03:0x:两个 eval 写好、本地 smoke 通过(1a ESM3dG 3DKM sp=0.33;1b ProteinMPNN 全 test per-domain **0.755**/overall 0.677 @mc=1,28 domains ~51k mutants)。esm3dg env 可同时 import stabddg(1b 无需单独 env)✓。代码同步 Ibex;**提交 job 47982066**(a100),1a max_batch=32、1b mc=20,RUNNING。
- 2026-07-03(cont.):jobs 完成(47982066,43min)。1a(raw)per-domain 0.7715、1b 0.7690。§5 填结果。
- 2026-07-05:口径修订——查 paper 确认 Megascale=cDNA 应 sigmoid 开=scaled;`eval_megascale.py` 改为同出 scaled+raw,**scaled 重跑 job 48064110(29:29,COMPLETED)**:scaled per-domain **0.7714**(≈raw 0.7715)。§5 主口径改 scaled、raw 挪 §6。status: DONE。

## 5. Results (COMPLETED 2026-07-03;job 47982066,a100,elapsed 43:28)

**Megascale test(28 domains,~51k mutants)folding-ΔΔG vs ddG_ML:**

| 指标 | **1a ESM3dG**(base 3-ens,**zero-shot**,MGnify-trained,**scaled**)| **1b ProteinMPNN-stage1**(20× MC,**Megascale-finetuned**)|
|---|---|---|
| **per-domain Spearman (mean)** | **0.7714** | **0.7690** |
| overall Spearman | 0.6077 | 0.6983 |
| overall Pearson | 0.6221 | 0.6422 |

> **1a 口径 = scaled**(Megascale 是 cDNA,按 paper 规则 sigmoid 开;由 job **48064110** scaled 重跑补齐,遵守 "cDNA→scaled" 规则)。**raw 几乎相同(per-domain 0.7715),见文末 §6 附录。** 1b ProteinMPNN 无 sigmoid,不涉及口径。

- csv 留证:`results/eval_megascale_esm3dg_base_ens_scaledraw.csv`(1a scaled+raw per-domain)、`results/eval_megascale_pmpnn_stage1_mc20.csv`(1b per-domain)。

**判读:**
1. **✅ 部署 sane + 迁移强**:per-domain 上 **ESM3dG 0.7714(scaled)≈ ProteinMPNN 0.7690**(ESM3dG 略高)。而 ESM3dG **对 Megascale 是纯 zero-shot**(只在 MGnify 训练过,从没见 Megascale),ProteinMPNN 却是**专门在 Megascale train 上 finetune** 的。ESM3dG 的折叠稳定性表示迁移到 Megascale **与专家模型持平** → 强验证部署正确 + 表示泛化好。
2. **overall 上 ProteinMPNN 更强(0.698 > 0.608)**:跨 domain 绝对排序 ProteinMPNN 占优。ESM3dG 又是"**per-domain 强 / overall 弱**"——与 SKEMPI 任务同一 pattern(擅长域内/复合物内相对排序,弱于跨样本绝对定标)。
3. **⚠️ 对比不对称**(已知,见 §2):1a 确定性 3 成员 ensemble;1b 随机 20× MC ensemble——各自报告级配置。且 1a 是 zero-shot、1b 是 in-domain finetuned,ESM3dG 仍打平,说明其 zero-shot 迁移相当强。
4. 交叉印证 mc=1 smoke:ProteinMPNN per-domain 0.755(mc=1)→ 0.769(mc=20),MC 略升、稳定,符合预期。

## 6. 附:1a ESM3dG 的 raw-output 额外验证(非主口径)
Megascale=cDNA → 主口径用 scaled(§5)。raw 仅作交叉核对:

| 1a ESM3dG 指标 | scaled(主)| raw(附)|
|---|---|---|
| per-domain Spearman | 0.7714 | 0.7715 |
| overall Spearman | 0.6077 | 0.6081 |
| overall Pearson | 0.6221 | 0.6152 |

- **结论**:scaled 与 raw **几乎完全一致**(per-domain 0.7714 vs 0.7715)→ 印证 SigmoidScaling 在 [0,4] 恒等、对 ddG 排序无影响,**对比结论(ESM3dG ≈ ProteinMPNN)与口径无关**。两列同存于 `results/eval_megascale_esm3dg_base_ens_scaledraw.csv`。
- **口径修订史**:1a 首跑(job 47982066,2026-07-03)用的是 raw;经查 paper Methods 确认 Megascale(cDNA)应 sigmoid 开=scaled,故 job **48064110**(2026-07-05)scaled 重跑补齐,主口径改为 scaled。结论不变。
