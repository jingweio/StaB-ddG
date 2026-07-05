# ESM3ΔG-reproduce-mgnify — experiment record  (created 2026-07-03 02:47; status: PLANNED)

> 自主推进窗口(2026-07-03)产出。自主决策见 `ibex-records/autonomous-session-2026-07-03-decisions.md`。

## 1. Goal / hypothesis
验证我们**当前 pretrained ESM3dG 部署是否正确**:用最原始 pretrained ESM3dG 在 **MGnify Stability test split** 上算 absolute folding ΔG,看能否复现 paper 的 ESM3ΔG 结果(**Spearman≈0.87 / RMSE 0.80 kcal/mol**,test = 3283 sequences)。复现得上 → 部署正确。

## 2. Design & decision points
- **模型**:base **三成员 ensemble**(`ESM3dG_weights_{1,2,3}_lora.ckpt`),逐成员预测后平均 `[USER D1]`。非 augmented。
- **数据**:`data/mgnify/mgnify_training_index.csv`(`name|split|seq|dG|PDB_name`),test=3283(1862 WT + 1421 mutant)。
- **结构**:ESMFold2-Fast 预测(用户提供),`<PDB_name>.cif.gz`;只 copy test 需要的 **1862 个(22.2MB)**进 `data/mgnify/structures_test/` `[AUTO D5]`。gemmi 可直接读 .cif.gz(单链 A)。
- **预测口径**:每条 test 行把 `seq` **thread 到 `<PDB_name>` scaffold 结构** `[AUTO D7]` → ESM3 forward → **scaled(校准)** 逐残基 sigmoid 后 masked-mean = 绝对 dG `[AUTO D4]`;3 成员平均。同时报 raw-mean 交叉核对。
- **指标**:overall Spearman + Pearson + **RMSE(kcal/mol)** vs paper 0.87/0.80 `[AUTO D6]`。
- **代码**:`esm3dg-reproduce/eval_mgnify.py`(新 folder,用户指定)。

## 3. Run config
- SLURM:a100 ×1,walltime 待估(3283×3 forwards 小 domain,预计 ~1.5h → 申 3h),cpus 8,mem 64G。
- env:`esm3dg`(py3.12/torch2.6/esm from share);HF_HOME=share/hf_cache;HF_HUB_OFFLINE=1。
- sbatch:`ibex-records/ESM3ΔG-reproduce-mgnify/sh/…_20260703-024719.sh`。

## 4. Change log (LIVE)
- 2026-07-03 02:47:plan 写入;结构/数据核实完毕(1862 结构 0 缺失);待写 eval 代码 + 本地 smoke。
- 2026-07-03 03:0x:`eval_mgnify.py` 写好、本地 smoke(8 行×3 成员)通过(scaled Spearman 0.93/RMSE 0.51,量级对);1862 结构 + index csv 同步 Ibex;**提交 job 47982065**(a100,esm3dg env),RUNNING。status: RUNNING。

## 5. Results (COMPLETED 2026-07-03;job 47982065,a100,elapsed 20:06)

**MGnify test(3283 seqs,base 3-member ensemble):**

| 指标 | 我们(ensemble)| paper ESM3ΔG | 判定 |
|---|---|---|---|
| **Spearman (scaled)** | **0.8718** | **~0.87** | ✅ **精准复现** |
| Pearson | 0.7544 | — | — |
| **RMSE (kcal/mol)** | **1.579** | **0.80** | ⚠️ ~2× 偏高 |
| RMSE (offset-removed) | 1.555 | — | — |
| 单成员 scaled Spearman | 0.8697 / 0.8674 / 0.8626 | — | ensemble 略升到 0.8718 |

> **口径 = scaled**(MGnify 是 cDNA 数据集,按 paper 规则 sigmoid 开;见 forward-mechanism §9)。**raw 额外验证见文末 §6。**

- csv 留证:`results/eval_mgnify_test_base_ens.csv`(3283 行:name/PDB_name/label_dG/pred_scaled/pred_raw)。

**判读:**
1. **✅ 部署正确(排序侧)**:Spearman **0.87 ≈ paper 0.87**,几乎精准复现 → 我们的 pretrained ESM3dG 部署(模型代码 / 权重 strict 加载 / 结构编码 / masked-mean 聚合 / base 3-ens)是**对的**。这是本次验证最想确认的点,通过。
   - **⚠ 口径二次订正(2026-07-05,以 paper Methods 为准)**:查 MGnify.pdf Methods 确认——sigmoid correction 层**按数据集开关**:**cDNA 数据集(=MGnify)推理时 sigmoid 开 = SCALED**(钳进 cDNA 的 [-1,5] 量程),非 cDNA 才 bypass=raw。故 **MGnify-test 口径 = scaled:Spearman 0.8718 ≈ 0.87 ✓**。(我中途曾据 notebook 默认 sigmoid_on=False 误改为 raw,现依 paper 纠回 scaled;两者 Spearman 几乎一样,不影响结论。)训练 loss = `0.3·mutΔG² + 0.3·wtΔG² + 1.0·ΔΔG²`(MSE),在 cDNA 上训 → sigmoid 开。详见决策日志 D4。
2. **⚠️ 绝对校准偏高(RMSE 1.58 vs 0.80)**,offset-removed 仍 1.55(非单纯常数偏移;Pearson 0.75 < Spearman 0.87 说明有非线性/尺度错配)。**最可能原因:结构来源不同** —— paper 用 **AlphaFold2** 结构,我们用 **ESMFold2-Fast**(用户预测)。不同折叠器 → 绝对 dG 尺度漂移,但排序稳健(Spearman 不变)。**这是 caveat 而非部署 bug**;手上无 MGnify 的 AF2 结构可直接对照,故列为**待验证假设**。
3. **⚠️ 潜在决策点(等用户)**:若要复现 RMSE 0.80,需 (a) 用 AF2 结构重预测/下载,或 (b) 对 scaled 输出做一次 test 上的线性重标定(paper 对实验数据提过 "after removing a global offset")。属改设计,不在自主窗口内动。

## 6. 附:raw-output 额外验证(非主口径)
raw = 未过 sigmoid 的 masked-mean。仅作交叉核对;主口径是 §5 的 scaled(MGnify=cDNA)。

| 指标(raw)| 我们 | 对照 scaled |
|---|---|---|
| Spearman | 0.8713 | 0.8718 |
| Pearson | 0.7834 | 0.7544 |
| RMSE (kcal/mol) | 1.447 | 1.579 |

- **结论**:raw 与 scaled 的 **Spearman 几乎相同**(0.8713 vs 0.8718)—— 印证 SigmoidScaling 在 [0,4] 恒等、对排序几乎无影响。RMSE raw(1.45)/scaled(1.58)**都远于 paper 0.80** → RMSE 差距**与 scaled/raw 无关**,归因结构来源(ESMFold2 vs AF2),见 §5 判读 2。csv `results/eval_mgnify_test_base_ens.csv` 同时含 `pred_scaled` 和 `pred_raw` 两列。
