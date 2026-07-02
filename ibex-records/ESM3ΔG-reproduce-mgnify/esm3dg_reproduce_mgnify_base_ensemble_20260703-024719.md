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

## 5. Results (job 完成后填)
- _待填:overall Spearman/Pearson/RMSE(scaled + raw)vs paper 0.87/0.80;per-member;结论:部署是否正确。_
