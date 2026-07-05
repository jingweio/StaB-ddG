# skempiv2-zeroshot-evaluation-comparison — experiment record (created 2026-07-05 15:56; status: PLANNED)

> 用户第 3 实验。设计经并行调研 workflow `wf_77829bd7-c6d`(4 研究 agent + chainbreak 对抗验证,verdict=CORRECT)。

## 1. Goal / hypothesis
在 **SKEMPIv2 test(81 complexes / 1491 mutants)binding ΔΔG** 上,**zero-shot** 对比两个"折叠稳定性模型"(都没在 SKEMPI 上训练过)经 StaB 分解迁移到 binding 的能力:
- **原始 ESM3dG**(MGnify-trained)vs **StaB stage-1 ProteinMPNN**(Megascale-finetuned,`stability_finetuned.pt`)。
- 附带回答:ESM3dG 吃 multi-chain 用**直接拼接** vs **chainbreak** 哪个更好。

## 2. Design & decision points
**三个 zero-shot 模型(都在 SKEMPI test binding ddG 上,无 SKEMPI 训练):**
| id | 模型 | 构造 | 口径/集成 |
|---|---|---|---|
| **A** | ESM3dG base 3-ens | multi-chain **直接拼接**(现 struct_to_seq_coords)| **raw**,3 成员平均 |
| **B** | ESM3dG base 3-ens | multi-chain **chainbreak**(`\|` + 每 pipe 一行 NaN atom37)| **raw**,3 成员平均 |
| **C** | ProteinMPNN stage-1 `stability_finetuned.pt` | 原生 chain-aware | 20× MC(StaB skempi_eval.py)|

**KEY DECISIONS:**
- **模型型号**:ESM3dG = **base 3 成员 ensemble**(weights_1/2/3),沿用 [[let-user-decide-model-config-choices]] 里 D1 的"原始 ESM3dG=base 3-ens"。ProteinMPNN = `model_ckpts/stability_finetuned.pt`(= Megascale stage-1,**未训 SKEMPI = zero-shot**)。
- **ESM3dG 口径 = raw**(用户要求 (2);binding ddG 非 cDNA,binding_ddG 天然走 raw 的 folding_dG,scaled=False)。
- **multi-chain 两构造(用户要求 (3))**:(a) 直接拼接;(b) chainbreak `\|`。chainbreak 经 workflow 对抗验证 = 官方 `from_protein_complex→encode` 路径,CORRECT。
- **指标口径(统一)**:用 StaB `baselines/eval_utils.compute_metrics`(**THRESHOLD=10**:per-structure Spearman = ≥10 突变的复合物均值 + overall Spearman/Pearson)。三模型 + baseline 全用同一口径 → 可直接对比 **StaB baseline 0.448 / 0.531**。三模型都产 `#Pdb,ddG,ddG_pred` 的 per-mutation CSV,再统一 compute_metrics。
- **chainbreak mask 细节**:folding_dG 的 masked-mean **把 `\|`(token 31)位置 mask 掉**(只平均真实残基;对无 pipe 的 A/Megascale/MGnify 是 no-op,安全)——采纳 verify agent 建议。
- **对标**:ProteinMPNN 全 StaB(stage-2,SKEMPI-finetuned)per-structure **0.448** / overall 0.531;我们之前 ESM3dG-SKEMPI-finetuned 0.158;本实验都是 **zero-shot 下限**。

**代码改动(local-first,不破坏现有实验):**
- `skempi_data.py`:`struct_to_seq_coords(sd, chainbreak=False)` 加 chainbreak 分支;`stab_mut_to_esm_tokens` 改 **pipe-aware**;`build_skempi(..., chainbreak=False)` 透传。默认 chainbreak=False → **现有 SKEMPI/其它实验行为不变**。
- `scorer.py`:`folding_dG`/`folding_dG_both` 加 `mask *= (seq_tokens != 31)`(no-op for pipe-free)。
- 新增 `esm3dg_stab/eval_skempi_ens.py`:zero-shot ESM3dG(不 load_adapters)、base 3-ens、raw binding_ddG、`--chainbreak` 开关、输出 per-mutation CSV。
- 新增 metrics 小工具(或直接调 `baselines/eval_utils.compute_metrics`)对 3 个 CSV 统一算指标。
- **ProteinMPNN**:直接用 `skempi_eval.py --checkpoint model_ckpts/stability_finetuned.pt --skempi_pdb_dir data/SKEMPI2_PDBs --run_name zeroshot_stability --ensemble 20`(其余默认;无 *-1 翻转)。

## 3. Run config
- SLURM:a100 ×1;可能 2 个 job(job1: ESM3dG A+B 串行,3-ens;job2: ProteinMPNN skempi_eval 20×MC)或合一。walltime 待估(ESM3dG SKEMPI 81 complex × 3 ens × 2 variant;ProteinMPNN 20×MC)。env=esm3dg(可 import stabddg,Task1b 已验证)。
- **隔离(用户要求 (4))**:新 dir `ibex-records/skempiv2-zeroshot-evaluation-comparison/`;fresh timestamp `20260705-155627`;job-name/文件名全新;**不碰 megascale-evaluation-comparison 及其它已有实验**。ProteinMPNN cache 用 StaB 默认 `cache/skempi_full_mask_pdb_dict.pkl`(首跑重建,几分钟),**不复用** ESM3 的 `skempi_esm3_test_pdb_dict.pkl`。
- sbatch:`ibex-records/skempiv2-zeroshot-evaluation-comparison/sh/…_20260705-155627.sh`。

## 4. Change log (LIVE)
- 2026-07-05 15:56:调研 workflow 完成(chainbreak 验证 CORRECT);plan 写入;project dir 建好。待编码 + 本地 smoke。

## 5. Results (jobs 完成后填)
- _待填:A/B/C 的 per-structure(THRESHOLD=10)+ overall Spearman/Pearson 对比表;concat vs chainbreak 差异;vs baseline 0.448 / vs ESM3dG-finetuned 0.158;结论(zero-shot 折叠→binding 迁移能力 + chainbreak 是否有用)。_
