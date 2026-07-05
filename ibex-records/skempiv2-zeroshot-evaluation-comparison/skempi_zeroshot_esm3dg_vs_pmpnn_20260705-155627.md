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
| **A** | ESM3dG base 3-ens | multi-chain **直接拼接**(concat)| **raw**,3 成员平均 |
| **B1** | ESM3dG base 3-ens | **chainbreak**(`\|`+NaN),`\|` **计入** mean(官方 ESM3dG 原样,default)| **raw**,3 成员平均 |
| **B2** | ESM3dG base 3-ens | **chainbreak**,`\|` **mask 掉**(不计入 mean,`--mask_pipe`)| **raw**,3 成员平均 |
| **C** | ProteinMPNN stage-1 `stability_finetuned.pt` | 原生 chain-aware | 20× MC(StaB skempi_eval.py)|

> **B1 vs B2(用户 2026-07-05 要求"两个都试")**:chainbreak 的 `\|` 位置该不该计入 masked-mean 无官方先例(ESM3 无 mean;ESM3dG 官方多链是 concat、从不用 `\|`)。B1=官方 ESM3dG 原样(计入,最小偏离);B2=把 `\|` mask 掉(更"只平均真实残基")。两者都跑,实测差异(预期极小,~0.5% 稀释)。concat(A)无 `\|`,不受影响。

**KEY DECISIONS:**
- **模型型号**:ESM3dG = **base 3 成员 ensemble**(weights_1/2/3),沿用 [[let-user-decide-model-config-choices]] 里 D1 的"原始 ESM3dG=base 3-ens"。ProteinMPNN = `model_ckpts/stability_finetuned.pt`(= Megascale stage-1,**未训 SKEMPI = zero-shot**)。
- **ESM3dG 口径 = raw**(用户要求 (2);binding ddG 非 cDNA,binding_ddG 天然走 raw 的 folding_dG,scaled=False)。
- **multi-chain 两构造(用户要求 (3))**:(a) 直接拼接;(b) chainbreak `\|`。chainbreak 经 workflow 对抗验证 = 官方 `from_protein_complex→encode` 路径,CORRECT。
- **指标口径(统一)**:用 StaB `baselines/eval_utils.compute_metrics`(**THRESHOLD=10**:per-structure Spearman = ≥10 突变的复合物均值 + overall Spearman/Pearson)。三模型 + baseline 全用同一口径 → 可直接对比 **StaB baseline 0.448 / 0.531**。三模型都产 `#Pdb,ddG,ddG_pred` 的 per-mutation CSV,再统一 compute_metrics。
- **chainbreak 聚合口径(2026-07-05 修订,follow 官方)**:**不特殊处理 `\|`**——用 **ESM3dG 原样的 masked-mean**(只排除 `<cls>`/`<eos>`,`mask[1:L-1]`,即 `\|` 按中间 token **包含**在 mean 里)。理由:masked-mean 是 ESM3dG 的聚合(非 ESM3),而 ESM3dG 官方多链是 variant(a)裸拼、从不喂 `\|`,故"`\|` 在 mean 里怎么办"无官方先例;为最小化非官方偏离,采用 ESM3dG 原样 mean。(曾考虑 verify agent 建议的"mask 掉 `\|`",因非官方、且用户要求 follow 官方 ESM3,已撤销;`\|` 仅 N-1 个、约 0.5% 稀释,影响小。若要严谨可 A/B。)
- **对标**:ProteinMPNN 全 StaB(stage-2,SKEMPI-finetuned)per-structure **0.448** / overall 0.531(THRESHOLD=10 口径,与本实验一致 → 可直接比)。
  - ⚠ **不以之前的 ESM3dG-SKEMPI-finetuned "0.158" 作参照**(用户 2026-07-05 指出那次实验存在问题):(1) 那次 `eval.py` 的 per-structure 用的是 **≥2 突变**口径,**不是 THRESHOLD=10**,与 0.448 / 本实验**不可比**;(2) 用户认为那次实验本身可能有问题(待复核)。故本实验只对 **0.448/0.531(同口径)** 对标,0.158 仅作历史备注、不作结论依据。

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
- 2026-07-05 15:56:调研 workflow 完成(chainbreak 验证 CORRECT);plan 写入;project dir 建好。
- 2026-07-05 16:xx:代码写好、两 variant 本地 smoke 通过;撤销非官方 mask-out(follow 官方);弃用 0.158 参照。提交 **3 job**:48071431(A concat)/ 48071432(B1 chainbreak mask-in 官方)/ 48071433(C ProteinMPNN 20×MC),均 RUNNING。
- 2026-07-05 16:xx:用户要求"两个都试"→ 加 `--mask_pipe` 可选 flag(default off),提交 **第 4 job 48071517**(B2 chainbreak mask-out)。4 job 全部 submitted。status: RUNNING。

## 5. Results (COMPLETED 2026-07-05;jobs 48071431/432/517/433,a100)

**SKEMPI test(81 complexes / 1491 mutants)zero-shot binding ΔΔG,统一 `skempi_metrics.py` THRESHOLD=10:**

| 模型(zero-shot)| per-structure (TH=10) | overall Spearman | overall Pearson |
|---|---|---|---|
| **A** ESM3dG concat | **−0.037** | −0.027 | 0.038 |
| **B1** ESM3dG chainbreak(`\|`计入)| **−0.031** | −0.039 | 0.037 |
| **B2** ESM3dG chainbreak(`\|`mask 掉)| **−0.031** | −0.039 | 0.037 |
| **C** ProteinMPNN stage-1(20×MC)| **0.398** | 0.452 | 0.440 |
| _参照_ ProteinMPNN 全 StaB(stage-2)| 0.448 | 0.531 | — |

- csv 留证:`results/eval_skempi_esm3dg_base_ens_{concat,chainbreak,chainbreak_maskpipe}.csv` + `eval_skempi_pmpnn_stage1.csv`。0 OOM-skip(81/81)。

**结论:**
1. **原始 ESM3dG zero-shot 在 SKEMPI binding ddG 上 ≈ 0(略负)** —— MGnify 折叠表示经 StaB 分解**没有可用的 binding 信号**。3 种多链构造(concat / chainbreak `\|`计入 / mask 掉)**几乎无差**(−0.031~−0.037)→ 信号本就 ~0,构造选择不影响。
2. **ProteinMPNN stage-1 zero-shot = 0.398**,近满 StaB(stage-2)0.448 → **StaB 的 binding 能力大部分来自 stage-1 Megascale 折叠 + StaB 分解,stage-2 SKEMPI 微调只 +~0.05**。
3. **两个"折叠→binding via 分解"模型 zero-shot 天差地别**(ProteinMPNN 0.398 vs ESM3dG ~0)。呼应之前发现:ProteinMPNN 多链/binding 训练分布内、其 dG=Σlog P 迁移强;ESM3dG 单体 MGnify 训练、其回归 head 迁移弱。
4. **对 exp4 的启示**:ESM3dG SKEMPI 微调是"从 ~0 起步造 binding 信号",要超越 0.448 是硬仗。
