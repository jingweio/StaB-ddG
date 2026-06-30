# esm-lora-replace — experiment record  (created 2026-06-30 13:51; status: RUNNING)

> 一篇 md 贯穿整个 two-stage 实验（Stage-1 / Stage-2 / eval / ablation 作为 §5 表格的行，不另开文件）。

## 1. Goal / hypothesis
把 StaB-ddG 的 folding 打分 backbone 从 **ProteinMPNN 换成 ESM3 + LoRA + stability head**（纯粹受控替换），测它在 **SKEMPI binding ΔΔG（per-interface Spearman, homology-OOD test split, THRESHOLD=10）** 上能否：(a) 超过旧 full-FT+likelihood ESM3 的 **0.242**；(b) 接近/超过 ProteinMPNN baseline **0.445**。
**Hypothesis:** LoRA（少量可训参数 → 抑制旧实验诊断到的 OOD 过拟合）+ 直接 regress 物理 ΔG 的 stability head（取代 likelihood-as-energy proxy）能缩小 0.242↔0.445 的 gap。

## 2. Design & decision points
**Controlled experiment:** StaB 的 data / 两阶段 / ΔΔG loss / SKEMPI eval / split **全不变**，只换 backbone + 微调方式。新旧用 config（`--backbone esm3_stab` vs `esm3`）区分，旧 0.242 路径 bit-不变可复现。

**KEY DECISIONS（选择 + why）:**
- **backbone = `esm3_stab`**: ESM3 (`esm3_sm_open_v1`) **frozen** + **LoRA r=4**（`inject_adapter_in_model`, target `["layernorm_qkv.1","out_proj"]`, dropout 0.15）+ **per-residue stability head**（`Linear1536→1536→LN→ReLU→1`, float32）→ `folding_dG = Σ per-res ΔG over valid residues`（extensive，保 binding 分解可加）。**不用 released ESM3ΔG 权重、不用 MGnify**（user 要求纯粹自训）。
- **loss = StaB 原本 ΔΔG MSE**；**NO sigmoid、NO absolute-ΔG term**（user decision ①）。
- **data = Megascale (Stage-1 folding, 239 train domains) → SKEMPI (Stage-2 binding)**；同 StaB。
- **Stage-1 HP (USER CONFIRMED 2026-06-30):** `lr=1e-3, AdamW, wd=0.05, warmup_frac=0.05, cosine, epochs=15, batch_size=2000 tokens`, per-epoch ckpt。(lr=1e-3 = 参考论文 ESM3ΔG 训练用的值；user 在 lr{1e-3 / 5e-4 / 扫} 中选定 1e-3。)
- **Stage-2 HP (PLANNED — 待 Stage-1 完成后向 user 确认再锁):** `lr≈5e-4, AdamW wd=0.05, warmup=0.05, cosine, epochs=15, batch=1000`, 从 Stage-1 收敛 ckpt chain。
- **ensemble:** ESM scorer `ensemble=1`（deterministic）；baseline 0.445 用 ProteinMPNN MC ensembling（口径差异，透明标注，对 ESM 不利方向）。

**Baselines compared against:** ProteinMPNN two-stage **0.445** (per-iface) / 0.531 (overall) / AUC 0.73；旧 full-FT-likelihood ESM3 **0.242**（不同方法，仅作能力阶梯参照）。
**Ablation:** SKEMPI-only（no Stage-1）复测 Stage-1 贡献（对照旧 recipe 的 0.148→0.242）。

## 3. Run config
- **SLURM:** a100 x1, `--time 24h`（Stage-1）, `--cpus-per-task 12`, `--mem 96G`。code+data 在 `/ibex/user/guoj0f/StaB-ddG/esm-replace`（per-branch path）。
- **env:** `esm-backbone`（peft 0.19.1, transformers 5.12.1, torch 2.5.1+cu124）；HF cache `/ibex/user/guoj0f/repos/esm/.hf_cache`（esm3 biohub mirror），`HF_HUB_OFFLINE=1`。env python 直接调（conda activate non-login 不稳）。
- **sbatch:** `ibex-records/esm-lora-replace/sh/megascale_stage1_esm3lora_20260630-135103.sh`
- **job ids:** smoke `47894361` (DONE); Stage-1 full `<TBD>`; Stage-2 `<TBD>`; eval `<TBD>`; ablation `<TBD>`。
- **outputs on /ibex:** Stage-1 ckpts `runs/s1_esm3lora/esm3lora_s1_{epoch}.pt`（adapter-only ~13MB/each）。

## 4. Change log
- 2026-06-30: code 半场 (Tasks 1-5) done + final whole-branch review (opus, Ready=Yes, 0 Critical/Important); pushed `a488532..63c6049`。
- 2026-06-30: Ibex sync 到 per-branch path。**GOTCHA**: `data/SKEMPI2_PDBs` 是指向 `/home/...`（worktree 外主仓库）的 symlink，`rsync -a` 拷成 broken link（Ibex 上只 1 个）；已用 rsync 真实文件修复（1092 on Ibex）。其余 data 为真文件。
- 2026-06-30 13:45: Stage-1 GPU smoke（job `47894361`, `--limit 3`）COMPLETED 44s, loss 0.325, adapter ckpt 13.1MB → **bf16 `out.embeddings` GPU path 验证通过**（final-review residual 关闭）。
- 2026-06-30 13:51: 按新 ibex-usage §5（plan-first record md）写本文件；即将 submit Stage-1 full（lr 1e-3, 15 ep, 239 train domains）。

## 5. Results  (← fill AFTER jobs finish)
| stage / config | job id | per-iface Spearman | overall | notes |
|---|---|---:|---:|---|
| ProteinMPNN two-stage (baseline) | — | 0.445 | 0.531 | 旧 RESULTS.md，未重跑 |
| 旧 ESM3 full-FT + likelihood | — | 0.242 | — | 旧方法，参照 |
| **ESM3 LoRA+head two-stage (this)** | TBD | _pending_ | | esm3_stab |
| ESM3 LoRA+head SKEMPI-only (ablation) | TBD | _pending_ | | no Stage-1 |

(pending — Stage-1 训练中)
