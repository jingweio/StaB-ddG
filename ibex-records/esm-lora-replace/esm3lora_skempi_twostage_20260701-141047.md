# esm-lora-replace — experiment record (canonical, faithful env)  (created 2026-07-01 14:10; status: RUNNING)

> **Canonical run** on the rebuilt **faithful py3.12 env** (matching reference Cho et al.). The prior
> esm-lora-replace attempt was purged — it ran on a non-faithful env (py3.11 hack + transformers 5.12.1
> + esm fork), so its results (incl. the "lr=1e-3 too high" impression) are discarded. One md for the
> whole two-stage experiment (stages/ablation as §5 rows).

## 1. Goal / hypothesis
StaB-ddG 的 folding scorer 从 ProteinMPNN 换成 **ESM3 + LoRA + stability head**（纯粹受控替换），测 **SKEMPI binding ΔΔG per-interface Spearman**（homology-OOD test, THRESHOLD=10）vs **ProteinMPNN 0.445**。Hypothesis: LoRA（抑制 OOD 过拟合）+ 直接 regress ΔG 的 head（取代 likelihood proxy）能缩小 gap。

## 2. Design & decision points
Controlled: StaB 的 data / 两阶段 / ΔΔG loss / SKEMPI eval / split 全不变，只换 backbone + 微调方式。新旧靠 `--backbone esm3_stab` vs `esm3` 区分。
**KEY DECISIONS:**
- **backbone `esm3_stab`**: ESM3 (`esm3_sm_open`) frozen + LoRA **r=4** (`layernorm_qkv.1`,`out_proj`, dropout **0.15**) + per-residue stability head (Linear1536→1536→LN→ReLU→1, float32) → `folding_dG=Σ per-res ΔG`（extensive）。不用 released ESM3ΔG 权重、不用 MGnify。
- **loss = StaB ΔΔG MSE**；no sigmoid, no absolute-ΔG term。
- **data = Megascale (Stage-1, 239 train domains) → SKEMPI (Stage-2)**。
- **ENV (faithful to reference, rebuilt 2026-07-01):** py3.12.13 / torch2.6.0+cu124 / **transformers4.57.6** / **esm3.3.0 官方 main (commit cf002f1d)** / biotite1.7.1 / peft0.19.1；权重 `biohub/esm3-sm-open-v1`。（旧 py3.11+transformers5.12.1+esm-fork 环境不忠实、已弃。）
- **Stage-1 HP:** **lr=5e-4** (canonical), AdamW, warmup_frac=0.05, cosine, **15 epochs**, batch_size=2000 tokens, per-epoch ckpt。(参考 MGnify 用 200 ep；我们 Megascale 用 15 ep + per-epoch ckpt，按 Stage-2 eval 判优，不足再延。)
  - ⚠️ **lr 历史:** 初始按参考 `config_esm3.py` 用 **lr=1e-3** (job 47933357) → **在忠实环境上发散** (mean train loss 连涨 1.11→2.52→2.67 @ep1-3, cosine lr 仍近峰值) → ep3 kill。**USER DECISION 2026-07-01 17:15: 降到 lr=5e-4 重跑** (job 47936491)。此发散在忠实环境复现,证明"1e-3 偏高"是真实信号、非旧环境伪影。
- **Stage-2 HP (PLANNED, 待 Stage-1 后确认):** lr≈5e-4, AdamW, warmup0.05, cosine, 15 ep, batch1000, 从 Stage-1 收敛 ckpt chain。
- **ensemble=1**（deterministic）；baseline 0.445 用 ProteinMPNN MC ensembling（口径差异，透明标注）。

**Baselines:** ProteinMPNN two-stage **0.445** (per-iface) / 0.531 (overall)。
**Ablation:** SKEMPI-only（no Stage-1）测 Stage-1 贡献。

## 3. Run config
- **SLURM:** a100 x1, cpus 12, mem 96G. Stage-1 `--time 24h`.
- **env:** `esm-backbone` (py3.12, versions above); env python `/ibex/user/guoj0f/anaconda3/envs/esm-backbone/bin`. sbatch exports `HF_HOME=/ibex/user/guoj0f/share/hf_cache` + `HF_HUB_OFFLINE=1`.
- **code+data:** `/ibex/user/guoj0f/StaB-ddG/esm-replace` (per-branch); esm code + weights in `/ibex/user/guoj0f/share`.
- **sbatch:** `sh/{smoke_stage1,megascale_stage1}_esm3lora_20260701-141047.sh` (旧, lr1e-3); **canonical Stage-1 = `sh/megascale_stage1_esm3lora_lr5e4_20260701-171537.sh`** (lr5e-4)
- **job ids:** GPU smoke `47932618` (DONE, loss 0.325 — Ibex faithful stack validated); Stage-1(lr1e-3) `47933357` (**DIVERGED @ep3, CANCELLED**); **Stage-1(lr5e-4) `47936491` (✅ COMPLETED, 11:57h, 15ep; train loss 0.355→谷底 0.163@ep10-11→0.188@ep15)**; Stage-2 `<TBD>`; eval `<TBD>`; ablation `<TBD>`.
- **outputs:** Stage-1(lr5e-4) ckpts `runs/s1_esm3lora_lr5e4/esm3lora_s1_lr5e4_{epoch}.pt` (adapter-only ~13MB); 发散的 lr1e-3 run 在 `runs/s1_esm3lora/`。

## 4. Change log
- 2026-07-01: 大整改完成 —— 废弃/删 repos/esm（污染 fork）；建 share/ 共享存储；**重建 py3.12 忠实 env 对齐参考**；清旧记录 + Ibex 整个重建。本地 8/8 offline 测试过；Ibex env 验证版本一致。
- 2026-07-01 14:10: plan-first record；即将跑 GPU 冒烟 → canonical Stage-1 (lr 1e-3)。
- 2026-07-01 14:34: GPU 冒烟 (job 47932618) COMPLETED 40s, loss 0.325, adapter ckpt 13.1MB, device=cuda, 无 error → **Ibex 忠实栈 (py3.12+esm3.3.0官方+biohub权重+a100 bf16) 验证通过**。
- 2026-07-01 14:3x: 提交 Stage-1 = job `47933357` (lr 1e-3, AdamW, warmup0.05, cosine, 15 ep, batch 2000, 239 train domains, a100, --time 24h)。
- 2026-07-01 17:15: **Stage-1(lr1e-3, 47933357) 发散** — mean train loss 连涨 1.11→2.52→2.67 (ep1-3), cosine lr 仍近峰值 → ep3 kill (CANCELLED, elapsed 2:33)。**USER DECISION: 降 lr→5e-4 重跑 = job `47936491`** (其余全同: AdamW/warmup0.05/cosine/15ep/batch2000, 独立 out dir `runs/s1_esm3lora_lr5e4`)。此发散在**忠实环境复现**,坐实"1e-3 偏高"为真实信号,非旧环境伪影。
- 2026-07-02 12:22: **Stage-1(lr5e-4, 47936491) ✅ COMPLETED** (11:57h, 15 epoch)。train loss **单调下降到 ep10-11 触底 0.163**,之后随 cosine lr→0 微升到 ep15 的 0.188。健康收敛(对比发散的 lr1e-3)。15 个 per-epoch adapter ckpt 全存于 `runs/s1_esm3lora_lr5e4/`。⚠️ **ep15 非 train-loss 最优(ep10/11 才是)** → Stage-2 chain 的 ckpt 待定(ep11 vs ep15 vs 都测,见下)。

## 5. Results (fill AFTER jobs)
| stage / config | job id | per-iface Spearman | overall | notes |
|---|---|---:|---:|---|
| ProteinMPNN two-stage (baseline) | — | 0.445 | 0.531 | 旧 RESULTS.md，未重跑 |
| **ESM3 LoRA+head two-stage (this)** | TBD | _pending_ | | esm3_stab, faithful env |
| ESM3 LoRA+head SKEMPI-only (ablation) | TBD | _pending_ | | no Stage-1 |

(pending)
