# ESM3ΔG-SKEMPIv2 — fine-tune ESM3dG on SKEMPI binding ΔΔG (created 2026-07-06;status: DESIGN LOCKED, sweep pending 2 open decisions)

> 用户第 4 实验(REDO,老实验代码+结果已清 local+Ibex)。目标:在**原始 pretrained ESM3dG** 上用 SKEMPI train-split 微调 binding ΔΔG,**超越 StaB 论文 report 的 ProteinMPNN(per-structure 0.448 / overall 0.531,THRESHOLD=10 口径)**。
> 本文档汇总本轮所有决策点 + 支撑证据。批量 probe/前置实验见 [probe_batch_a100_20260706-132142.md](probe_batch_a100_20260706-132142.md)。

## 1. Goal / hypothesis
- 原始 ESM3dG(MGnify 折叠稳定性预训练)经 LoRA 微调,能否在 SKEMPI binding ΔΔG 上追平/超越 StaB(ProteinMPNN)0.448?
- **起点很低**(exp3 实测:原始 ESM3dG zero-shot binding ≈ 0/−0.03,MGnify 折叠表示对 binding 无迁移信号)→ 微调是"从 ~0 造 binding 信号",硬仗。

## 2. 架构 & 可训练部分
- **冻结 ESM3(esm3_sm_open_v1,d_model=1536)+ LoRA r=4(target `["layernorm_qkv.1","out_proj"]`,dropout 0.15)+ stability head(1536→1536→1)+ SigmoidScaling**。只 LoRA+head+scaling 训练 = **4,144,131 参数**;trunk 冻结。
- binding_ddG = folding_ddG(complex) − folding_ddG(b1) − folding_ddG(b2);folding_ddG = dG(mut) − dG(wt);dG = stability head 对 embedding 的 masked-mean。
- **【决策】LoRA rank 保持 4**:这样加载 MGnify 微调好的 LoRA 作为 **warm-start 继续微调**(不是从头训);改 r 或 target_modules 会 shape 不匹配 → LoRA 随机重初始化 = 从头训。可自由改的超参(不影响复用):lr/optimizer/wd/epochs/batch/seed/dropout。

## 3. 口径:raw(非 scaled)
- **【决策】inference 和 training/fine-tuning 都用 raw output**。理由:binding ddG 非 cDNA 数据集 → SigmoidScaling OFF;`binding_ddG→folding_dG` 天然走 `scaled=False`。已核对代码:训练 loss=MSE(binding_ddG, label),binding_ddG 走 raw。

## 4. 数据 & split
- **【决策】不切 val-split**(见 §5c 证据:StaB 没放 val、SKEMPI 也没用 val);**全 train-split 训练**。
- SKEMPI:train 120 复合物 / 3050 突变;test 81 / 1491。split 用 StaB 的 `data/SKEMPI/{train,test}_pdb.pkl`(同源 clusters 划分,与 baseline 同)。
- **【决策】训练时丢弃 OOM 复合物(不开 checkpointing)**:做训练 step 时连 micro-batch=1 都 OOM(a100-80GB)的复合物 → `oom_skipped_complexes` 完全丢弃。**实测(训练 run + 显存模型一致)= 6 个复合物 / 157 突变(占 3050 的 5.1%)**,阈值 L≳~1475 tok;点名的最大 3 个:3VR6(L=3406,16mut)、1KBH(L=2123,85mut)、4GXU(L=1937,3mut),另 3 个 L∈[~1475,1930)。〔订正:早前只按 probe 点名的最大 3 个写成“104 突变”,实际 `oom_skipped=6`→157。〕
  - **代价**:训练样本比 StaB 少 157 个(StaB 用极小 ProteinMPNN、L=3406 也塞得下、不丢);含全 train 集最富数据的 1KBH(85mut)。已知、接受。
  - **v100-32GB 对比(2026-07-07 估算,a100 显存曲线校准)**:若换 v100,B=1 OOM 阈值降到 L≳~760 → **丢 23/120 复合物 = 19.2% / 424 突变 = 13.9%**(~4× a100 的损失)→ 佐证 v100 不适合本 memory-bound 训练。
- **test 不丢任何样本**:eval=`no_grad`(显存低 ~5-10×,exp3 实测 81/81 全过);且那 3 个巨型复合物都在 **train**,test 最大仅 L≈1029。→ **test = 完整 81/1491,与 StaB 0.448 同口径可直接比**。
- **不加 Megascale stage**:exp4 = 纯 SKEMPI 微调(from 原始 pretrained ESM3dG)。task2 已证 Megascale folding stage-1 对 binding **净负**(THRESHOLD=10 下 0.148 < task1 的 0.193)。

## 5. 训练配置决策
### 5a. lr sweep
- **【决策】{1e-6, 5e-6, 1e-5, 5e-5, 1e-4}(5 个)**。

### 5b. epochs / 存 ckpt / 选模型
- **【决策】max-epochs = 50**(不搞自动 early-stop)。理由:200ep(拉齐 StaB)≈ 92h/job 完全不可行。**a100 2ep 计时 smoke(job 48111099)实测:epoch1=1670s, epoch2=1673s ≈ 27.9min/ep**(120 复合物,oom_skipped=6)→ **50ep ≈ 23.4h + build(~10min) → 设 walltime 30h**(buffer;会路由到 gpu72)。
- **【决策】每 10 epochs 存一个 ckpt**(`--save_freq 10` → ep10/20/30/40 + final ep50)。
- **【决策】事后逐 ckpt 在 test 上 eval,随时看轨迹**。
  - ⚠ **口径提醒**:"逐 ckpt 挑 test 最好" = 在 test 上选模型(轻微 test-peeking)。StaB 0.448 是固定训练后的数(无 val 选择)→ **报结果时同时给 (i) final-ep50(与 StaB 严格同口径)和 (ii) best-of-{ep10..50}(test-selected 上界)**。

### 5c. 为何不切 val(调研结论)
- **StaB 没放出 val-split**:`data/SKEMPI/` 只有 train/test 的 clusters+pdb,**无 val 文件**。
- **StaB 的 SKEMPI 也没用 val**:`skempi_finetune.py` 默认 `--val_split_path=""` + `--train_val_freq=-1` → val_epoch 不跑;放出的模型 = 固定跑满 epochs、存 final + 每 10ep 周期 ckpt,**无 val 选择**。(`early_stopping_*.pt` 是 Megascale stage-1 的 ckpt 名,非 SKEMPI。)
- StaB 若开 val,指标 = **val per-structure Spearman**(逐复合物平均,val_ensemble/trials,**不带 THRESHOLD=10**)—— 供参考,本实验不用。

### 5d. 梯度累积(对齐 StaB 的优化粒度)—— 已实现 + 数值验证
- **背景(probe 实测)**:训练显存随 L 塌缩(attention 激活 O(L²)),max 训练 batch:L≲600→6-8,L~800→4,L~1030→2,L>1500→0。StaB/ESM3dG 有效 batch 比 ~3-5×(指标相关复合物)。
- **StaB finetune 机制**:`for sample(复合物): for chunk in range(0,N,M=10000//L): backward; step; zero_grad` —— **每个 M-chunk 一次 optimizer step,无累积,逐复合物**(不同复合物更新分开,是结构约束:binding_ddG 的 forward 需该复合物固定的 3 份结构编码,物理上无法跨复合物组 batch)。StaB 唯一优势是 M 大。
- **【决策+实现】给 `finetune.py` 加梯度累积**:每个 **窗口 Mwin = batch_tokens//L = 10000//L 个突变** 累积成 **1 次 optimizer step**,物理 micro-batch ≤ `max_batch=4`(OOM-fallback →2→1),loss 按 `chunk/window` 缩放。→ **每次梯度平均的突变数、每复合物 step 数都与 StaB 逐字对齐**。
  - **数值验证 PASS**:累积梯度 == 全窗口梯度(全局相对误差 6e-6,cosine 1.0)。本地端到端 smoke PASS(4 复合物,drop/spearman/save 均 OK)。
  - **参数**:`batch_tokens=10000`(= StaB token budget = 累积窗口),`max_batch=4`(物理 micro-batch 显存上限)。

### 5e. 其它超参
- optimizer=adamw,weight_decay=0.05,seed=0。LoRA dropout 0.15(训练时开;StaB ProteinMPNN dropout=0)。
- **单成员 weights_1** 跑 sweep;最优配置再用 weights_1/2/3 做 3-member ensemble。

## 6. 多链构造 & sweep 规模(已定 2026-07-06)
- **【决策】构造 = "拼接含 chainbreak(`\|`+NaN 分隔) + output masked-mean 时 mask 掉 `\|`"** = `--chainbreak --mask_pipe`(= exp3 的 B2)。**单一构造**(不扫 concat、不扫 b1)。
  - 依据:exp3 已证 b1(`\|`计入)≈ b2(`\|`mask)完全相同;用户选 mask-out 变体(只平均真实残基)。
- **【决策】sweep = 10 job = {base, augmented} × 5 lr**(用户 2026-07-06 决定两套权重都扫;见 §11 base/aug)。无 staging。每 job:`--epochs 50 --save_freq 10 --batch_tokens 10000 --max_batch 4 --chainbreak --mask_pipe`,单成员(weights_1 / weights_augmented_1)。
- 代码:`finetune.py` 已加 `--chainbreak/--mask_pipe`(2026-07-06),本地 smoke PASS。
- **存储命名规范(防混淆,用户要求)** —— 所有结果制品都自描述带 `{base|aug}_lr{值}`:
  - **两个独立 sbatch**:`sh/ft_cbmask_{base,aug}_sweep_20260706-145437.sh`(各 `--array=0-4` 映射 5 lr)。
  - **训练日志**:`ft_cbmask_{base,aug}_lr%a_20260706-145437.{out,err}`(%a=lr 索引:0→1e-6,1→5e-6,2→1e-5,3→5e-5,4→1e-4;每 job 首行 echo `WEIGHTSET=... lr=...`)。
  - **adapter ckpt**:`cache/esm3dg_skempi_cbmask_{base,aug}_lr{值}[_ep{10,20,30,40}].pt`;final(ep50)= `..._lr{值}.pt`。
  - **eval 结果 CSV(事后)**:`results/eval_skempi_cbmask_{base,aug}_lr{值}_ep{N}.csv`。
- **【评测口径两级(用户 2026-07-07 定)】**:
  - **本地 A4500 快评** = 仅 **sanity-check**(验证流程没出大问题),结果只落 scratchpad,**不作正式结果、不进 `results/`、不参与选模型**。
  - **formal-evaluation + best-model 选择都在 Ibex a100 上做**(与训练同型号 GPU,整条 pipeline 一致;用户 2026-07-07 明确"选 best-model 也在 a100"):config 跑完后,用 `eval_skempi_finetuned.py`(一次加载、循环评多 adapter)在 a100 上**把该 config 全部 5 个 ckpt(ep10/20/30/40/50)都正式评一遍** → CSV 落 `results/`(留证)→ `skempi_metrics.py` THRESHOLD=10 → **从 a100 的数里选 best-ep**。正式结果表按 {base,aug} × {best-ep, ep50} 报(两者都已 a100 评过),对标 StaB 0.448/0.531 → commit。

## 7. 指标口径(统一)
- **per-structure Spearman THRESHOLD=10**(≥10 突变复合物均值,StaB baseline 0.448 口径)+ overall Spearman/Pearson(无阈值),用 `esm3dg_stab/skempi_metrics.py`。对标 **ProteinMPNN 0.448 / 0.531**。见 memory [[stabddg-per-interface-threshold10]]。

## 8. 支撑证据(前置实验)
- **exp3(zero-shot,skempiv2-zeroshot-evaluation-comparison)**:原始 ESM3dG zero-shot binding ≈ 0(concat −0.037 / chainbreak −0.031,b1≈b2);ProteinMPNN stage-1 zero-shot 0.398 ≈ 近满 StaB 0.448。→ ESM3dG 微调从 ~0 起步;构造 b1/b2 无差。
- **task1/task2(旧,口径订正后)**:SKEMPI-only per-structure(TH=10)0.193;+Megascale 0.148(净负);均 « 0.448。⚠ 那两次 run 可能有问题,本实验为干净 REDO。
- **probe(job 48106749)**:max 训练 batch vs L 曲线;3 巨型复合物 B=1 OOM。

## 9. Env / infra
- env `esm3dg`(py3.12+torch2.6+cu124,esm 3.3.0 editable from `share/esm`);a100-80GB;per-branch ibex `/ibex/user/guoj0f/StaB-ddG/MGnify-replace/`;`HF_HOME=/ibex/user/guoj0f/share/hf_cache`。
- base 权重 `data/esm3dg_weights/ESM3dG_weights_{1,2,3}_lora.ckpt`。

## 11. 权重溯源(2026-07-06 paper+repo 双源核查,both high-confidence)& base-vs-augmented 待决
- **发布的 ESM3ΔG 只有两套,都只训在 MGnify,均未用 Megascale**:
  - **base(`weights_{1,2,3}`)= MGnify-only**(960,216 seqs,K50dG DMSv4/v5/v7,WT+点突变,无 indel;ΔG+ΔΔG 联合 loss 0.3/0.3/1.0,ΔΔG 来自 MGnify WT/mut 对而非 Megascale)。MGnify test **0.87**(SaProtΔG 0.88)。
  - **augmented(`weights_augmented_{1,2,3}`)= 同一 MGnify 数据的去偏版**:受限子集(456k,剔除末端>2 无结构残基的域)+ 人工加末端片段(50% 概率、每端 1–15 随机残基)消 cDNA-proteolysis 末端 bias。MGnify test 上**更弱**,但真实纯化蛋白 S1724 上**最强** → **repo README 标 augmented 为 downstream“recommended”**,论文 Fig 3–6 全用 augmented。
- **既往误解订正**:早前以为“augmented=MGnify+Megascale 合并、0.89”——**错**。0.89/0.90 是实验重复性(trypsin vs chymotrypsin,Supp Fig 1),非模型;真正的合并数据(MGnify+Megascale+indels+ThermoMut)模型是 **SaProtΔG 消融(Fig 2a),未发布、非 ESM3ΔG**。
- **Megascale split**:ESM3ΔG **不训 Megascale**,仅用其做**评测**(复用 ThermoMPNN 的 28-WT / 28,172 点突变 test set,Dieckhaus 2024)。repo **不含任何 Megascale split 文件**,无 StaB `mega_splits.pkl` 的对应物。→ 与 StaB(训在 Rocklin `mega_splits.pkl`:train239/val31/test28)**不是同一 split、也不是同一用途**。
- **【开放决策 — 影响 exp4 起点】base vs augmented 作为微调起点**:
  - 现 sweep 用 **base weights_1**(沿用 2026-07-03 用户决策“原始 ESM3dG=base 3-ens”)。
  - 但 augmented 是论文/repo **明确推荐的泛化版**(真实蛋白最强);SKEMPI binding 属“真实世界泛化”而非 cDNA in-distribution → **augmented 可能是更好的微调起点**。待用户决定 base 还是 augmented(与 Megascale 无关,是去偏/泛化之别)。

## 10. Change log (LIVE)
- 2026-07-06(晚):**10-job sweep 提交**(base+aug × 5lr)。首提 walltime 30h → 仅进 `gpu,gpu72`、est start ~33h(07-07 23:50)。**改 walltime `23:59:00` 重提交** → 进 `gpu,gpu24,gpu72`(+gpu24 ~32 节点,实测 dry-run 确认路由),大幅提前。当前 job:**base 48114396 / aug 48114397**(旧 48113240/241 已 cancel)。监控轮询挂着(第一个 ckpt 或全结束时通知)。
- 2026-07-06:老实验清理(local+Ibex);probe 实测 max-batch(job 48106749);`finetune.py` 加梯度累积(数值验证 PASS)+ per-epoch 计时;本地端到端 smoke PASS;a100 2ep 计时 smoke **完成**(job 48111099,**27.9min/ep**);构造决策=chainbreak+mask_pipe(5 job,§6);**paper+repo 双源核查权重溯源(§11):base/augmented 均 MGnify-only,均未训 Megascale;订正 megascale_vs_mgnify/README 的“augmented=合并 0.89”错误**。待决:base vs augmented 起点。
