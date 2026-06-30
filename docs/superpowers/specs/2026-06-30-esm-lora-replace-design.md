# ESM-LoRA-Replace 设计文档 (spec)

- **日期**: 2026-06-30
- **状态**: 已批准，待写 implementation plan
- **Working branch**: `esm-replace`（不另开分支；新旧实验靠 **config（`--backbone`）** 区分）
- **project_name (ibex-records)**: `esm-lora-replace`
- **关联旧实验**: `esm-backbone-test/`（旧 esm-replace：full-FT + likelihood，结论 0.242）—— 本实验**完全隔离**，不触碰其任何产物。

---

## 1. 背景与动机

StaB-ddG（arXiv:2507.05502）用 folding energy 预测突变对 **binding ΔΔG** 的影响，headline metric = **homology-OOD SKEMPI test split 上的 per-interface Spearman**，ProteinMPNN baseline = **0.445**。

我们之前的 esm-replace 实验把 ProteinMPNN 换成 **ESM3 全量微调（full-FT）+ inverse-folding likelihood** 打分（`folding_dG = Σ log P(s|struct)`），经过彻底 HP sweep 后最优只到 **0.242**（train ~0.565），诊断为 **OOD generalization 瓶颈**（能拟合 train、泛化差），并非 under-tuning。

**两个改进假设（用户提出）：**
1. **不该 full-FT** 那么大的 ESM3，应该用 **LoRA** 参数高效微调（减少可训练参数 → 抑制 OOD 过拟合）。
2. **应该换掉 likelihood 打分**，给 ESM3 接一个专门的 **stability head**，直接 regress folding energy proxy（ΔΔG），而不是用序列似然当能量。

**参考工作**：Cho, Tsuboyama, Rocklin et al. 2026, *"Accurate protein stability prediction for small domains using mega-scale experiments"*（repo: `absolute-stability-predictor`，模型 ESM3ΔG / SaProtΔG）。该工作的 recipe **正是上面两个假设**：ESM3 frozen + **LoRA r=4** + **per-residue stability head**，并提供两条直接支持假设的证据：
- **LoRA > full-FT**：ESM3 LoRA Spearman 0.87 vs ESM3 full fine-tune 0.84（其 stability test）。
- **head regression ≫ likelihood**：zero-shot 结构似然与 absolute ΔG 相关性弱（ProteinMPNN cross-entropy −0.40，ESM-IF pseudo-perplexity 0.43 最佳），而 head regressor 达 0.87。

> **注意（隔离原则）**：本实验**借用其架构 recipe（LoRA + head）**，但 **不使用其 released ESM3ΔG 权重、不使用 MGnify 数据**。我们在 **StaB-ddG 自己的 pipeline 与数据**（Megascale Stage-1 → SKEMPI Stage-2）上**从 pretrained ESM3 自行训练**，做一个"把 ProteinMPNN 换成 ESM3"的**纯粹受控实验**。

---

## 2. 目标与成功标准

**目标**：在 StaB-ddG 框架内，把打分骨干 ProteinMPNN 换成 **ESM3 + LoRA + stability head**，其余（数据、两阶段、loss、eval、split）全部不变，测量 SKEMPI binding ΔΔG 的 **per-interface Spearman**，与 **ProteinMPNN 0.445** 和**旧 full-FT-likelihood ESM3 0.242** 做 apples-to-apples 对比。

**成功标准**：
- **主**：产出一个**严格隔离、可复现**的 ESM3-LoRA-head 两阶段 test 数值，以及 **Stage-1 ablation**（SKEMPI-only vs two-stage）在新 recipe 下的贡献。
- **假设验证**：新 recipe 是否 **> 0.242**（验证 LoRA+head 优于 full-FT+likelihood），以及离 **0.445** 还有多远（是否缩小 OOD gap）。
- 即便是 clean 的负面结果（仍 < 0.445）也是有效科学结论；关键是**严谨、隔离、与基线口径一致**。

---

## 3. 受控实验框架（变 vs 不变）

| 维度 | 原 StaB-ddG / 旧 esm-replace | 本实验 (esm-lora-replace) |
|---|---|---|
| Backbone | ProteinMPNN（旧 run: ESM3） | **ESM3** (`esm3_sm_open_v1`) |
| 可训练参数 | 全部（旧 run: **full-FT 1.4B**） | **LoRA r=4** adapters + 小 head；base **冻结** |
| `folding_dG` | `Σ log P(s\|struct)` likelihood | **per-residue stability head → Σ = ΔG regression** |
| 数据 | Megascale (S1) + SKEMPI (S2) | **相同**（无 MGnify、无 released 权重） |
| 两阶段 pipeline | Megascale→SKEMPI | **相同**（`finetune.py` 已支持两 stage） |
| Loss | ΔΔG 上的 MSE | **相同**（ΔΔG MSE，无 sigmoid、无 absolute-ΔG 项） |
| Eval / metric / split | SKEMPI test, per-iface Spearman (THRESHOLD=10) | **相同**（对比 0.445 / 0.242） |

**唯一改变**：backbone（ProteinMPNN→ESM3）+ 微调方式（full-FT+likelihood → LoRA+head regression）。

---

## 4. 架构：`ESM3StabScorer`（新目录 `esm-backbone-test/track_esm3stab/`）

实现现有 `common/scorer.py: SequenceScorer` 接口（`folding_dG` / `get_wt_seq` / `backbone_module`）。

- **Base**: `ESM3_sm_open_v0`（与现有 `track_esm3/esm3_scorer.py` 一致；加载后统一 dtype：CPU float32 / GPU bf16）。
- **LoRA**（`peft.get_peft_model`）：`LoraConfig(r=4, target_modules=["layernorm_qkv.1","out_proj"], lora_dropout=0.15)` —— paper/repo 配置。base 全部 frozen，仅 LoRA adapters 可训练。
- **Stability head**：`Linear(1536,1536) → LayerNorm → ReLU → Linear(1536,1)`（即 repo 的 `ESM3_Stability_head`）。**无 SigmoidScaling**（per 决策①：纯 ΔΔG 监督，absolute offset 自由，sigmoid 是 absolute-ΔG assay 校准件，不需要）。
- **`folding_dG(domain, seqs)`**：ESM3 forward → `out.embeddings` `[B,T,1536]` → head → per-residue ΔG `[B,T]` → 在 **valid residue 位置（排除 BOS/EOS/chainbreak）求和** = `[B]` 的 ΔG。
  - **求和（extensive）** 而非 repo 的 average，保证 StaB 的 `binding_ddG = ΔΔG_complex − (ΔΔG_b1 + ΔΔG_b2)` 分解**可加**。
  - 复用 `track_esm3/esm3_struct.py: ESM3StructTokenizer`（固定结构 per domain，与旧 scorer 同一套对齐逻辑）。
  - **待实现期验证**：ESM3 forward 输出确有 `.embeddings` 字段（旧 scorer 用 `.sequence_logits`；repo `ModelWrapper` 用 `base_outputs.embeddings`）。smoke 前确认。
- **`backbone_module`**：返回包含 ESM3(LoRA) + head 的 wrapper（用于 eval 的 checkpoint 加载）。
- **新增 hooks**（供 `finetune.py` 用，见 §5）：
  - `trainable_parameters()` → 只含 LoRA + head 的参数。
  - `freeze_base()` → 确保 base 的 `requires_grad=False`。
  - `save_adapters(path)` / `load_adapters(path)` → 只存/读 LoRA + head 的 state_dict（小文件）。
- **命名**：backbone key = **`esm3_stab`**（**故意不叫 `esm3dg`**，避免与 released 模型混淆）。

---

## 5. 训练 pipeline 改动（conditional + additive，绝不破坏旧路径）

现有 `run/finetune.py` 两个训练循环（`finetune` for SKEMPI、`finetune_stability` for Megascale）都：(a) `optimizer = _build_optimizer(backbone.parameters(), args)`；(b) `backbone.requires_grad_(True)`；(c) 每 epoch `torch.save(backbone.state_dict())`。这对 LoRA 有两个问题：`requires_grad_(True)` 会**解冻 frozen base**；保存全量 state_dict 浪费且巨大。

**改动（全部条件化，旧 backbone 行为 bit-for-bit 不变）：**
1. **可训练参数**：若 scorer 暴露 `trainable_parameters()`，optimizer 用它；否则回退 `backbone.parameters()`（旧行为）。
2. **冻结**：把无条件的 `backbone.requires_grad_(True)` 改为：若 scorer 有 `freeze_base()` 则调用之（base frozen、仅 LoRA+head 可训）；否则保持 `requires_grad_(True)`（旧行为）。
3. **checkpoint**：若 scorer 有 `save_adapters`/`load_adapters` 则存/读 LoRA+head；否则用全量 state_dict（旧行为）。`_load_checkpoint_for_esm` 同样条件化。
4. **Loss**：**不变** —— 两 stage 都是 `MSELoss` on ΔΔG（`finetune_stability` 用 `scorer.folding_ddG`；`finetune` 用 `StaBddG.binding_ddG`）。新 scorer 的 `folding_ddG = folding_dG(mut) − folding_dG(wt)` 由基类自动给出，天然适配。
5. **batch size**：LoRA 去掉了 1.4B base 的 Adam optimizer state，显存大降 → token budget 可远高于旧 full-FT 的 500。`DEFAULT_BATCH_SIZE` 给 `esm3_stab` 设一个更大的默认（Ibex 上实测调），可加速训练。

> 旧 `--backbone esm3`（full-FT likelihood）与 `mpnn`/`esmc*` 路径**完全不受影响** → 0.242 run 仍可精确复现。

`common/build_scorer.py` 增加 `esm3_stab` 分支（构造 `ESM3StabScorer`，传 `pdb_dir`）。

---

## 6. 两阶段训练 + eval 计划

1. **Stage-1（Megascale folding ΔΔG）**：`--backbone esm3_stab --stage stability`，从 pretrained ESM3 起训 → 小的 LoRA+head ckpt。
2. **Stage-2（SKEMPI binding ΔΔG）**：`--stage skempi --checkpoint <stage1>` → 最终 ckpt。
3. **Eval**：`run/eval.py --backbone esm3_stab --checkpoint <stage2> --split test` → per-interface Spearman + overall Spearman + ROC AUC（与旧口径一致，THRESHOLD=10，全 81-complex test）。
4. **Ablation（顺带）**：SKEMPI-only（无 Stage-1，从 pretrained 直接 Stage-2）→ 在新 recipe 下重新测 Stage-1 的贡献（对照旧 recipe 的 0.148→0.242）。
5. **HP**：先用合理 LoRA 默认（lr≈1e-3/5e-4 量级、AdamW、warmup、cosine —— 参考 repo config learn_rate 0.001），若首个 two-stage 结果有潜力再小幅 sweep lr/rank。
6. **流程纪律**：每个 full run 前先 **local smoke**（`--limit` 几个 domain/complex，2 epochs）验证 forward/backward/ckpt save-load 通；再上 Ibex。

---

## 7. 结果隔离（用户硬性要求，双重保证）

- **配置隔离**：新 backbone key `esm3_stab` + 新目录 `track_esm3stab/`；旧 `track_esm3/`、`--backbone esm3`、旧 `esm-backbone-test/results/`（RESULTS.md、0.242、comparison.png）**一律不动**。
- **记录隔离**：所有 sbatch + results md + pkl 走新 `ibex-records/esm-lora-replace/`（见 §8），与旧 `esm-backbone-test/results/` 物理分离。
- **memory**：写一条**新** memory fact（link 到、但**不覆盖**旧 `esm-backbone-experiment.md` 的结论）。

---

## 8. Ibex / 计算计划（遵循更新后的 ibex-usage skill）

- **记录布局**（worktree 内，随 `esm-replace` branch 版本控制）：
  - sbatch → `ibex-records/esm-lora-replace/sh/{task}_{datetime}.sh`
  - results md → `ibex-records/esm-lora-replace/{task}_{datetime}.md`（中英结合、compact：同类结果汇一张表）
  - pkl 留证 → `ibex-records/esm-lora-replace/results/`
  - bulky `*.out`/`*.err`/`*.pkl`/ckpt → **gitignore**；scripts + md commit & push 到 `esm-replace`（`git push origin HEAD`，**绝不 push main**）。
- **task_name**（distinctive）：`megascale_stage1_esm3lora`、`skempi_stage2_esm3lora`、`skempi_test_eval_esm3lora`、`skempi_only_ablation_esm3lora` 等。
- **Per-branch ibex path**：`/ibex/user/guoj0f/StaB-ddG/esm-replace/`（`{repo}/{branch_safe}`）；从**本 worktree** rsync，`--exclude .git --exclude .claude/worktrees`。
- **数据 locality**：确认 worktree 自带 `data/` 拷贝（Megascale AF PDBs + Tsuboyama Dataset2/3 CSV + mega_splits.pkl + SKEMPI csv/PDBs）；若有引用 worktree 外路径，**copy 进来**并 gitignore。实现期核查。
- **Env**：在 Ibex 已有 `esm-backbone` env 上加装 `peft`（mirror 本地 env 名）；改 env 后 `compileall` 预编译字节码。
- **GPU**：**a100 only**（v100 仅在 a100 极紧张且征得用户同意后用）。Stage-1/Stage-2/eval 各 1×a100。
- **代码流程**：本地改 → commit/push → 本地 smoke → rsync 到 per-branch path → 提交 sbatch；不在 Ibex 上开发代码。

---

## 9. 风险

- **Domain shift**：ESM3ΔG recipe 原本在 60–80aa 小单体域上做 folding ΔG；我们用在更大的 SKEMPI 复合物 + binding ΔΔG 上。这正是实验要测的，不影响开工，但可能限制上限。
- **ESM3 长复合物 OOM**：旧 full-FT 时最长复合物（L≳1000，O(B·L²) attention）会 OOM-skip 6/120 train complexes；LoRA 显存更省，可能缓解，但仍需监控、沿用 skip-and-continue。eval 在全 test set 无 skip。
- **`out.embeddings` 可用性 / head 维度（1536）**：实现期 smoke 前确认。
- **集成口径**：ESM scorer 用 ensemble=1（确定性）；0.445 baseline 用 ProteinMPNN MC ensembling —— 沿用旧实验的透明标注。

---

## 10. Out of scope (YAGNI)

- 不复刻 paper 的 joint (absolute ΔG + ΔΔG) loss / sigmoid scaling（决策①）。
- 不使用 released ESM3ΔG 权重、不使用 MGnify 数据。
- 不做 SaProt 线、不做 ESMC 线（ESMC-6B 仍暂停）。
- 不改 StaB 的数据 split、binding 分解、eval metric。

---

## 11. 已定决策

- **①Loss**：沿用 StaB ΔΔG MSE（无 sigmoid / 无 absolute-ΔG）。
- **②Branch**：留 `esm-replace`，新旧靠 `--backbone esm3_stab` vs `esm3` 区分；pipeline 改动条件化、纯增量。
- **③结果**：`ibex-records/esm-lora-replace/`（新 skill convention）。
- per-residue ΔG 求和（extensive）；新目录 `track_esm3stab/`；env 在 `esm-backbone` 上加 `peft`。
