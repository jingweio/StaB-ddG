# ESM3ΔG 前向机制 —— 从 (structure, sequence) 到 absolute ΔG

> 目的:讲清楚 **ESM3ΔG** 到底怎么"吃进结构 + 序列、吐出每个残基的 embedding、再回归成一个 absolute 折叠稳定性 ΔG"。
> 所有结论都附**代码位置**(`esm` 库来自 `/home/guoj0f/share/esm`,esm 3.3.0;ESM3ΔG 代码见本仓库 `esm3dg_stab/`)。
> 模型 = `esm3_sm_open_v1`(1.4B open,**d_model = 1536**)。
>
> 相关文档:折叠/结合 ΔG·ΔΔG 的**符号约定**见 [`../dG-ddG-sign-note/`](../dG-ddG-sign-note/)(在 `reproduce` 分支);
> Megascale vs MGnify 数据对比见 [`../megascale_vs_mgnify/`](../megascale_vs_mgnify/)。

---

## 0. TL;DR:一句话本质

把 **sequence** 和 **structure** 都 tokenize → 每条 track 各自查 embedding 表 → **逐元素相加**成每残基一个 1536-d 向量 → 经 transformer 在 **aa-site 维度**上做 attention → 取**最后一层隐藏态(pre-final-LayerNorm)**当作每残基 embedding → 一个 2 层 MLP **回归头**逐残基出一个标量 → **masked-mean** 成整蛋白 `dG_fold`。

- **是监督回归,不是 likelihood**(和 ProteinMPNN/StaB-ddG 的 `Σ log P` 本质不同)。
- **不是 masked-marginal**:整条真实序列一次性喂进去,不逐个 mask。
- **结构只用 N/CA/C 骨架**:侧链不入模(见 §7)。
- **~95% 是 token-embedding 建模**,但有 **1 层连续几何注意力**用真实坐标(见 §4),所以不是"纯" embedding。

---

## 1. 全景流程图

```
                 ESMProtein(sequence="MK...", coordinates=atom37[L,37,3])
                                    │
              ┌─────────────────────┴──────────────────────┐   ① encode(): tokenize
              ▼                                             ▼
   sequence_tokens[B,L]                    coords ──►[取 N,CA,C]──► VQ-VAE structure encoder
   (每残基 1 个 aa token,词表 64)                                  │            │
                                                     structure_coords[B,L,3,3]   structure_tokens[B,L]
                                                     (连续骨架坐标)             (离散码本 4096)
              │                                             │                    │
              │  ② EncodeInputs:每条 track 各自 nn.Embedding,然后【逐元素相加】  │
              ▼                                                                  ▼
        sequence_embed(64→1536)  +  structure_tokens_embed(4096+5→1536)  +  [ss8/sasa/function/plddt… = pad 默认]
              └──────────────────────────────┬──────────────────────────────────┘
                                             ▼
                                       x[B, L, 1536]      每残基 = "我是什么 aa" ⊕ "我在什么局部结构 token"
                                             │
              ③ TransformerStack (n_layers 层):
                 ├─ block0 = GEOMETRIC attention ── 吃 affine 帧(由 N,CA,C 建)← structure_coords  ★连续几何通道
                 └─ block1..N-1 = 标准 multi-head self-attention (SwiGLU FFN)   ← aa-site 维度上混合
                                             │
                     输出两个:post_norm = LayerNorm(x)  |  pre_norm = x  ◄── embeddings 取这个
                                             ▼
                                  embeddings[B, L, 1536]   ◄── 你说的"每个 aa site 的 embedding"
                                             │
              ④ ESM3_Stability_head 逐残基回归: Linear(1536→1536)→LayerNorm→ReLU→Linear(1536→1)
                                             ▼
                                        dg[B, L]   (每残基 1 个稳定性标量)
                                             │  masked-mean over 真实残基(排除 <cls>/<eos>/pad)
                                             ▼
                                   dG_fold  (标量,absolute 折叠稳定性,越大越稳定)
```

---

## 2. ① tokenize —— 结构和序列如何变成 token(`esm3.encode()`)

`ESM3dG` 用 `get_esm3_input_info_direct()`(`esm3dg_stab/esm3dg_model.py:314`)构造 `ESMProtein(sequence, coordinates)`,调 `base.encode(prot)`:

- **sequence** → `sequence_tokens[B,L]`:每残基一个氨基酸 token(词表 64)。
- **coords(atom37)** → **VQ-VAE structure encoder** → `structure_tokens[B,L]`:离散码本 4096。编码器**只取 N,CA,C**:
  - `esm/models/vqvae.py:213` `assert coords.size(-1)==3 and coords.size(-2)==3, "need N, CA, C"`
  - `esm/models/vqvae.py:298` `coords = coords[..., :3, :]`
  它把每个残基的**局部骨架几何**量化成一个离散 token。
- **coords** 同时留作 `structure_coords[B,L,3,3]`(也只 N,CA,C),供 ③ 的几何注意力用。

> `parse_CIF`(`esm3dg_model.py:273`)虽然把侧链读进了 atom37,但在 encode 时被 `[...,:3,:]` 切掉——**侧链是假象,下游没用**(见 §7)。

---

## 3. ② 输入嵌入 —— 多轨道 embedding **相加**(`EncodeInputs.forward`, `esm/models/esm3.py:93-148`)

ESM3 是**多轨道**模型,每条 track 有独立 embedding 表,forward 里**全部相加**:

```python
# esm/models/esm3.py
self.sequence_embed        = nn.Embedding(64,       d_model)   # :74
self.structure_tokens_embed= nn.Embedding(4096 + 5, d_model)   # :80
# ss8 / sasa / function / residue / plddt …  各有自己的表
...
return ( sequence_embed + structure_embed                       # :139-148
       + ss8_embed + sasa_embed + function_embed + residue_embed
       + plddt_embed + structure_per_res_plddt )
```

- 对 **ESM3ΔG** 真正有信息的只有 **`sequence_embed` + `structure_embed`**;其余 track(ss8/sasa/function/residue/plddt)在 `forward` 里被填成 pad/mask 默认 token(`esm3.py:331-351`),embedding 照加,但每个位置是"未知"常量。
- 结果 `x[B,L,1536]` 的含义 = 每残基:**"我是什么氨基酸" ⊕ "我处在什么局部结构 codebook"**。

---

## 4. ③ Transformer —— aa-site 维度 attention + 一条连续几何通道(`TransformerStack`)

`esm/layers/transformer_stack.py`:

- `n_layers_geom = 1`(:32),`use_geom_attn = i < n_layers_geom`(:48)→ **只有第 0 层是 geometric attention**,其余是标准多头自注意力(SwiGLU FFN)。
- 第 0 层吃 `affine`(残基坐标系),由 `build_affine3d_from_coordinates(structure_coords)` 从 **N,CA,C** 构建:
  - `esm3.py:353-356` `structure_coords = structure_coords[..., :3, :]` → `build_affine3d_from_coordinates(...)`
  - `esm/utils/structure/affine3d.py:529` `N, CA, C = bb_positions.unbind(dim=-2)`
  这是**连续 3D 几何**(残基间相对朝向/距离)注入的通道,补在离散 structure token 之上。

**⚠ "纯 embedding 建模"的边界:** 主体确实是 token-embedding + attention;但第 0 层这条 geometric-attention **用的是真实连续坐标,不是查某个离散码本的 embedding**。所以准确说是「**离散 structure-token embedding(查表)+ 1 层连续几何注意力(真坐标)双通道注入结构**,其余全是标准 token-embedding attention」。

**两个输出**(`transformer_stack.py:115` `return self.norm(x), x, ...`):

| 返回 | 是什么 | 谁用 |
|---|---|---|
| `post_norm = self.norm(x)` | 最终 LayerNorm **之后** | ESM3 各 logit 头(sequence/structure/…)|
| `pre_norm = x` | 最后一层 block 输出、最终 LayerNorm **之前** | 作为 `embeddings` 返回 → **ESM3ΔG 取这个** |

---

## 5. ④ ESM3ΔG 抓 pre-norm 的 `embeddings`,逐残基回归 → masked-mean

```python
# esm3dg_stab/esm3dg_model.py  ModelWrapper.forward
features          = base_outputs.embeddings          # :94  ← pre-norm 隐藏态 [B,L,1536]
stability_output  = self.stability_head(features)    # :95  ← 逐残基回归
```
```python
# esm3dg_stab/model_utils.py  ESM3_Stability_head  :56-68
Linear(1536→1536) → LayerNorm(1536) → ReLU → Linear(1536→1)      # 2 层 MLP 回归头
```
```python
# esm3dg_stab/scorer.py  folding_dG  :106
dG = (dg * mask).sum(dim=-1) / valid       # masked-MEAN over 真实残基 → 整蛋白 dG_fold
```

- `embeddings`(你问的"aa embedding")= **最终 LayerNorm 之前、融合了 seq+struct+几何 上下文的每残基 1536-d 隐藏态**。
- head 是**新增的监督回归头**(不是 ESM3 原生 logit 头),在 MGnify 上训练成 `MEAN(dg) ≈ 真实 ΔG`。
- 聚合是 **MEAN**(原版 `ESM3dG_predict` 的 `pred_dg_avg = sum/valid_len` 同款)。

---

## 6. 为什么 ESM3ΔG 弃用 ESM3 原生输出头、而新造一个回归头?

ESM3 原生有 6 个输出头(`OutputHeads`,esm3.py:151-185),都是**逐位点的 token 分类头**(`sequence_head`→64 维词表、`structure_head`→4096 维…,`RegressionHead` 是 esm 里的命名误导,实为分类投影)。ESM3ΔG 全部弃用、另加一个 `Linear→LN→ReLU→Linear` 的**标量回归头**。原因有两层:

**原因一(硬性):原生头的输出空间不是标量 ΔG。** 要预测 absolute 折叠稳定性(一个 kcal/mol 连续值),必须有输出标量的头;原生 6 个头没有一个输出空间对得上 → 无论如何都得新加。

**原因二(更深刻):full unmasked sequence 进去后,原生 sequence-likelihood 头失去了稳定性区分度。**
- ESM3 是**双向 MLM**,这次 forward 把**完整未 mask 的序列**喂进去。位点 `i` 的表示里已含 `aa_i` 本身(`sequence_embed(aa_i)` 在输入就进了残差流,双向 attention 让 `i` 能"看到自己")。
- 于是原生 `sequence_head` 在位点 `i` 会趋向**直接"抄"输入的氨基酸**:`log P(aa_i | 结构, 其余序列, 且 aa_i 自己) ≈ 很高`,**不管该残基稳不稳定** → likelihood 被输入序列绑死,对稳定性没有区分度。

**为什么 ProteinMPNN 的 `Σ log P` 却能当稳定性代理?** 因为它是**自回归 inverse folding**:预测位点 `i` 时只给结构 + 已解码的其他位点,**绝不把 `aa_i` 当输入**。所以 `log P(aa_i | 结构)` 真实衡量"这个氨基酸在这个结构环境里合不合适"。它成立的前提恰恰是"模型没看到被打分的那个残基"——ESM3 在"单次 forward + 全序列可见"下把这个前提破坏了。

**能救回原生头吗?能,但 ESM3ΔG 没选。** 走 **masked-marginal / pseudo-log-likelihood**(逐位点 mask 掉 `aa_i`、读 `log P(aa_i | 结构+其余)`,即 ESM-1v/ESM2 的零样本变体打分套路)可以恢复信号,但:① **贵**——要 L 次 forward(ESM3ΔG 只 1 次);② 仍**非校准的物理 ΔG**,只是相对排序。

**ESM3ΔG 的选择:** 单次 forward 拿每残基 embedding(编码了"`aa_i` 处于该 序列+结构 全局上下文的局部环境",信息远比单个 likelihood 标量丰富)→ 监督回归头映射到 ΔG。信息不浪费、便宜、且借 96 万条 MGnify 标签直接校准到 kcal/mol。本质是两种范式的取舍:

| | likelihood-as-energy(ProteinMPNN/StaB)| supervised head on embedding(ESM3ΔG)|
|---|---|---|
| 前提 | 模型**不能看到**被打分残基(自回归/mask)| 看到全序列无妨,靠 head 从表示里学 |
| 依赖标签 | 弱(likelihood 本身即先验)| 强(960k MGnify ΔG 监督)|
| forward 成本 | 天然 1 次 | 1 次 |
| 输出 | pseudo-log-likelihood(≤0)| 校准的 absolute ΔG(kcal/mol)|

---

## 7. 关键澄清(易踩的坑)

1. **不是 likelihood**:ESM3ΔG 的 dG 是回归头读 embedding,不是 `Σ log P(aa|struct)`。likelihood 是 ProteinMPNN/StaB-ddG 的做法。
2. **不是 masked-marginal**:整条真实序列 + 结构一次性喂进去,`embeddings` 是 full-context 隐藏态;**不逐个 mask aa 再预测**(那是 ESM2/ESM-1v 的零样本 pseudo-perplexity)。
3. **突变怎么被感知**:做 ΔΔG 时只换 `sequence_tokens`(structure_tokens + coords 固定在 WT)。②里只有突变位点的 `sequence_embed` 变,经 ③ attention 扩散到全序列 embedding,④重新读出 → dG 变化。`ddG = dG(mut) − dG(wt)`。
4. **侧链不入模**:结构信息只来自 N,CA,C(structure token + 第 0 层几何注意力都 `[...,:3,:]`)。侧链(CB、CG…)和骨架 O 都不进模型 —— 这是 ESM3 架构决定的,原版训练和本实验都如此,也是 backbone-类模型做 ΔΔG 的**共同天花板**(ProteinMPNN 同样)。

---

## 8. 与 ProteinMPNN / StaB-ddG 的对比

| | ProteinMPNN / StaB-ddG | **ESM3ΔG** |
|---|---|---|
| 每残基量 | `log P(aa_i \| 结构)` —— 原生 inverse-folding **likelihood** | 回归头作用于 **embedding** 的**学习标量** |
| 新增参数 | 无(用原生 21 类 softmax 头) | 有:2 层 MLP stability head(+ SigmoidScaling) |
| 结构编码 | 骨架(N,CA,C,O + 虚拟 CB)图消息传递 | 骨架 N,CA,C:离散 structure token + 1 层几何注意力 |
| 聚合 | **SUM** over residues → 序列 log-likelihood | **MEAN** over residues → absolute ΔG |
| dG 含义 | pseudo-log-likelihood(≤0,越大越稳定) | 校准的 absolute ΔG(−1~5 kcal/mol,越大越稳定)|
| 微调 | 全量 FT 所有权重 | LoRA(r=4)+ head,trunk 冻结(= ESM3ΔG 原生训练法)|
| binding ΔΔG | `complex − binder1 − binder2`(同一 folding 机器)| **完全一致**(StaB 分解逻辑保留)|

> 符号朝向:ESM3ΔG 的 dG「越大越稳定」,与 StaB 内部 dG(Σ log P)、以及 SKEMPI/Megascale 标签(stabilizing=正)**全部同向**;本实现无任何符号翻转,详见 [`../dG-ddG-sign-note/`](../dG-ddG-sign-note/)。

---

## 9. 训练细节(权威来源:MGnify paper `Sources/MGnify.pdf` Methods)

> 来自 paper Methods 的 "Fine-Tuning with a Sigmoid-Corrected Stability Head" / "Loss Function with ΔΔG Regularization"。
> ⚠ **官方仓库(absolute-stability-predictor)未发布训练代码**——只放了推理 + 权重;以下训练细节**以 paper 为准**。

### 9.1 训练数据 + 微调
- 训练集 = **MGnify Stability**(cDNA display proteolysis 测),960,216 序列(525k WT + 435k 点突变,无 indel),与 test 集 <30% 序列同一性。
- LoRA(r=4,target `["layernorm_qkv.1","out_proj"]`)+ stability head + SigmoidScaling **可训**,ESM3 trunk **冻结**(见 §2/§6)。
- **base = 3 次独立训练的 ensemble**(paper: "ensembles of three independent training runs";"single"=单次)。→ 我们两个 Task 用的就是 base 3-成员 ensemble。

### 9.2 Loss —— 三项联合 MSE(paper 明确)
$$L = 0.3\,(\Delta G_{pred,mut}-\Delta G_{true,mut})^2 + 0.3\,(\Delta G_{pred,WT}-\Delta G_{true,WT})^2 + 1.0\,(\Delta\Delta G_{pred}-\Delta\Delta G_{true})^2$$
其中 $\Delta\Delta G_{pred}=\Delta G_{mut}-\Delta G_{WT}$。→ **同时监督 WT 绝对 dG(0.3)+ mutant 绝对 dG(0.3)+ ΔΔG(1.0,权重最大)**。
> 对比:ProteinMPNN/StaB **只能训 ddG**(其 dG=未校准 Σlog P);ESM3ΔG 因 dG 校准,**能同时训绝对 dG + ΔΔG**。

### 9.3 Sigmoid correction 层 + "按数据集开关"(最关键)
head = 2 层 MLP(逐残基)→ **piecewise sigmoid correction**(= 代码 `SigmoidScaling`):
```
f(x) = 2/(1+e^(−α1·x)) − 1        for x < 0      # 负端 sigmoid 尾
     = x                          for 0 ≤ x ≤ 4   # 中段线性(恒等)
     = 3 + 2/(1+e^(−α2·(x−4)))    for x > 4       # 正端 sigmoid 尾   (α1,α2 可训练斜率)
```
**为什么**:cDNA display proteolysis 只在 **[-1,5] kcal/mol** 可靠(超出被截断/失真)→ sigmoid 把预测钳进该量程,可训斜率允许外推。

**⚠ 开关规则(paper 原文,本节要点):**
> "At inference time, we **activate the sigmoid correction layer for cDNA-based datasets** … For **non-cDNA datasets (e.g., calorimetry), we bypass this layer**, under the assumption that the model has already learned to map stabilities to the real kcal/mol scale during training."

- **cDNA 数据集(MGnify、Megascale)→ sigmoid 开 = scaled**(钳进 [-1,5]);
- **非 cDNA(量热/CD/宽量程如 S1724)→ sigmoid 关 = raw**(允许 >5 外推;raw 已在训练中学到 kcal/mol)。
- **按数据集来源/量程开关,不是按 dG-vs-ddG 分**;整个 head 输出(WT-ΔG / mut-ΔG / ΔΔG)统一走这套。训练在 cDNA(MGnify)上做 → sigmoid 开 → 三项 loss 都建在 scaled 输出上。
- **作者消融**:S1724(宽量程)关 sigmoid 改善大蛋白(1LVE 7.7、1YYX 9.5 kcal/mol)预测;Supplementary Fig 9 直接对比 sigmoid vs non-sigmoid ESM3ΔG。

### 9.4 数据集归类 + 我们从 paper 明确的口径结论
| 数据集 | cDNA? | 说明 | 我们任务口径 |
|---|---|---|---|
| **MGnify Stability** | ✅ | 本 paper,cDNA proteolysis | Task2 复现 → sigmoid 开 = **scaled** |
| **Megascale**(Tsuboyama)| ✅ | 同 cDNA proteolysis | Task1 做**相对 ddG 排序**,raw/scaled 排序几乎无差 → 用 raw |
| **SKEMPIv2** | ❌ | **binding 亲和力**(ITC/SPR),**非折叠、非 cDNA** | 我们 binding ddG 回归 → sigmoid 不适用,用 raw |

**从 paper 明确的两条结论(留档):**
1. **MGnify(cDNA)复现口径 = scaled**;我们 Task2 scaled Spearman **0.8718 ≈ paper 0.87** → **pretrained ESM3dG 部署正确**(排序侧)。
2. **RMSE 1.58(scaled)/1.45(raw)都远于 paper 0.80 → 与 scaled/raw 无关**,归因于**结构来源:ESMFold2-Fast(我们) vs AlphaFold2(paper)**——paper 的 sigmoid 恰校准到 cDNA 的 [-1,5] 尺度,换折叠器 → 绝对尺度漂移,排序稳健。

---

## 10. 代码位置索引

**esm 库(`/home/guoj0f/share/esm`,editable)**
- `esm/models/esm3.py:62-148` `EncodeInputs`(多轨道 embedding 相加);`:151-185` `OutputHeads`;`:267-389` `ESM3.forward`;`:353-356` 坐标切 N,CA,C + 建 affine
- `esm/layers/transformer_stack.py:32/48` 几何层数;`:115` 返回 post_norm/pre_norm
- `esm/models/vqvae.py:213/298` structure encoder 只取 N,CA,C
- `esm/utils/structure/affine3d.py:519-562` `build_affine3d_from_coordinates`(N,CA,C→帧)

**ESM3ΔG(本仓库 `esm3dg_stab/`,vendored from absolute-stability-predictor, MIT)**
- `esm3dg_model.py:80-96` `ModelWrapper`(取 `base_outputs.embeddings`);`:102-121` `TransferModel.forward`;`:273` `parse_CIF`;`:314` `get_esm3_input_info_direct`;`:345` `ESM3dG_predict`(原生 `pred_dg_avg` = MEAN)
- `model_utils.py:16-40` `SigmoidScaling`;`:56-68` `ESM3_Stability_head`
- `scorer.py:96-120` `folding_dG`(masked-mean)/ `folding_ddG` / `binding_ddG`
