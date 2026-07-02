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
- **结构只用 N/CA/C 骨架**:侧链不入模(见 §5)。
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

> `parse_CIF`(`esm3dg_model.py:273`)虽然把侧链读进了 atom37,但在 encode 时被 `[...,:3,:]` 切掉——**侧链是假象,下游没用**(见 §5)。

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

## 6. 关键澄清(易踩的坑)

1. **不是 likelihood**:ESM3ΔG 的 dG 是回归头读 embedding,不是 `Σ log P(aa|struct)`。likelihood 是 ProteinMPNN/StaB-ddG 的做法。
2. **不是 masked-marginal**:整条真实序列 + 结构一次性喂进去,`embeddings` 是 full-context 隐藏态;**不逐个 mask aa 再预测**(那是 ESM2/ESM-1v 的零样本 pseudo-perplexity)。
3. **突变怎么被感知**:做 ΔΔG 时只换 `sequence_tokens`(structure_tokens + coords 固定在 WT)。②里只有突变位点的 `sequence_embed` 变,经 ③ attention 扩散到全序列 embedding,④重新读出 → dG 变化。`ddG = dG(mut) − dG(wt)`。
4. **侧链不入模**:结构信息只来自 N,CA,C(structure token + 第 0 层几何注意力都 `[...,:3,:]`)。侧链(CB、CG…)和骨架 O 都不进模型 —— 这是 ESM3 架构决定的,原版训练和本实验都如此,也是 backbone-类模型做 ΔΔG 的**共同天花板**(ProteinMPNN 同样)。

---

## 7. 与 ProteinMPNN / StaB-ddG 的对比

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

## 8. 代码位置索引

**esm 库(`/home/guoj0f/share/esm`,editable)**
- `esm/models/esm3.py:62-148` `EncodeInputs`(多轨道 embedding 相加);`:151-185` `OutputHeads`;`:267-389` `ESM3.forward`;`:353-356` 坐标切 N,CA,C + 建 affine
- `esm/layers/transformer_stack.py:32/48` 几何层数;`:115` 返回 post_norm/pre_norm
- `esm/models/vqvae.py:213/298` structure encoder 只取 N,CA,C
- `esm/utils/structure/affine3d.py:519-562` `build_affine3d_from_coordinates`(N,CA,C→帧)

**ESM3ΔG(本仓库 `esm3dg_stab/`,vendored from absolute-stability-predictor, MIT)**
- `esm3dg_model.py:80-96` `ModelWrapper`(取 `base_outputs.embeddings`);`:102-121` `TransferModel.forward`;`:273` `parse_CIF`;`:314` `get_esm3_input_info_direct`;`:345` `ESM3dG_predict`(原生 `pred_dg_avg` = MEAN)
- `model_utils.py:16-40` `SigmoidScaling`;`:56-68` `ESM3_Stability_head`
- `scorer.py:96-120` `folding_dG`(masked-mean)/ `folding_ddG` / `binding_ddG`
