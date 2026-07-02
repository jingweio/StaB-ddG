# 侧链信息:数据里有没有 vs 模型吃不吃

> 两个**正交**的问题,常被混为一谈,这里分开讲清楚并附**实测证据 + 代码位置**:
> - **A. 数据侧**:MGnify、Megascale(AF2 预测)、SKEMPIv2 的结构文件里**含不含侧链坐标**?
> - **B. 消费侧**:ESM3ΔG(以及 ProteinMPNN/StaB-ddG)在前向里**实际吃不吃**侧链?
>
> 相关文档:ESM3ΔG 前向机制见 [`../esm3dg-forward-mechanism/`](../esm3dg-forward-mechanism/)(§7 有侧链要点);符号约定见 [`../dG-ddG-sign-note/`](../dG-ddG-sign-note/)。

---

## 0. TL;DR:一张表

| 数据集 | 结构来源 | 文件**含侧链**? | 模型**消费侧链**? |
|---|---|---|---|
| **SKEMPIv2**(task1/task2 主评测)| 实验晶体结构(PDB)| ✅ 含(重原子全侧链)| ❌ 否 |
| **Megascale**(task2 stage-1)| **AlphaFold2 预测** | ✅ 含(侧链 **+ 氢原子**)| ❌ 否 |
| **MGnify**(ESM3ΔG 原生训练)| **AlphaFold2 预测** | ✅ 含(重原子全侧链)| ❌ 否 |

**结论:三类数据文件都带侧链(AF2 甚至带氢),但 ESM3(和 ProteinMPNN)在编码时只取骨架 N/CA/C,侧链一律丢弃。** 侧链信息"有,但没被用"。

---

## A. 数据侧:文件里含不含侧链?——实测(逐个抽查原子名)

### A.1 SKEMPIv2 — 实验晶体结构,含全侧链(重原子)
`data/SKEMPI2_PDBs/1A22_A.pdb`(共 1092 个 chain PDB):
```
第一个残基的原子:  N CA C O CB CG CD1 CD2 CE1 CE2 CZ        (含侧链)
全文件 distinct:    …CB CG CG1 CG2 CD CD1 CD2 CE CE1 CE2 CE3 CZ CZ2 CZ3 CH2
                     ND1 ND2 NE NE1 NE2 NH1 NH2 NZ OD1 OD2 OE1 OE2 OG OG1 OH SD SG…
```
→ **含完整侧链重原子**(无氢)。SKEMPI v2 本就基于 PDB 里的**真实实验复合物结构**(测过结合亲和力的),不是预测结构。

### A.2 Megascale — AF2 预测,含侧链 + 氢原子
`data/megascale/AlphaFold_model_PDBs/1A0N.pdb`(共 862 个 PDB):
```
第一个残基的原子:  N H H2 H3 CA HA C CB HB O CG1 HG11 HG12 HG13 CG2 HG21…   (侧链+氢!)
全文件 distinct:    …CB CG CG1 CG2 CD… + 大量 H/HA/HB/HG/HD/HE/HZ/HH… + OXT
```
→ **含完整侧链,且被完全质子化(带氢原子)**。这是 AF2 预测结构(经加氢/优化),信息最全。

### A.3 MGnify — AF2 预测,含全侧链(重原子)
样例 `absolute-stability-predictor/examples/stability_1998520.pdb`:
```
第一个残基的原子:  N CA C CB O CG2 OG1        (含侧链)
全文件 distinct:    …CB CG CG1 CG2 CD… ND1 ND2 NE… OD1 OD2 OE1 OE2 OG OG1 SD…
```
→ **含完整侧链重原子**。MGnify Stability 训练结构由 **AlphaFold2 预测**(MGnify paper 正文:"predicted aligned error (PAE) matrices from **AlphaFold2** predicted structures … filtered on **AlphaFold2 pLDDT** confidence",行 168-169)。
> 注:`absolute-stability-predictor/mgnify-structure-prediction/` 里另有一套 **ESMFold2** 重预测管线,那是后续/扩展工作;ESM3ΔG **原生训练**用的是 AF2 结构。

---

## B. 消费侧:模型实际吃不吃侧链?——不吃,层层丢弃

侧链坐标从"文件里有"到"进模型"要过两道关,**两道都在丢**:

### B.1 第一道:`parse_CIF` 只保留 37 个**重原子**槽位(丢掉氢)
`esm3dg_stab/esm3dg_model.py:37-51` 的 `atom_types` 只列了 **37 个重原子**(N, CA, C, CB, O, CG…),`atom_order` 里没有任何 H。所以即便 Megascale 文件带氢,`parse_CIF` 也只读重原子进 `atom37[L,37,3]`。
> 此时**侧链重原子还在**(atom37 含 CB/CG/… 槽位),看起来"读进来了"。

### B.2 第二道:ESM3 编码时切成只剩 N/CA/C(丢掉所有侧链)
`atom37` 送进 `ESM3.encode` / `ESM3.forward` 后,被显式切片到前 3 个原子:
- 结构 tokenizer(VQ-VAE):`esm/models/vqvae.py:213` `assert coords.size(-1)==3 and coords.size(-2)==3, "need N, CA, C"`;`:298` `coords = coords[..., :3, :]`
- 几何注意力路径:`esm/models/esm3.py:353-356` `structure_coords = structure_coords[..., :3, :]` → `build_affine3d_from_coordinates`(从 N/CA/C 建残基坐标系)

→ **CB 及之后的所有侧链原子在这一步被彻底丢弃**,连骨架 O 都不要。ESM3(`esm3_sm_open_v1`)架构上就只吃 **N、CA、C** 三个骨架原子。

### B.3 丢弃链路一览
```
PDB/CIF 文件(SKEMPI 全侧链 / AF2 侧链+氢)
        │  parse_CIF: 只留 37 个重原子(丢氢)
        ▼
   atom37[L,37,3]  (侧链重原子还在)
        │  ESM3.encode/forward: coords[..., :3, :]   ← 丢掉 CB…及 O
        ▼
   N, CA, C  (每残基仅 3 个骨架原子)  → structure token + 几何注意力
```

---

## C. 这意味着什么

1. **侧链信息"有但没用"**:三个数据集(含 AF2 预测的 MGnify/Megascale)都带侧链,`parse_CIF` 甚至把侧链重原子读进了 atom37,但 ESM3 在编码时切掉,**一点侧链几何都没进模型**。这是 ESM3 架构决定的,原生训练和本实验一致。
2. **对 ΔΔG 是共同天花板,不是本实验的缺陷**:突变的很多物理效应(位阻、氢键、盐桥)都在侧链几何里,而 backbone-only 模型看不到。但 **ProteinMPNN/StaB-ddG 同样是骨架类模型**(也只用骨架 + 虚拟 CB),所以对"忠实替换"这个目标而言两边对齐,不是我们引入的短板。
3. **ΔΔG 场景下侧链本就"帮不上忙"**:做 `ddG = dG(mut) − dG(wt)` 时,结构(token + 坐标)固定在 **WT**、只换序列(见 [`../esm3dg-forward-mechanism/`](../esm3dg-forward-mechanism/) §6/§7)。即便模型用侧链,那也是 **WT 侧链**、对 mut/wt 完全一样,不携带突变特异信息——除非另做侧链重建(如 Rosetta/AF2 对突变体重折叠),而本 pipeline 和 StaB-ddG 都不做。

---

## 附:代码 / 数据位置

**数据(本 worktree,gitignored)**
- SKEMPI:`data/SKEMPI2_PDBs/*.pdb`(1092,实验结构)
- Megascale:`data/megascale/AlphaFold_model_PDBs/*.pdb`(862,AF2 预测,带氢)
- MGnify 样例:`absolute-stability-predictor/examples/stability_*.pdb`(AF2 预测)

**代码**
- `esm3dg_stab/esm3dg_model.py:37-51` `atom_types`(37 重原子,无氢);`:273` `parse_CIF`
- `esm/models/vqvae.py:213/298`(结构 tokenizer 只取 N/CA/C)
- `esm/models/esm3.py:353-356`(几何路径切 N/CA/C)
- `esm/utils/structure/affine3d.py:519-562`(N/CA/C → 帧)

**MGnify 结构来源**:MGnify paper 正文行 168-169(AF2 pLDDT/PAE 过滤)。

**复现抽查命令**
```bash
# 某结构文件是否含侧链(CB 及以后)
awk '$1=="ATOM"{print $3}' <file.pdb> | grep -qE "^(CB|CG|CD|CE|CZ|NZ|OG|SG|ND|NE|OD|OE|OH|NH)" && echo "含侧链" || echo "仅骨架"
```
