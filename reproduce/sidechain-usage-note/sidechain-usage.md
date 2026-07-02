# StaB-ddG 吃不吃 side-chain?—— 结构输入梳理与验证

> 目的:讲清楚两件**互相独立**的事:(1) Megascale / SKEMPIv2 的结构文件里**有没有** side-chain;
> (2) StaB-ddG 的 ProteinMPNN 实际**用不用** side-chain。所有结论附代码位置 + 实测原子名,可复核。
>
> 结论一句话:**结构文件都是全原子含 side-chain 的,但 StaB-ddG 的 ProteinMPNN 只读 N/CA/C/O 骨架
> (+ 一个从骨架算出来的虚拟 Cβ),side-chain 在 parse 阶段就被丢掉了。**

---

## 0. TL;DR

| 问题 | 答案 |
|---|---|
| 结构文件里有 side-chain 吗? | **有**(Megascale AF 结构甚至含氢;SKEMPI 实验结构含全套 side-chain 重原子) |
| ProteinMPNN 吃 side-chain 吗? | **不吃**,只吃 backbone 4 原子 N/CA/C/O + 虚拟 Cβ |
| SKEMPIv2 用的是 AF2 预测结构吗? | **不是**,是实验晶体结构;只有 Megascale 用 AlphaFold 预测 |
| 突变怎么进模型? | 只改喂给 log-likelihood 的**序列 one-hot**;**backbone 一个原子都不动**,WT 与 mut 共用同一结构 |

---

## 1. ProteinMPNN 只吃 backbone(N, CA, C, O + 虚拟 Cβ)

**只用 4 个 backbone 重原子,side-chain 完全丢弃。** 这是 ProteinMPNN 作为 **inverse-folding 模型**的本质:
它要"给定 backbone 预测序列",所以**绝不能看 side-chain**(否则等于泄漏了要预测的氨基酸身份)。

代码证据(`stabddg/mpnn_utils.py`):
- `parse_PDB` 里变量名 `sidechain_atoms` 是 ProteinMPNN 原始的**误导性命名**——它实际取的是
  `['N','CA','C','O']`(第186行);`ca_only` 模式才是 `['CA']`(第184行):
  ```python
  # mpnn_utils.py:184-187
  if ca_only:
      sidechain_atoms = ['CA']
  else:
      sidechain_atoms = ['N', 'CA', 'C', 'O']       # ← 名字叫 sidechain,实际是 backbone 4 原子
  xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
  ```
- 坐标只存 N/CA/C/O(第195-198行 `N_chain_/CA_chain_/C_chain_/O_chain_`)。
- featurize 时只把 N/CA/C/O 堆成 `[L, 4, 3]`(第260、275行:
  `np.stack([... N_chain, CA_chain, C_chain, O_chain ...], 1)`)。
- `ProteinFeatures.forward`(第532-545行)从 N、CA、C **算一个虚拟 Cβ**,再用这些骨架原子之间的距离做
  RBF 特征;`X[:,:,0/1/2/3,:]` 分别是 N/CA/C/O。

→ 哪怕输入 PDB 是全原子的,**parse 阶段 side-chain(和氢)全部被扔掉**。

---

## 2. 结构文件里确实含 side-chain(但注意 SKEMPI 不是 AF2)

实测各数据集 PDB 里的原子名:

| 数据集 | 结构来源 | 含 side-chain? | 实测样例 |
|---|---|---|---|
| **Megascale** | AlphaFold 预测(`AlphaFold_model_PDBs`) | **有,全原子含氢** | `EEHEE_rd3_0602.pdb`:CB/CG/CD/CE/CZ/ND/NE/NZ/OD/OE + 全套 H;699 原子 / 43 残基 ≈ **16 atoms/res** |
| **SKEMPIv2** | **实验晶体结构**(life.bsc.es/pid/skempi2) | **有**(重原子;X-ray 一般无 H) | `1A22_A.pdb`:CB×174、CG×120、CD1×55、CD×53、CD2×49、CZ×31、OE1×27… 全套 side-chain 重原子 |
| example(run_stabddg 演示) | 实验结构 | **有** | `examples/one_mutation/1AO7.pdb`:CB/CG/CD/CE/CZ/OD/OE/SD/SG/NH/NZ… 全套 |

**⚠ 纠正一个常见误解:SKEMPIv2 用的不是 AF2 预测结构,而是实验晶体结构**
(从 SKEMPI2 数据库下的 `SKEMPI2_PDBs.tgz`)。只有 **Megascale** 用的是 AlphaFold 预测结构。
- SKEMPI 里唯一和 AF 沾边的是一个**可选** flag `--af_apo_structures`(`stabddg/ppi_dataset.py:75-80, 94-98`):
  开启后**只把两个单体(binder)的 apo/未结合结构**换成 AlphaFold 预测的(文件名带 `_AF` 后缀);
  **复合物本身仍是实验结构**。默认关闭。

---

## 3. 关键含义:文件有 side-chain,但方法不用

因为 ProteinMPNN 只吃 backbone,所以在 StaB-ddG 里:

1. **WT 和 mutant 共用完全相同的 backbone 坐标**(就是那个参考 / WT 结构)。突变**只体现在喂给
   log-likelihood readout 的序列 one-hot 变了**(见 `folding_dG`:`dG = Σ one_hot(seq) · log P`),
   **结构一个原子都没动**。
2. 也就是说 **StaB-ddG 不显式建模突变引起的 side-chain 重排 / 骨架松弛 / 空间位阻**;它纯粹靠
   "这个突变氨基酸身份在这个固定 backbone 环境下有多'合理'(P(aa | backbone))"来推 ddG。
3. 这正是它**比 FoldX / FlexddG 快得多**(无需 repack / relax)的原因,但也是它的**软肋**:
   凡是必须靠 side-chain 构象变化才能解释的效应,它原理上看不到。
4. 推理时加的那点 backbone noise(0.1–0.2 Å 高斯,`model.py:74-76`)只是**方差缩减 / ensembling**
   用的扰动,**不是**在建模突变导致的构象变化。

---

## 附录:验证命令与位置

**代码位置**
- `stabddg/mpnn_utils.py:184-187`(atoms 列表 = N/CA/C/O);`:195-198`(存 N/CA/C/O 坐标);
  `:260, :275`(stack 成 [L,4,3]);`:532-545`(`ProteinFeatures`,从 N/CA/C 算虚拟 Cβ)
- `stabddg/model.py:18-36`(`folding_dG`:突变只改 `seqs` one-hot);`:74-76`(backbone noise)
- `stabddg/ppi_dataset.py:75-80, 94-98`(`af_apo_structures` 可选项,仅换 apo binder 为 AF 结构)

**数据位置**
- SKEMPI 实验结构(本地):`data/SKEMPI2_PDBs/*.pdb`(如 `1A22_A.pdb`)
- example:`examples/one_mutation/1AO7.pdb`
- Megascale AF 结构:不在 repo(外部下载)。当前在 Ibex:
  `/ibex/user/guoj0f/StaB-ddG/esm-replace/data/AlphaFold_model_PDBs/*.pdb`(如 `EEHEE_rd3_0602.pdb`)

**复现验证(看某 PDB 里有没有 side-chain 重原子)**
```bash
# 列出非骨架原子名(有输出 = 含 side-chain)
grep "^ATOM" <file.pdb> \
  | awk '{a=substr($0,13,4); gsub(/ /,"",a); print a}' \
  | grep -vE "^(N|CA|C|O|OXT)$" | sort -u
```

**复现验证(确认 StaB-ddG 只 parse N/CA/C/O)**
```bash
grep -n "sidechain_atoms\|N_chain_\|CA_chain_\|C_chain_\|O_chain_\|np.stack" stabddg/mpnn_utils.py
```
