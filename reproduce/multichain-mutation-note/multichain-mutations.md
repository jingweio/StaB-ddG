# SKEMPIv2 中的多链同时突变 —— 规模、界面分布与代码处理

> 问题:**模型消费的 SKEMPIv2 样本里,是否存在"同时突变多条链"的情况?**
> 数据源:`data/SKEMPI/filtered_skempi.csv`(4541 行 / 201 个界面)—— 这正是
> `skempi_finetune.py` 与 `skempi_eval.py` 共同读取的那份标签文件。
> 复现脚本:`reproduce/multichain-mutation-note/analyze_multichain_mutations.py`(结果落盘 `results.json`)
>
> **结论一句话:存在,且不罕见 —— 15.9% 的样本涉及 ≥2 条链,其中 537 条(11.8%)
> 跨越界面两侧;test split 上跨界面比例高达 21.6%,是 train 的 3 倍。
> StaB-ddG 的代码路径能正确处理(已实证),但这构成一个 train/test 分布偏移。**

---

## 0. 为什么要区分"多链"和"跨界面"

StaB-ddG 把 binding ddG 分解成三个 folding 项(`stabddg/model.py: binding_ddG`):

```
ddG_bind = ddG_fold(complex) − [ ddG_fold(binder1) + ddG_fold(binder2) ]
```

而 `#Pdb` 里编码了两个 binder 的**链分组**,例如 `1AO7_ABC_DE` 表示
binder1 = 链 ABC、binder2 = 链 DE。于是一条"多链突变"记录有两种截然不同的情形:

| 情形 | 落到哪些 dG 项 | 意义 |
|---|---|---|
| **多链但都在同一个 binder 内** | 只有该 binder 的项 + complex 项 | 与单链突变本质相同 |
| **跨越界面两侧(B1 + B2 都被突变)** | **三个项全部**各承担一部分 | 误差会在三项间叠加,是真正特殊的情形 |

所以下面把两者分开统计。突变 token 格式为 `{WT}{Chain}{Pos}{MUT}`
(如 `YA434A` = 链 A 第 434 位 Tyr→Ala,见 `ppi_dataset.py: mutations_to_seq`)。

---

## 1. 规模:每条记录涉及几条链

| 涉及的不同链数 | 行数 |
|---|---|
| 1 条链 | 3818 |
| **2 条链** | **708** |
| 3 条链 | 5 |
| 4 条链 | 9 |
| **6 条链** | **1** |

**涉及 ≥2 条链:723 / 4541 = 15.9%**

链数最多的样本(`3VR6_ABCDEF_GH`,9 个突变分布在 6 条链上,但都在 binder1 一侧):

```
LA477A,LB482A,LC472A,VA478A,VB483A,VC473A,LD386A,LE386A,LF389A
```

另有 `1QAB_ABCD_E` 的多条 4 链记录(如 `SA85A,SB76A,SC76A,SD76A`)——
这类是**同源多聚体上的对称突变**(同一位点在 4 条等价链上同时突变),
也全部落在 binder1 一侧。

---

## 2. 界面分布:突变落在哪一侧

| 类型 | 行数 | 占全体 |
|---|---|---|
| 只在 binder1 一侧 | 1438 | 31.7% |
| 只在 binder2 一侧 | 2566 | 56.5% |
| **跨越界面两侧(B1B2)** | **537** | **11.8%** |

723 条多链记录中,**537 条是跨界面两侧**,仅 186 条是"同一 binder 内的多链"。
即:**多链突变的绝大多数(74%)确实同时改动了界面的两个 binder。**

**典型例子**(`1A4Y_A_B`,A 侧与 B 侧同时突变):

| 突变 | 链 | ddG |
|---|---|---|
| `RB5A,YA434A` | A+B | −6.98 |
| `RB5A,YA434A,DA435A` | A+B | **−10.10** |
| `KB40G,YA437A` | A+B | −6.45 |

`1AO7_ABC_DE`(TCR/pMHC)也有跨界面例子:`DD26A,EA58A`(链 A+D)、`QD30L,LC1A`(链 C+D)。

**集中度高**:537 条跨界面记录只来自 **34 个复合物**(全体 201 个),头部极集中:

| 复合物 | 跨界面记录数 |
|---|---|
| `1JTG_A_B` | 109 |
| `1KBH_A_B` | 73 |
| `3S9D_A_B` | 59 |
| `1BRS_A_D` | 43 |
| `4G0N_A_B` | 33 |
| `1AO7_ABC_DE` | 26 |
| `1LFD_A_B` | 24 |
| `2VN5_A_B` / `5XCO_A_B` | 各 18 |
| `3HFM_HL_Y` | 16 |

---

## 3. ⚠ train / test 分布偏移(最值得注意的一点)

| split | 行数 | 多链(≥2) | 跨界面两侧 |
|---|---|---|---|
| **train** | 3050 | 351(11.5%) | 215(**7.0%**) |
| **test** | 1491 | 372(**24.9%**) | 322(**21.6%**) |

**test split 的跨界面双侧突变占 21.6%,是 train(7.0%)的 3 倍多。**

这不是随机噪声,而是切分方式的结果:test split 按**界面同源簇 OOD** 划分,
而跨界面突变高度集中在少数复合物(见 §2),这些复合物整簇进了 test。
含义:

- 报告的 per-interface 指标里,`1JTG_A_B`(109 条)、`1KBH_A_B`(73 条)、
  `3S9D_A_B`(59 条)这几个复合物的表现会被"双侧突变"这一特定难度显著影响;
- 模型在训练时见到的双侧突变样本比例远低于测试时,构成一个**分布偏移**。

---

## 4. 代码是否正确处理? —— 实测验证:是 ✅

机制在 `stabddg/ppi_dataset.py:190`,`mutations_to_seq` 为某个结构构造突变序列时
**跳过不属于该结构的链**:

```python
if not mut_chain in chain_offset.keys(): continue
```

因此 binder1 只吃 B1 侧突变、binder2 只吃 B2 侧、complex 吃全部,分解式在热力学上是自洽的:

```
ddG_bind = ddG_fold(complex, 全部突变) − [ ddG_fold(b1, 仅B1侧) + ddG_fold(b2, 仅B2侧) ]
```

用 `1A4Y_A_B` 的跨界面样本 `RB5A,YA434A`(A 侧 + B 侧各一个)实跑数据集,
打印每个结构上**实际被改动的位点**:

```
complex (A+B): [(434, Y→A), (465, R→A)]   ← 两个都改
binder1 (A)  : [(434, Y→A)]               ← 只吃 A 侧
binder2 (B)  : [(5,   R→A)]               ← 只吃 B 侧
长度: b1=460  b2=123  complex=583
```

注意 complex 上的第 465 位 = B 链第 5 位 + A 链长 460,**跨链 offset 计算正确**。

### 两个正面的数据卫生结论

1. **没有任何突变落在 `b1 ∪ b2` 之外的链上**(0 行)。即不存在"突变链不属于界面定义"的脏数据
   —— 若存在,该突变会只进 complex 项而不进任何 binder 项,污染分解式。
2. `mutations_to_seq` 里的 `assert mut_seq[mut_pos] == wt_aa` 在**加载时逐条校验 WT 残基**。
   数据能无异常加载,本身就证明所有跨链 offset 均正确。

---

## 5. 小结与延伸

**回答**:是,存在多链同时突变 —— 15.9% 的样本涉及 ≥2 条链,其中 **11.8%(537 条)
跨越界面两侧**,最多的一条涉及 6 条链。代码路径处理正确,数据无越界脏数据。

**但有一个需要留意的事实**:这类样本在 **test split 上占 21.6%,是 train 的 3 倍**,
且集中在 10 个左右的复合物上。

**延伸建议(尚未做)**:按"单链 vs 跨界面双侧"做一次**分层评估**,看两组的
per-interface Spearman 是否有显著差异 —— 双侧突变要求模型同时正确处理三个 dG 项、
误差会叠加,预期更难。做这个分层时**必须用 THRESHOLD=10 口径**(仅计入 ≥10 突变的复合物)
以对齐 StaB 论文基线,否则数字不可比。

---

## 复现

```bash
conda activate stabddg
python reproduce/multichain-mutation-note/analyze_multichain_mutations.py
# 追加 --verify 同时跑 per-binder 拆分的实证检查(需 PDB 文件)
python reproduce/multichain-mutation-note/analyze_multichain_mutations.py --verify
# 输出: results.json(全部统计量)
```

**相关文件**
- 标签数据:`data/SKEMPI/filtered_skempi.csv`;切分:`data/SKEMPI/{train,test}_pdb.pkl`
- 突变→序列:`stabddg/ppi_dataset.py: mutations_to_seq`(第 169–205 行,跳过逻辑在 190)
- binding 分解:`stabddg/model.py: binding_ddG`(第 55–65 行)
- 符号约定见 [`../dG-ddG-sign-note/dG-ddG-sign-conventions.md`](../dG-ddG-sign-note/dG-ddG-sign-conventions.md)
  (本文 ddG 沿用 filtered_skempi 口径:**stabilizing = 正**)
