# StaB-ddG 复现笔记:核心模块与实验流程

> 论文:*Predicting mutational effects on protein binding from folding energy* (Deng et al., ICML 2025, arXiv:2507.05502)
> 本文件梳理 repo 的核心模块、方法与代码的对应关系,以及完整实验流程,供 review 使用。
> 配套的可执行复现命令见 [`reproduce/COMMANDS.md`](./COMMANDS.md)。

---

## 0. 一句话概括

StaB-ddG 预测突变对**蛋白-蛋白结合自由能**的影响 ΔΔG_bind。核心创新是用**热力学恒等式**把"结合能"重写成"折叠能的差",从而能借用海量的**折叠稳定性数据**来弥补结合能数据稀缺的问题;底层用预训练的反折叠模型 **ProteinMPNN** 的序列对数似然作为折叠能代理,经**两阶段迁移微调**(先折叠、后结合)+ **方差缩减**得到最终模型。它是首个在 SKEMPIv2.0 基准上匹配 FoldX / Flex ddG 精度的深度学习方法,且推理快 ~1000×。

---

## 1. 核心思想(理论桥梁)

### 1.1 热力学恒等式(Eq.1, Fig.1a)
自由能是状态函数,结合能可沿"未折叠 → 折叠 → 结合"路径分解:

```
ΔG_bind(A:B) = ΔG_fold(A:B) − ΔG_fold(A) − ΔG_fold(B)
```

对突变 s → s′ 求差,得到预测目标(Eq.2):

```
ΔΔG_bind = ΔΔG_fold(A:B → A′:B) − ΔΔG_fold(A → A′)
```

**意义**:结合能数据极少(SKEMPI ~7,085 条、345 个复合物),折叠数据多两个数量级(Megascale 776,298 条)。把结合写成折叠的差,就能用折叠数据训练 → 这是整篇工作的核心 contribution。

### 1.2 用 ProteinMPNN 参数化折叠能(§3.1)

| 论文公式 | 含义 | 代码位置 |
|---|---|---|
| Eq.3 `f_θ(s) = log p_θ(s)` | 反折叠模型的序列对数似然 ≈ 折叠能代理 | `stabddg/model.py: folding_dG()` |
| Eq.4 `b_θ(s_A:B) = f_θ(s_A:B) − f_θ(s_A) − f_θ(s_B)` | **StaB 参数化**:复合物似然 − 两个单体似然 | `stabddg/model.py: binding_ddG()` |
| Eq.5 `Δb_θ(s,s′) = b_θ(s′) − b_θ(s)` | 最终 ΔΔG_bind 预测器 | `stabddg/model.py: forward()` |

一次 ΔΔG_bind 预测需要算最多 **6 次** f_θ(WT/Mut × 复合物/binder1/binder2)。论文证明该参数化天然满足 ΔΔG 预测器应有的三个性质——**反对称性、突变路径无关性、表达能力**(Table 1),FoldX 等方法不全具备。

**为什么用 ΔΔ 而非绝对 ΔG**:`log p_θ` 单位是对数概率(非 kcal/mol),且每个蛋白有未知的加性基线偏移;取差后偏移自动抵消,且与下游 ΔΔG_bind(本身是差)一致。

---

## 2. 两阶段顺序迁移微调(§3.2)— 训练主线

三个 checkpoint 一一对应,**均已包含在 `model_ckpts/`,推理无需训练**:

```
proteinmpnn.pt            ProteinMPNN 原始权重 (soluble, v_48_020)
      │  阶段1:Megascale 折叠稳定性微调(Eq.7,MSE on ΔΔG_fold)
      │         lr=3e-5, 70 epochs, batch=25000 aa, ~10h H100
      ▼
stability_finetuned.pt    论文 "Stability fine-tuned"
      │  阶段2:SKEMPI 结合能微调(Eq.8,MSE on ΔΔG_bind)
      │         lr=1e-6, 200 epochs, batch=25000 aa, ~5h H100
      ▼
stabddg.pt                最终模型(推理用)
```

### 关键概念澄清(review 重点)
- **Megascale** = Tsuboyama et al. 2023 (Nature) 的大规模**单体蛋白折叠稳定性**数据集:412 个参考蛋白 + 各自大量**突变体**,共 776,298 条 ΔG_fold 测量。**数据集里是有突变的**。
- **阶段1 目标**:用折叠 ΔΔG 标签有监督微调,让 ProteinMPNN 的 likelihood 成为更准的折叠能 proxy。借鉴了 Dieckhaus/ThermoMPNN 的数据准备协议;它是"核心迁移策略的第一步",但概念原创点在 §1 的参数化,而非微调动作本身。
- **阶段2 目标**:从 `stability_finetuned.pt` 出发,在真实结合能上微调得到最终 SOTA 模型 `stabddg.pt`。这一步调用完整的 `binding_ddG`,最能体现核心设计。
- **微调对象**:两阶段微调的**都是 ProteinMPNN 的全部权重**。`StaBddG` 类本身**没有任何额外参数**(纯可微计算图,无回归头),`optimizer = Adam(model.parameters())` 等价于只训练 ProteinMPNN,保存的也是 `model.pmpnn.state_dict()`。论文强调 "despite not introducing additional parameters to ProteinMPNN"。
- `model.py` 里的 `LinearModel` 是 Appendix D 的消融(加氨基酸特异线性偏移修正尺度失配),**结果无显著效果,主流程不用**。

---

## 3. 方差缩减(§3.3)— 推理精度关键

ProteinMPNN 的随机性来自**随机解码顺序**和**骨架高斯噪声**,引入预测方差。两招压低:

1. **对偶变量 (Antithetic variates)**:WT 与 Mut 共用同一解码顺序(ε′=ε)和同一份骨架噪声,使两者正相关 → `Var[b_θ(s′) − b_θ(s)]` 下降。
   - 代码:`model.py: folding_ddG()` 先 `_get_decoding_order()` / `_get_backbone_noise()` 采样一次,再同时传给 wt/mut 的 `folding_dG()`(`use_antithetic_variates=True`)。
2. **Monte Carlo 平均**:对 M=20 个独立采样求平均,方差降为 1/M。
   - 代码:`skempi_eval.py` / `run_stabddg.py` 外层 `ensemble=20` 循环取 `.mean()`。

论文 Fig.2 消融:ProteinMPNN 零样本 → +方差缩减 → +StaB参数化 → +折叠微调,逐级提升。

---

## 4. 代码模块地图

### 4.1 `stabddg/`(核心库)
| 文件 | 作用 |
|---|---|
| `model.py` | **核心**。`StaBddG` 类:`folding_dG`(Eq.3)、`folding_ddG`(Eq.6,含对偶变量)、`binding_ddG`(Eq.4/5,forward)。`LinearModel`(消融,未用)。 |
| `mpnn_utils.py` | ProteinMPNN 网络实现 + 工具:`parse_PDB`(读结构)、`featurize`(结构→张量)、`StructureDataset`、`ProteinMPNN`(`forward` 支持 `fix_order`/`fix_backbone_noise` 以实现对偶变量)、`EncLayer/DecLayer/ProteinFeatures` 等。改自 ProteinMPNN / Graph-based protein design。 |
| `ppi_dataset.py` | 数据管线。`PPIDataset` 基类:读 PDB、用 `extract_chains` 把复合物拆成两个 binder 的 PDB、把突变字符串转成序列索引矩阵、缓存结构字典。`SKEMPIDataset` / `YeastDataset` 子类负责各自的 csv 解析。 |
| `utils.py` | `extract_chains`(按链拆 PDB,基于 BioPython)、`renumber_pdb`(把残基编号从 1 重新编号——模型假设每条链从 1 开始)。 |

### 4.2 训练 / 推理脚本(repo 根目录)
| 文件 | 作用 | 对应 |
|---|---|---|
| `stability_finetune.py` | 阶段1:Megascale 折叠微调 | §2 阶段1, Eq.7 |
| `skempi_finetune.py` | 阶段2:SKEMPI 结合微调 | §2 阶段2, Eq.8 |
| `skempi_eval.py` | **复现主结果**:在 SKEMPI 测试集推理,输出 `cache/eval.csv` | §5.2, Fig.3 |
| `run_stabddg.py` | 单个/批量预测的用户接口(单突变或 csv 列表) | README 示例 |
| `setup.py` | 把 `stabddg` 装成包 | — |

### 4.3 `data/`
| 路径 | 内容 |
|---|---|
| `SKEMPI/skempi_v2.csv` | 原始 SKEMPIv2.0 标签表 |
| `SKEMPI/filtered_skempi.csv` | 质量过滤后(`quality_filtering.ipynb` 产出),含 cluster 列 |
| `SKEMPI/{train,test}_pdb.pkl` | 基于界面同源聚类的 train(120)/test(81) 划分 |
| `SKEMPI/{train,test}_clusters.{pkl,txt}` | 聚类划分 |
| `SKEMPI/skempi_splits.ipynb` | 划分脚本(界面同源聚类) |
| `rocklin/mega_splits.pkl` | Megascale 的 train/val/test 划分(来自 ThermoMPNN) |
| `TCRm_case_study/labels.csv` | TCR mimic case study 标签 |

### 4.4 `baselines/`(指标与对比)
| 文件 | 内容 |
|---|---|
| `eval_utils.py` | `compute_metrics`(Spearman/Pearson/RMSE/MAE/AUROC + 自助法标准误)、`struct_metrics`(per-interface,≥10 突变才算)、`t_test`(配对 t 检验)、`latex_table_format`。 |
| `read_results_skempi_test.ipynb` | 读预测 csv,算指标,复现论文数字。 |
| `StaB-ddG.csv` / `foldx.csv` / `flexddg.csv` | 作者提供的 StaB-ddG / FoldX / Flex ddG 预测,用于对比(后两者含 train+test,需按 `test_pdb.pkl` 过滤)。 |

### 4.5 `model_ckpts/`
见 §2,三个权重均已直接提交。

---

## 5. 评估指标说明(`eval_utils.py: compute_metrics`)

- **Per-interface(per-structure)指标**:对每个复合物分别算 Spearman/Pearson/RMSE,再跨复合物取平均;只统计**突变数 ≥ 10**(`THRESHOLD=10`)的复合物(否则方差太大)。论文主图 Fig.3 报告的就是 **per-interface Spearman**。
- **Overall 指标**:把所有突变混在一起算一个相关系数。
- **标准误**:用 **cluster bootstrap**(对复合物有放回重采样,`BOOTSTRAP=300` 次)估计。
- **二分类**:ΔΔG<0 视为去稳定突变,算 Precision/Recall/ROC-AUC/PR-AUC。
- **显著性**:`t_test` 做配对 t 检验比较两个方法的 per-interface 指标。

论文主结果:StaB-ddG **per-interface Spearman ≈ 0.45**(Fig.3),匹配 FoldX(~0.48)/Flex ddG(~0.42),显著优于其它 DL 方法;推理 0.2 s/突变 vs FoldX 210 s/突变(~1000×)。

---

## 6. 复现范围阶梯

| 范围 | 需下载 | GPU 耗时 | 对应论文 |
|---|---|---|---|
| A 仅指标计算 | 无(用自带 csv) | 0 | 跑 notebook 复现 Fig.3 数字 |
| **B 重跑推理 + 指标(本次执行)** | SKEMPI2_PDBs(~30MB) | 几十分钟 | 用 `stabddg.pt` 跑测试集,复现 Fig.3 主结果 |
| C + SKEMPI 微调 | SKEMPI2_PDBs | +数小时 | 阶段2,重训得 stabddg.pt |
| D 全流程 | + Megascale AlphaFold PDBs(数 GB) | +10h 量级 | 阶段1+2,从 ProteinMPNN 起 |

> 本次复现执行 **方案 B**。具体命令与产出见 [`reproduce/COMMANDS.md`](./COMMANDS.md)。

---

## 7. 环境与数据位置(本机)

- Conda 环境:`stabddg`(由 `environment.yaml` 创建;python 3.10 + torch 2.6.0 + biopython 等)。
- SKEMPI2 结构数据:`data/SKEMPI2_PDBs/`(repo 内,已在 .gitignore 中,大文件不随代码提交)。
- GPU:NVIDIA TITAN X (12GB) / RTX A4500 (20GB)。
