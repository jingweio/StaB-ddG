# StaB-ddG 的 dG / ddG 符号约定 —— 梳理与验证

> 目的:讲清楚 StaB-ddG 里 `dG` / `ddG` 到底是什么、符号朝向和"平时的物理约定"哪里一样、哪里相反,
> 以及训练 / 评估 / 对外输出各用的是哪套约定。所有结论都附了**代码位置 + 实测数据**,可复核。
> 结论一句话:**你的物理直觉全对,而且和 StaB-ddG 最终对外输出一致;但模型内部把 dG 当 log-likelihood
> (越大越稳定)、训练/eval 的 ddG 标签又是 stabilizing=正,这两层都和"平时"相反。**

---

## 0. TL;DR:一张符号表

| 量 | destabilizing | **stabilizing** | 说明 |
|---|---|---|---|
| 经典物理 ΔΔG = ΔG(mut) − ΔG(wt) | > 0 | **< 0** | 通常物理约定(也是最终对外约定) |
| filtered_skempi.csv `ddG` 标签 | < 0 | **> 0** | = −(ΔG_mut−ΔG_wt),训练/eval 目标 |
| Megascale `ddG_ML` 标签 | < 0 | **> 0** | Tsuboyama 原生朝向,直接用 |
| StaB 内部 `dG`(= Σ log P(seq\|struct)) | 更低 | **更高(接近 0)** | log-likelihood,≤0,像 −(energy) |
| StaB 内部 `ddG`(模型裸输出 = 训练目标) | < 0 | **> 0** | 和两个标签同朝向 |
| `skempi_eval.py` 输出 `ddG_pred`(**不翻**) | < 0 | **> 0** | 和标签同朝向,好算指标 |
| `run_stabddg.py` 输出(`*= -1` **之后**) | > 0 | **< 0** | 扳回经典物理约定 |

**要点:训练/eval 全程 stabilizing=正,只有生产脚本 `run_stabddg.py` 在最后 `*= -1` 扳回物理约定。**

---

## 1. ProteinMPNN 的输出头是怎么变成 "dG head" 的

**关键:没有加任何新的 regression head,也没有新参数。** `stabddg/model.py:78` 那个 `LinearModel` 类
**定义了但从未被实例化/使用**(全仓库 grep 只有类定义,无调用),是遗留代码。

真正的 "dG head" 就是 ProteinMPNN **原本的 per-residue 21 类氨基酸 softmax 输出头**,只是换了读取方式:

```python
# stabddg/model.py:18-36  folding_dG
log_probs = self.pmpnn(...)                 # [B, L, 21] 每个位点 log P(aa | 结构, context)
seq_oh    = one_hot(seqs, 21)
dG = torch.sum(seq_oh * log_probs, dim=(1,2))   # = Σ_i log P(s_i | 结构)
```

即 **`dG` := 这条序列在该 backbone 上的 (pseudo-)log-likelihood** = Σ log P(seq | structure)。
因为是 log 概率之和,`dG ≤ 0`;序列越贴合折叠 → 概率越高 → `dG` 越接近 0(越大)→ 越稳定。

配套三件事:
- **Binding 分解**(`stabddg/model.py:55-65`):
  `ddG_bind = ddG_fold(complex) − [ ddG_fold(binder1) + ddG_fold(binder2) ]`
  用**同一个** `folding_dG` 机器分别算复合物和两个单体,所以连"binding head"都不用单独造。
- **两阶段 fine-tune**:先 Megascale 单体折叠稳定性(`stability_finetune.py`),再 SKEMPI binding
  (`skempi_finetune.py`)。整个 ProteinMPNN 端到端 fine-tune,极小 lr(~1e-6)。
- **方差缩减**:antithetic variates(WT 与 mutant 共享同一随机 decoding order + 同一份 backbone noise,
  `model.py:45-49, 70-76`)+ Monte Carlo ensembling(默认 20 次取平均)。

`folding_ddG`(`model.py:38-53`): `ddG = mut_dG − wt_dG`。
- WT 原生序列在自己 backbone 上概率高 → `wt_dG` 较高(接近 0);
- destabilizing 突变让某位点更不贴合局部结构 → 概率降低 → `mut_dG` 更负 → `ddG = mut_dG − wt_dG < 0`。
- 所以**内部 ddG:destabilizing < 0,stabilizing > 0**。

---

## 2. 三套符号的来龙去脉

### 2.1 内部 `dG` 是 log-likelihood,方向和物理 ΔG 相反
物理 ΔG_fold = G_folded − G_unfolded,稳定→**负**,越负越稳定。
StaB 内部 `dG` = Σ log P,稳定→**更高(更接近 0)**。两者单调方向相反。所以
"dG 越小越稳定"对**物理 ΔG_fold** 成立,对 **StaB 内部 dG 恰好相反(越大越稳定)**。

### 2.2 训练/eval 的 ddG 标签都用 "stabilizing = 正"
这样模型可以用**一套符号**同时在 Megascale + SKEMPI 上训练。两个数据集**原生朝向不同**,处理方式不同:

| | 原生朝向 | 要不要翻到 stabilizing=正 | 翻转在哪 |
|---|---|---|---|
| **Megascale** | `ddG_ML` 本来就是 stabilizing=正(Tsuboyama 把 `dG_ML` 定义成"越大越稳定") | **不用翻** | 无 |
| **SKEMPI** | 从亲和力算的 ΔΔG = `dG_mut−dG_wt` 是 stabilizing=负(物理) | **要翻** | `quality_filtering.ipynb` 里显式 `ddG = -(dG_mut - dG_wt)` |

> 注意:Megascale 端 **StaB 代码没有任何翻转动作**,`ddG_ML` 进来就是对的朝向;
> 它相对"物理 ΔG_fold"确实是翻过的,但那是 **Tsuboyama 数据集自己 ΔG 定义**里带的,不是 StaB pipeline 的一步。
> (另:`ppi_dataset.py:279` 的 `YeastDataset` 有 `row['ddG'] = -row['ddG']`,因为 Yeast 源朝向相反,
> 翻一下是为了对齐到同一朝向;SKEMPI/Megascale 的 loader 都不翻。)

### 2.3 全仓库唯一的翻转:`run_stabddg.py`
```python
# run_stabddg.py:36-37
binding_ddG_pred = torch.stack(binding_ddG_pred_ensemble).mean(dim=0)  # 模型算完+ensemble平均
binding_ddG_pred *= -1   # Convert to convention of negative values for stabilizing mutations
```
- 发生在**模型之外**、ensemble 平均之后、写 CSV 之前 → **不进 loss、不进梯度、不影响任何学到的参数**。
- 它把"内部/标签约定(stabilizing=正)"翻成"经典物理约定(stabilizing=负)"仅用于**对外输出**。
- `skempi_eval.py` **不翻**:算指标时 pred 和 label 必须同朝向。

---

## 3. 证据链(全部实测,可复核)

### 3.1 SKEMPI:`ddG = −(dG_mut − dG_wt) = dG_wt − dG_mut`
定义来自 `data/SKEMPI/quality_filtering.ipynb`:
```
dG_wt  = (8.314/4184) * Temperature * ln(Affinity_wt_parsed)     # = RT·ln(Kd), 负,越负结合越紧
dG_mut = (8.314/4184) * Temperature * ln(Affinity_mut_parsed)
ddG    = -(dG_mut - dG_wt)                                        # 显式负号 = 相对物理翻转
```
在 `data/SKEMPI/filtered_skempi.csv`(4541 行)上验证:
- `corr(ddG, dG_wt − dG_mut) = 1.0`,`max|ddG − (dG_wt − dG_mut)| ≈ 7e-15` → 公式确认。
- `dG_wt` 均值 −11.94,`dG_mut` 均值 −10.40(突变平均削弱结合,符合"多数突变 destabilizing")。
- 按结合强弱分组(Kd_mut>Kd_wt 记为 destabilizing,占 80.2%):
  - **destabilizing 突变 `ddG` 均值 = −2.13(负)**
  - **stabilizing 突变 `ddG` 均值 = +0.89(正)**
  → filtered_skempi 的 `ddG`:**stabilizing=正,destabilizing=负**。

### 3.2 Megascale:`ddG_ML` 原生就是 stabilizing=正
原始 CSV(666M,外部数据,当前在 Ibex:
`/ibex/user/guoj0f/StaB-ddG/esm-replace/data/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`)
含 `Stabilizing_mut` ground-truth 列。实测:
```
Stabilizing_mut = True  (stabilizing)  → ddG_ML 均值 = +1.49  (正)   n=2389
Stabilizing_mut = False (destabilizing)→ ddG_ML 均值 = −1.32  (负)   n=530882
mut_type == wt                         → ddG_ML ≈ +0.004 ≈ 0         (自洽性检查✓)
dG_ML (每个变体折叠稳定性)             → 均值 +1.76, 绝大多数为正   (越大越稳定)
```
→ Tsuboyama 的 `ddG_ML`:**stabilizing=正,destabilizing=负**;`dG_ML` 用"stability/解折叠"符号(越大越稳定)。
`stability_finetune.py:196-207` 直接 `mut_df['ddG_ML']` 原样读入,**无任何符号操作**。

### 3.3 codebase 里对 ddG 的符号操作
grep 全仓库,对 ddG 起作用的符号翻转只有两处,且都不在训练/eval 主路径的模型计算里:
- `run_stabddg.py:37` `binding_ddG_pred *= -1`(对外输出)
- `stabddg/ppi_dataset.py:279` `row['ddG'] = -row['ddG']`(仅 `YeastDataset`,对齐朝向)

训练/eval 用裸 MSE 直接比对(`skempi_finetune.py:124`、`stability_finetune.py:88`、`skempi_eval.py`),**无翻转**。

### 3.4 eval 输出与标签同朝向(交叉验证)
`baselines/StaB-ddG.csv`(`skempi_eval.py` 产物,不翻):
- `corr(ddG_pred, label ddG) = +0.53`(同向 → 正相关)
- label<0(destabilizing)的 pred 均值 −1.16;label>0(stabilizing)的 pred 均值 −0.16
  → stabilizing 的预测更高 → 和标签同朝向(stabilizing=正)。确认 eval 不翻、且内部裸输出 stabilizing=正。

---

## 4. ⚠ 一个文档 bug:README 与代码矛盾

- **README 第 23 行**:"a negative value (ΔΔG < 0) represents a **destabilizing** mutation"
- **`run_stabddg.py:37` 注释**:"negative values for **stabilizing** mutations"

两者互相矛盾。**以代码为准**:裸输出 stabilizing=正,`*=-1` 之后 stabilizing=负 → `run_stabddg.py`
实际输出 **负 = stabilizing**(= 经典物理约定,= 代码注释)。**README 第 23 行写反了**,别信那句。

---

## 5. 复现实操 takeaways

1. **自己训练 / 算 loss / 算 Spearman·Pearson·RMSE** → 用裸输出,**不要翻转**,和标签(stabilizing=正)对齐。
   - 一致性才是关键:对 pred 和 label **同时**全局翻转,三种指标数值都不变;
     但**只翻一边**(如只翻 pred)会让相关系数变号、RMSE 直接炸。
2. **只想给用户一个物理约定(负=更稳定)的 ddG 数值** → 才 `*= -1`(照 `run_stabddg.py`)。
3. **拿到一个预测 CSV 先看是哪个脚本产的:**
   - `skempi_eval.py` 的 `ddG_pred` → stabilizing=正(负=destabilizing)
   - `run_stabddg.py` 的 `Prediction`/`pred_1` → stabilizing=负(负=stabilizing)
   - 两者差一个负号,**别混用**。
4. **标签朝向**:filtered_skempi `ddG` 与 Megascale `ddG_ML` **都是 stabilizing=正**,可直接混合训练。

---

## 附录:关键位置与复现命令

**代码位置**
- `stabddg/model.py:18-36` `folding_dG`(dG = Σ log P);`:38-53` `folding_ddG`;`:55-65` `binding_ddG`;`:78-85` `LinearModel`(未使用)
- `run_stabddg.py:36-37`(唯一对外翻转);`skempi_eval.py`(不翻);`skempi_finetune.py:124` / `stability_finetune.py:88`(裸 MSE)
- `stability_finetune.py:196-207`(`ddG_ML` 原样读入);`stabddg/ppi_dataset.py:232-249`(SKEMPI 原样读 `ddG`);`:268-283`(Yeast 翻转)

**数据位置**
- SKEMPI:`data/SKEMPI/filtered_skempi.csv`(含 `dG_wt/dG_mut/ddG`);生成脚本 `data/SKEMPI/quality_filtering.ipynb`
- Megascale:原始 CSV 不在 repo(外部下载)。当前在 Ibex:
  `/ibex/user/guoj0f/StaB-ddG/esm-replace/data/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`
  repo 内只有 split 名单 `data/rocklin/mega_splits.pkl`

**复现验证命令(SKEMPI,本地)**
```python
import pandas as pd, numpy as np
df = pd.read_csv('data/SKEMPI/filtered_skempi.csv')
print(np.corrcoef(df['ddG'], df['dG_wt']-df['dG_mut'])[0,1])           # ≈ 1.0
weaker = df['Affinity_mut_parsed'] > df['Affinity_wt_parsed']          # destabilizing
print(df.loc[weaker,'ddG'].mean(), df.loc[~weaker,'ddG'].mean())       # −2.13 / +0.89
```

**复现验证命令(Megascale,Ibex)**
```python
import pandas as pd
F='/ibex/user/guoj0f/StaB-ddG/esm-replace/data/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'
df = pd.read_csv(F, usecols=['mut_type','dG_ML','ddG_ML','Stabilizing_mut'], low_memory=False)
df = df[df['ddG_ML']!='-']; df['ddG_ML']=pd.to_numeric(df['ddG_ML'],errors='coerce')
print(df.groupby('Stabilizing_mut')['ddG_ML'].mean())                 # True +1.49 / False −1.32
```
