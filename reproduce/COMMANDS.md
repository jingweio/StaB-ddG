# StaB-ddG 复现命令(方案 B:重跑推理 + 指标)

> 目标:用已提供的最终权重 `model_ckpts/stabddg.pt`,在 SKEMPI v2.0 测试集(81 个复合物)上重跑推理,
> 计算论文 Figure 3 的主结果(per-interface Spearman),并与 FoldX / Flex ddG 对比。
> 方法与模块说明见 [`reproduce/MODULES.md`](./MODULES.md)。

本机环境:Ubuntu / conda 25.11 / GPU: TITAN X (12GB, idx0) + RTX A4500 (20GB, idx1)。

---

## 0. 复现结果(本次实际跑出)

| 方法 | Per-interface Spearman(复现) | 论文/作者参考值 |
|---|---|---|
| **StaB-ddG(复现)** | **0.445 ± 0.043** | 0.448 ± 0.039(论文 ≈0.45)|
| FoldX | 0.477 ± 0.034 | ~0.48 |
| Flex ddG | 0.419 ± 0.042 | ~0.42 |

- 复现预测 vs 作者提供预测:全部 1491 条突变 **Pearson = 0.9931**。
- 0.445 vs 0.448 的差异来自 Monte Carlo 随机性(随机解码顺序 + 骨架噪声,ensemble=20),属正常波动。
- 完整指标见 [`reproduce/results_summary.txt`](./results_summary.txt)。

---

## 1. 创建并激活 conda 环境

```bash
cd /home/guoj0f/repos/StaB-ddG
conda env create -f environment.yaml      # python3.10 + torch2.6.0 + biopython 等
conda activate stabddg
# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望:2.6.0+cu124 True
```

> 模型权重(`proteinmpnn.pt` / `stability_finetuned.pt` / `stabddg.pt`)已随 repo 提供,**推理无需训练或额外下载**。

---

## 2. 下载 SKEMPI v2.0 结构文件(唯一需要下载的数据)

标签 CSV(`data/SKEMPI/filtered_skempi.csv`)已在 repo;只需下载结构 PDB。
所有数据都放在 **repo 内** `data/SKEMPI2_PDBs/`,该目录已在 `.gitignore` 中(大文件不随代码提交)。

```bash
cd /home/guoj0f/repos/StaB-ddG
mkdir -p data/SKEMPI2_PDBs
cd data/SKEMPI2_PDBs
wget https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz
tar -xzf SKEMPI2_PDBs.tgz                 # 解压出一个 PDBs/ 子目录
mv PDBs/* . && rmdir PDBs                 # 把 .pdb 直接放到 data/SKEMPI2_PDBs/
cd /home/guoj0f/repos/StaB-ddG
```

可选:确认 81 个测试集复合物结构齐全
```bash
python -c "
import pickle, os
test = pickle.load(open('data/SKEMPI/test_pdb.pkl','rb'))
d = 'data/SKEMPI2_PDBs'
missing = [n.split('_')[0] for n in test if not os.path.exists(os.path.join(d, n.split('_')[0]+'.pdb'))]
print('missing:', missing or 'NONE — all 81 present')
"
```

---

## 3. 在 SKEMPI 测试集上重跑推理

生成预测到 `cache/eval.csv`(默认 `--ensemble 20`,即 20 次 Monte Carlo 平均)。
本机用 GPU1(A4500),约 34 分钟。

```bash
cd /home/guoj0f/repos/StaB-ddG
conda activate stabddg
export CUDA_VISIBLE_DEVICES=1                       # 选 RTX A4500;不设则用默认 GPU0
mkdir -p cache

python skempi_eval.py \
    --skempi_pdb_dir data/SKEMPI2_PDBs \
    --run_name eval
# 默认参数:--checkpoint ./model_ckpts/stabddg.pt  --ensemble 20
#           --skempi_path data/SKEMPI/filtered_skempi.csv
#           --skempi_split_path data/SKEMPI/test_pdb.pkl
# 产出:cache/eval.csv  (列:#Pdb, Mutation, ddG, ddG_pred),81 复合物 / 1491 突变
```

> 首次运行会把结构解析缓存到 `cache/skempi_full_mask_pdb_dict.pkl`,再次运行会直接复用,更快。

---

## 4. 计算指标并与 baseline 对比

```bash
cd /home/guoj0f/repos/StaB-ddG
conda activate stabddg
python reproduce/compute_metrics.py --pred cache/eval.csv
```

该脚本(`reproduce/compute_metrics.py`)调用 `baselines/eval_utils.py: compute_metrics`,输出:
- 复现的 StaB-ddG 指标(per-interface / overall Spearman、Pearson、RMSE、ROC-AUC + cluster-bootstrap 标准误);
- 作者提供的 StaB-ddG / FoldX / Flex ddG 在测试集上的指标(后两者按 `test_pdb.pkl` 过滤);
- 复现预测与作者预测的一致性(Pearson)。

等价地,官方 notebook 也能复现作者数字:
```bash
cd baselines && jupyter nbconvert --to notebook --execute read_results_skempi_test.ipynb
```

---

## 附:其它可选复现范围(本次未执行)

```bash
# C. 自己重跑 SKEMPI 结合微调(阶段2,从 stability_finetuned.pt 出发,需数小时 GPU)
python skempi_finetune.py \
    --train_split_path data/SKEMPI/train_pdb.pkl \
    --run_name my_skempi_ft \
    --checkpoint model_ckpts/stability_finetuned.pt \
    --skempi_pdb_dir data/SKEMPI2_PDBs \
    --skempi_path data/SKEMPI/filtered_skempi.csv
# 产出 cache/skempi_finetuned/my_skempi_ft_final.pt,可替换 --checkpoint 重新评估

# D. 全流程还需先做阶段1(Megascale 折叠微调),需额外从 Zenodo 下载数 GB AlphaFold PDB:
#   https://zenodo.org/records/7992926  ->  AlphaFold_model_PDBs.zip + Processed_K50_dG_datasets.zip
python stability_finetune.py \
    --stability_data <data_dir>/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
    --pdb_dir <data_dir>/AlphaFold_model_PDBs
```

---

## 文件清单(本次新增于 `reproduce/`)

```
reproduce/
├── MODULES.md            # 方法 + 核心模块梳理(含讨论细节)
├── COMMANDS.md           # 本文件:可执行复现命令
├── compute_metrics.py    # 指标计算 + baseline 对比脚本
└── results_summary.txt   # 本次复现的指标输出
```
