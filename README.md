# BTRI-Kcat

BTRI-Kcat 是一个酶催化速率预测项目。模型结合蛋白质序列特征、RXNFP 反应指纹和 ChemBERTa 反应特征，预测 `log10(kcat)`（`kcat` 单位：s⁻¹）。核心模型位于 [`model/BTRIKcat.py`](model/BTRIKcat.py)，训练入口分别覆盖 CatPred-DB、PMAK-DB 和 TurnUp 冷启动划分。

## 目录结构

```text
.
├── data/                 # 原始 CSV 数据及来源说明
├── model/BTRIKcat.py     # 模型和消融变体
├── preprocess/           # CSV 到 token 特征的预处理
├── train/                # 三组数据的训练与评估入口
├── scripts/              # 结果汇总和数据审计工具
├── installment.txt       # Python 依赖
└── README.md
```

以下命令均在本目录（包含 `installment.txt` 的目录）执行。

## 环境安装

建议使用 Linux、Python 3.9，并根据本机设备先安装 PyTorch 2.8.0 的 CPU 或 CUDA 构建。以下是 CPU 安装示例；GPU 用户应改用与驱动兼容的 PyTorch 构建。

```bash
conda create -n btrikcat python=3.9 -y
conda activate btrikcat
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r installment.txt
python -m pip install --no-deps rxnfp==0.1.0
```

RXNFP 的旧依赖约束与本项目的其他依赖版本冲突，因此单独使用 `--no-deps` 安装。首次提取特征时，ESM2、ChemBERTa 等编码器还需要下载预训练权重；也可以通过预处理参数指定本地模型目录。默认 ESM2-3B 的计算和显存开销较高。

## 从数据生成特征

仓库附带 CatPred-DB、PMAK-DB 和 TurnUp 的源 CSV。TurnUp 数据位于 `data/PMAK-DB/turnup/`；运行预处理时应显式指定此目录，因为脚本的历史默认值指向仓库外部路径。

先只检查文件和字段，再生成特征：

```bash
python preprocess/token_preprocess.py --dataset all --turnup-data-root data/PMAK-DB/turnup --check-only
python preprocess/token_preprocess.py --dataset all --turnup-data-root data/PMAK-DB/turnup
```

仅处理单组数据时可将 `--dataset all` 改为 `catpred`、`pmak` 或 `turnup`。默认特征写入 `features/P128_C32/`；每个数据划分包含蛋白质、RXNFP、ChemBERTa 特征以及对齐的标签和元数据。训练脚本读取这些生成的特征，而非直接读取 CSV。`features/` 不纳入 Git。

## 训练与评估

完成对应数据集的特征提取后，在本目录运行：

```bash
# CatPred-DB：验证集选模；默认仅训练种子 42
python train/train_BTRIKcat_inCatpredDB.py

# PMAK-DB：固定轮数训练，最终权重评估 test 与 OOD
python train/train_BTRIKcat_inPmakDB.py --epochs 100

# TurnUp：分别训练 cold_enzyme 和 cold_reaction，并评估独立测试集
python train/train_BTRIKcat_inCold.py --dataset both --phase final --epochs 100
```

CatPred 十种子运行可使用 `--seeds 37,38,39,40,41,42,43,44,45,46`；消融实验可使用 `--all-ablations`。TurnUp 五折验证可使用 `--dataset cold_enzyme --phase cv --seeds 42`，也可将数据集改为 `cold_reaction`。各入口均支持 `--help` 和 `--print-config`；先运行后者可检查实际配置与默认路径，而不会开始训练。训练输出默认写入 `results/`，此目录不纳入 Git。

## 数据与复现说明

数据划分及来源见 [`data/README.md`](data/README.md)。请在公开分发源 CSV 前核对上游数据集的归属和使用条款。仓库不包含预训练编码器权重、预处理生成的特征或训练好的模型权重，因此从新环境复现实验需先完成依赖安装与特征提取。
