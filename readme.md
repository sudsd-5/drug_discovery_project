# 药物发现平台

基于深度学习的药物发现平台，专注于药物-靶点相互作用预测、分子性质预测和虚拟筛选。

## 项目概述

本项目实现了一个端到端的药物发现计算平台，利用最新的深度学习技术，包括图神经网络(GNN)和卷积神经网络(CNN)来预测药物候选分子与靶点蛋白质之间的相互作用。主要功能包括：

1. **药物-靶点相互作用预测**：预测药物分子与蛋白质靶点的相互作用概率
2. **分子性质预测**：预测药物分子的物理化学性质和生物活性
3. **虚拟筛选**：筛选大量化合物库中可能与特定靶点相互作用的候选分子

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- RDKit
- scikit-learn, NumPy, pandas
- matplotlib, seaborn
- TensorBoard

推荐的硬件配置：
- CUDA 兼容的 GPU (例如 RTX 4060)
- 16GB+ RAM
- 100GB+ 存储空间

## 安装指南

1. 克隆项目：
   ```bash
   git clone https://github.com/yourusername/drug_discovery_project.git
   cd drug_discovery_project
   ```

2. 创建并激活虚拟环境：
   ```bash
   conda create -n drug_disc python=3.8
   conda activate drug_disc
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 数据准备

1. 将 `drug_target_interactions.csv` 和 `molecular_properties.csv` 文件放到以下位置：
   - `data/raw/drugbank/drug_target_interactions.csv`
   - `data/raw/molecular_properties.csv`

2. 运行数据处理脚本：
   ```bash
   python src/process_drug_target_interactions.py
   ```

## 项目结构

```
drug_discovery_project/
├── configs/                     # 配置文件
│   ├── dti_config.yaml         # 药物-靶点相互作用预测配置
│   └── training_config.yaml    # 训练配置
├── data/                        # 数据目录
│   ├── raw/                    # 原始数据
│   │   ├── drugbank/          # DrugBank 数据
│   │   │   └── drug_target_interactions.csv
│   │   └── molecular_properties.csv
│   └── processed/              # 处理后的数据
│       ├── features/          # 分子特征
│       └── interactions/      # 药物-靶点相互作用数据
├── logs/                        # 日志文件
├── models/                      # 保存的模型
├── runs/                        # TensorBoard 日志
├── src/                         # 源代码
│   ├── dti_model.py           # 药物-靶点相互作用预测模型
│   ├── process_drug_target_interactions.py   # 数据处理脚本
│   ├── train_dti.py           # 训练脚本
│   └── md_sim.py              # 分子动力学模拟脚本
├── run_dti_pipeline.py          # 运行完整流程的脚本
├── requirements.txt             # 项目依赖
└── README.md                    # 项目说明
```

## 使用方法

### 运行完整流程

使用 `run_dti_pipeline.py` 可以运行完整的药物-靶点相互作用预测流程：

```bash
# 运行完整流程
python run_dti_pipeline.py --all

# 只运行数据处理
python run_dti_pipeline.py --process_data

# 只运行模型训练
python run_dti_pipeline.py --train

# 使用特定配置文件
python run_dti_pipeline.py --config configs/my_custom_config.yaml
```

### 单独运行各个模块

也可以单独运行各个模块：

```bash
# 数据处理
python src/process_drug_target_interactions.py

# 模型训练
python src/train_dti.py
```

### 可视化训练过程

使用 TensorBoard 可视化训练过程：

```bash
tensorboard --logdir runs
```

## 模型架构

本项目使用了双流神经网络架构，分别对药物分子和蛋白质序列进行编码：

1. **药物编码器**：使用图神经网络(GCN/GAT)处理分子图结构
2. **蛋白质编码器**：使用多尺度CNN处理蛋白质氨基酸序列
3. **预测器**：将药物和蛋白质表示拼接后通过全连接网络预测相互作用概率

## 性能评估

模型性能使用以下指标评估：
- **准确率(Accuracy)**
- **AUC-ROC**：ROC曲线下面积
- **AUC-PR**：精确率-召回率曲线下面积

## 贡献者

- [您的名字] - 初始开发

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 引用

如果您在研究中使用了本项目，请引用：

```
@misc{DrugDiscoveryProject,
  author = {[您的名字]},
  title = {Drug Discovery Platform},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/drug_discovery_project}
}
```