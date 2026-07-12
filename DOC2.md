# 基于图卷积网络的骨架动作识别

> 东北大学 人工智能课程设计报告 · 深度学习部分  
> 对比实验：ST-GCN / AGCN / ST-GIN / ResNet+VirtualRadar + 预训练迁移

---

## 摘要

本文在 NTU RGB-D 骨架数据（10 类，700 训练 / 300 测试）上系统评估了三种图卷积网络（ST-GCN、AGCN、ST-GIN）和一种频谱图 CNN（ResNet18+VirtualRadar）的动作识别性能。实验包括：（1）小样本从头训练；（2）数据增强、学习率调度、Dropout 等训练策略消融；（3）NTU-60 预训练权重零样本评估。此外，将深度学习模型与传统机器学习方法（SVM RBF, 80.3%）进行对比，分析小样本场景下深度学习的局限性与潜力。

**关键词**：骨架动作识别；图卷积网络（GCN）；ST-GCN；AGCN；ST-GIN；VirtualRadar；预训练迁移

---

## 1. 问题定义

给定人体骨架序列 $\mathcal{S} = \{\mathbf{X}_1, \dots, \mathbf{X}_T\}$，每帧包含 25 个关节的 3D 坐标，目标是学习分类函数 $f: \mathcal{S} \to \{0, \dots, 9\}$。10 个动作类别：

| 编号 | 类别 | 动作描述 |
|------|------|---------|
| A01 | drink water | 举杯喝水 |
| A02 | eat meal | 进食 |
| A03 | brush teeth | 刷牙 |
| A04 | brush hair | 梳头 |
| A05 | drop | 丢下物品 |
| A06 | pick up | 拾起物品 |
| A07 | throw | 投掷 |
| A08 | sit down | 坐下 |
| A09 | stand up | 起立 |
| A10 | clapping | 拍手 |

---

## 2. 数据集与预处理

### 2.1 数据来源

NTU RGB+D 数据集前 10 种动作（A001–A010），每种 100 个 `.skeleton` 文件，共 1000 样本。按 7:3 划分为训练集（700）和测试集（300）。

输入张量形状：$(N, C, T, V, M) = (\text{batch}, 3, \text{frames}, 25, 2)$ —— 3 通道坐标、25 个关节、最多 2 人。

### 2.2 预处理

```bash
uv run skeletal-gendata          # 生成 train/val_data_joint.npy + label.pkl
```

深度学习模型使用 Batch Normalization 进行内部标准化，训练时不做手工中心化或归一化（`normalization: False`）。数据增强（随机帧选取、时间偏移、空间平移）在消融实验中有控制地开启。

---

## 3. 模型方法

### 3.1 方法概览

| 模型 | 核心思想 | 参数量 | 来源 |
|------|---------|--------|------|
| **ST-GCN** | 时空图卷积 + 边重要性加权 | ~3.1M | AAAI 2018 |
| **AGCN** | 自适应邻接矩阵 + 空间自注意力 | [待填充] | 2s-AGCN |
| **ST-GIN** | 图同构卷积替代标准图卷积 | [待填充] | GIN + ST-GCN |
| **ResNet+Radar** | 骨架→虚拟雷达频谱图→ResNet18 | [待填充] | — |

### 3.2 图构造

人体骨架建模为时空图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$：
- 节点 $\mathcal{V}$：25 个关节
- 空间边 $\mathcal{E}_S$：人体自然骨骼连接
- 时间边 $\mathcal{E}_T$：同一关节的跨帧连接

邻接矩阵 $\mathbf{A} \in \mathbb{R}^{K \times V \times V}$（$K=3$），按 spatial configuration 划分为自连接、向心、离心三个子集。

### 3.3 ST-GCN

**空间图卷积**（ConvTemporalGraphical）：
$$\mathbf{X}' = \text{einsum}(\text{'nkctv,kvw→nctw'}, \text{Conv}_{1\times1}(\mathbf{X}), \mathbf{A} \odot \mathbf{M})$$

其中 $\mathbf{M}$ 为可学习的边重要性权重。

**时间卷积**：$K_t \times 1$ 卷积核（$K_t = 9$），等价于沿时间轴的一维卷积。

**网络结构**：10 个 ST-GCN Block，通道 [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]，第 5、8 块 stride=2 时间下采样。每块：空间 GCN → BN → ReLU → 时间 Conv → BN → Dropout → +残差 → ReLU。

**输入预处理**：$(N, C, T, V, M) \rightarrow$ permute $\rightarrow (N \times M, V \times C, T) \rightarrow$ BatchNorm1d $\rightarrow$ reshape $\rightarrow (N \times M, C, T, V)$。这与 mmskeleton 官方实现一致。

### 3.4 AGCN

在 ST-GCN 基础上引入**自适应邻接矩阵**和**空间自注意力**：

$$\mathbf{A}_{\text{final}} = \mathbf{A} + \mathbf{P}_A + \text{softmax}\left(\frac{\theta(\mathbf{X})\phi(\mathbf{X})^T}{\sqrt{d}}\right)$$

- $\mathbf{P}_A$：可学习的全局邻接矩阵（与输入无关）
- $\theta, \phi$：两个 1×1 卷积，计算输入相关的节点相似度

**网络结构**：10 个 TCN_GCN_unit，通道配置与 ST-GCN 相同。输入 BN 使用 $(N, M \times V \times C, T)$ 维度，与 2s-AGCN 一致。分类器使用 `Linear(256, num_class)`（而非 ST-GCN 的 Conv2d）。

### 3.5 ST-GIN

将 ST-GCN 的空间图卷积替换为**图同构卷积（GraphIsoConvTD）**：

$$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) \cdot \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}\right)$$

- 每个邻接矩阵分区独立 MLP 处理
- $\epsilon$ 为可学习的自连接权重
- 使用前 2 个邻接矩阵分区（无自连接分区）
- GIN 的判别能力理论上等价于 Weisfeiler-Lehman 图同构测试

### 3.6 ResNet+VirtualRadar

骨架序列 → 虚拟雷达频谱图 → ResNet18 分类：

1. **VirtualRadar**：将每个骨骼段建模为椭球体，计算雷达后向散射截面（RCS），叠加所有骨骼段的回波
2. **STFT**：通过可微分的短时傅里叶变换生成频谱图（nnAudio）
3. **ResNet18**：标准 ImageNet 风格的 18 层残差网络，输入 1 通道频谱图

$$RCS = \frac{\pi a^4}{(\sin^2\theta \cos^2\phi + \sin^2\theta \sin^2\phi + a^2\cos^2\theta)^2}$$

### 3.7 预训练 ST-GCN（NTU-60 X-Sub）

使用 mmskeleton 在完整 NTU-60 数据集（Cross-Subject 协议，40320 训练样本）上预训练的 ST-GCN 权重，在 NEU 10 类测试集上进行**零样本评估**。预训练模型输出 60 类，NEU 的 10 类（A001–A010）直接映射到 NTU-60 的前 10 类。该实验用于量化大规模预训练对小样本场景的潜在增益。

---

## 4. 实验设计

### 4.1 统一训练配置

| 超参数 | 基线值 | 改进值 |
|--------|-------|--------|
| 优化器 | SGD + Nesterov (momentum=0.9) | 同 |
| 初始学习率 | 0.1 | 0.05 |
| 学习率衰减 | epoch 20, 30 (×0.1) | epoch 40, 60 (×0.1) |
| 训练轮数 | 40 | 80 |
| 批大小 | 32 | 32 |
| 权重衰减 | 0.0001 | 0.0001 |
| 损失函数 | CrossEntropyLoss | 同 |
| 数据增强 | 关闭 | random_choose/shift/move |

### 4.2 实验矩阵

#### 模型对比实验

| 编号 | 模型 | 配置文件 | 说明 |
|------|------|---------|------|
| DL-01 | ST-GCN | `train_joint_stgcn.yaml` | 基线 (40ep, lr=0.1) |
| DL-02 | AGCN | `train_joint_agcn.yaml` | 基线 (40ep, lr=0.1) |
| DL-03 | ST-GIN | `train_joint_stgin.yaml` | 基线 (40ep, lr=0.1, dropout=0.5) |
| DL-04 | ResNet+Radar | `train_joint_resnet.yaml` | 基线 (40ep, lr=0.1) |

#### ST-GCN 训练策略消融

| 编号 | 配置文件 | 数据增强 | LR调度 | Dropout | Epoch |
|------|---------|:---:|:---:|:---:|:---:|
| AB-01 | `train_joint_stgcn.yaml` | ✗ | decay@20,30 | ✗ | 40 |
| AB-02 | `train_joint_stgcn_aug.yaml` | ✓ | decay@20,30 | ✗ | 40 |
| AB-03 | `train_joint_stgcn_lr.yaml` | ✗ | decay@40,60 | 0.5 | 80 |

#### AGCN Dropout 消融

| 编号 | 配置文件 | Dropout | 模型文件 |
|------|---------|:---:|------|
| AB-04 | `train_joint_agcn.yaml` | ✗ | `agcn.py` |
| AB-05 | `train_joint_agcn_dropout.yaml` | 0.5 | `agcn_dropout.py` |

#### 预训练迁移

| 编号 | 模型 | 说明 |
|------|------|------|
| PT-01 | ST-GCN (NTU-60 X-Sub) | 零样本评估, 60 类输出 → 10 类 NEU |

### 4.3 评估指标

- **Top-1 / Top-5 准确率**
- **混淆矩阵**：各类别混淆模式
- **参数量**：模型总参数
- **与 ML 基线对比**：SVM RBF (80.3%), Random Forest (78.7%)

---

## 5. 实验结果

### 5.1 模型对比

| 模型 | Top-1 (%) | Top-5 (%) | 参数量 (M) |
|------|----------|----------|------------|
| ST-GCN | [待填充] | [待填充] | ~3.1 |
| AGCN | [待填充] | [待填充] | [待填充] |
| ST-GIN | [待填充] | [待填充] | [待填充] |
| ResNet+Radar | [待填充] | [待填充] | [待填充] |

### 5.2 ST-GCN 训练策略消融

| 实验 | Top-1 (%) | Δ vs 基线 | 分析 |
|------|----------|:---:|------|
| AB-01 基线 | [待填充] | — | — |
| AB-02 +数据增强 | [待填充] | [待填充] | 增强对小样本的收益 |
| AB-03 +LR+Dropout | [待填充] | [待填充] | 更长训练+正则化的效果 |

### 5.3 AGCN Dropout 消融

| 实验 | Top-1 (%) | Δ | 分析 |
|------|----------|:---:|------|
| AB-04 无 Dropout | [待填充] | — | — |
| AB-05 Dropout=0.5 | [待填充] | [待填充] | Dropout 对小样本过拟合的缓解 |

### 5.4 预训练迁移

| 模型 | Top-1 (%) | Top-5 (%) | 说明 |
|------|----------|----------|------|
| ST-GCN (scratch) | [待填充] | [待填充] | 700 样本从头训练 |
| ST-GCN (NTU-60 预训练) | [待填充] | [待填充] | 零样本, 无微调 |

### 5.5 与 ML 方法对比

| 方法 | 测试准确率 (%) |
|------|:---:|
| SVM (RBF) | 80.3 |
| Random Forest | 78.7 |
| SVM (Linear) | 75.3 |
| AdaBoost | 67.7 |
| ST-GCN (最佳) | [待填充] |
| AGCN (最佳) | [待填充] |
| ST-GIN (最佳) | [待填充] |
| ResNet+Radar | [待填充] |

### 5.6 各类别性能（[最佳模型]）

| 动作 | Precision | Recall | F1 |
|------|:---:|:---:|:---:|
| drink water | [~] | [~] | [~] |
| eat meal | [~] | [~] | [~] |
| brush teeth | [~] | [~] | [~] |
| brush hair | [~] | [~] | [~] |
| drop | [~] | [~] | [~] |
| pick up | [~] | [~] | [~] |
| throw | [~] | [~] | [~] |
| sit down | [~] | [~] | [~] |
| stand up | [~] | [~] | [~] |
| clapping | [~] | [~] | [~] |

---

## 6. 结果分析

### 6.1 DL vs ML：小样本的挑战

[待填充] 深度学习模型在 700 样本下可能不如手工特征+SVM。分析原因：
1. GCN 参数量大（~3M），700 样本不足以充分训练
2. 手工特征（关节角度/距离）编码了人体先验知识，在小样本下更有效
3. 数据增强和 Dropout 能否缩小差距？

### 6.2 训练策略消融分析

[待填充]
- 数据增强的实际收益
- LR 调度 + 更长训练的效果
- Dropout 是否有效缓解过拟合

### 6.3 预训练迁移分析

[待填充]
- NTU-60 预训练权重在 NEU 上的零样本表现
- 预训练模型的预测分布（是否倾向于预测 NTU-60 中与 NEU 类别相似的动作）
- 微调 vs 零样本的潜在改进空间

### 6.4 模型架构对比

[待填充]
- AGCN 的自适应邻接矩阵是否比 ST-GCN 固定图更有效？
- ST-GIN 的图同构卷积在小样本下的表现

### 6.5 混淆分析

[待填充] 参考 DOC1.md 格式，分析易混淆类别对（如 drink/eat/brush）

---

## 7. 结论

[待填充]

---

## 8. 代码结构

```
NEU_skeletal-ml/
├── config/neu/                          # 训练配置文件 (7个)
│   ├── train_joint_stgcn.yaml           # ST-GCN 基线
│   ├── train_joint_stgcn_aug.yaml       # ST-GCN + 数据增强
│   ├── train_joint_stgcn_lr.yaml        # ST-GCN + LR优化 + Dropout
│   ├── train_joint_agcn.yaml            # AGCN 基线
│   ├── train_joint_agcn_dropout.yaml    # AGCN + Dropout
│   ├── train_joint_stgin.yaml           # ST-GIN + Dropout
│   └── train_joint_resnet.yaml          # ResNet+VirtualRadar
├── src/skeletal_dl/
│   ├── model/
│   │   ├── stgcn.py                     # ST-GCN (mmskeleton AAAI'18)
│   │   ├── agcn.py                      # AGCN (2s-AGCN)
│   │   ├── agcn_dropout.py              # AGCN + Dropout 变体
│   │   ├── stgin.py                     # ST-GIN + Dropout
│   │   ├── resnet.py                    # ResNet+VirtualRadar
│   │   ├── resnet18.py                  # ResNet18 主干
│   │   ├── virtual_radar.py             # VirtualRadar 层
│   │   └── gcn_blocks.py               # 共享 GCN 算子
│   ├── graph/                           # 图定义 (ntu_rgb_d, tools)
│   ├── feeders/                         # 数据加载器
│   ├── trainer.py                       # 训练/测试管线
│   ├── ensemble.py                      # Joint+Bone 集成
│   ├── evaluate_pretrained.py           # 预训练模型评估
│   ├── compare_models.py                # 多模型对比
│   ├── demo_predict.py                  # 3D 骨架可视化
│   └── data_gen/                        # 数据生成
└── scripts/
    └── skeletal-dl.sh                   # 一键训练脚本
```

---

## 9. 复现命令

### 9.1 环境

```bash
git clone ...
cd NEU_skeletal-ml
uv sync          # 安装所有依赖 (含 nnAudio)
```

### 9.2 数据准备

```bash
uv run skeletal-gendata
```

### 9.3 训练（逐个或批量）

```bash
# 一键训练全部 7 个实验
bash scripts/skeletal-dl.sh train

# 或单独训练
uv run skeletal-train-dl --config config/neu/train_joint_stgcn.yaml
uv run skeletal-train-dl --config config/neu/train_joint_agcn.yaml
# ... 等等
```

### 9.4 评估

```bash
# 预训练 ST-GCN 零样本评估
uv run python src/skeletal_dl/evaluate_pretrained.py eval
# → outputs/pretrained_stgcn_eval.txt + confusion.png

# 全部模型对比（训练完成后修改 compare_models.py 中的 checkpoint 路径）
uv run python src/skeletal_dl/compare_models.py
# → outputs/model_comparison.txt + model_comparison.png
```

### 9.5 可视化

```bash
# 预训练模型预测
uv run python src/skeletal_dl/demo_predict.py data/test/S001C001P001R001A005.skeleton

# 自定义模型
uv run python src/skeletal_dl/demo_predict.py data/test/S001C001P001R001A005.skeleton \
    --checkpoint runs/neu_stgcn_joint-39-XXX.pt \
    --model skeletal_dl.model.stgcn.Model --num-class 10
```

---

## 参考文献

[1] Yan, S., Xiong, Y., & Lin, D. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition." AAAI 2018.

[2] Shi, L., Zhang, Y., Cheng, J., & Lu, H. "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition." CVPR 2019.

[3] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. "How Powerful are Graph Neural Networks?" ICLR 2019.

[4] Shahroudy, A., et al. "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis." CVPR 2016.

[5] He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.

---

> 所有 `[待填充]` 字段在训练完成后填入。可视化输出保存在 `outputs/`，训练日志在 `work_dir/`。
