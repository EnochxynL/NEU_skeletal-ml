# 基于图神经网络与深度学习的骨架动作识别

> 东北大学 人工智能课程设计报告  
> 使用图卷积网络（ST-GCN/AGCN/ST-GIN/ST-PGCN）及频谱图CNN（ResNet+VirtualRadar）在 NTU RGB-D 骨架数据上进行 10 类人体动作识别，并与传统机器学习方法进行性能对比

---

## 摘要

本文在 NTU RGB-D 骨架数据上系统对比了 7 种深度学习模型与 4 种传统机器学习方法的动作识别性能。深度学习模型涵盖了图神经网络（ST-GCN、AGCN、ST-GIN、ST-PGCN、ST-PGCN-Pool）和频谱图CNN（ResNet18+VirtualRadar）两大范式，分别在关节（joint）和骨骼（bone）两种数据模态上进行训练和评估。实验结果表明，[待填充：最佳模型]在 10 类动作上取得了 [待填充]% 的测试准确率，较传统机器学习方法（SVM RBF, 80.33%）提升了 [待填充] 个百分点。此外，我们通过消融实验分析了图卷积算子的影响，并通过 joint+bone 集成融合进一步提升了识别精度。

**关键词**：骨架动作识别；图卷积网络（GCN）；时空图卷积（ST-GCN）；自适应图卷积（AGCN）；深度学习；人体动作识别

---

## 1. 问题定义

（与 DOC1.md §1 一致，此处简要复述）

给定一段人体动作的骨架序列 $\mathcal{S} = \{\mathbf{X}_1, \dots, \mathbf{X}_T\}$，每帧包含 25 个关节的 3D 坐标，目标是学习分类函数 $f: \mathcal{S} \to \{1,\dots,10\}$。10 个动作类别为：drink water (A1), eat meal (A2), brush teeth (A3), brush hair (A4), drop (A5), pick up (A6), throw (A7), sit down (A8), stand up (A9), clapping (A10)。

与 DOC1.md 中的手工特征 + ML 方案不同，本文采用端到端的深度学习方法，直接从原始骨架坐标学习时空特征表示，无需手工特征工程。

---

## 2. 数据集与预处理

### 2.1 数据来源与划分

数据来源与 DOC1.md §2.1 一致：NTU RGB-D 数据集，前 10 种动作各 100 个样本，共 1000 样本，7:3 训练/测试划分。

### 2.2 数据模态

本文使用两种骨架表示模态：

| 模态 | 描述 | 数据文件 |
|------|------|---------|
| **Joint** | 25 个关节的原始 $(x,y,z)$ 坐标，形状 $(N, 3, T, 25, 2)$ | `train_data_joint.npy` |
| **Bone** | 骨骼向量（关节间差分），由 joint 数据计算得到，形状 $(N, 3, T, 25, 2)$ | `train_data_bone.npy` |

生成 bone 数据：
```bash
uv run skeletal-genbone
```

### 2.3 数据预处理

深度学习模型内部使用 Batch Normalization 对输入数据进行标准化，无需手工中心化或归一化。与 DOC1.md 中显式进行中心归一化不同，BN 层在训练过程中自动学习最优的缩放和平移参数。

---

## 3. 模型方法

### 3.1 方法分类

我们将 7 种深度学习模型分为两大范式：

| 范式 | 模型 | 核心思想 | 来源 |
|------|------|---------|------|
| **图神经网络 (GCN)** | ST-GCN | 空间图卷积 + 时间卷积 | AAAI 2018 |
| | AGCN | ST-GCN + 自适应邻接矩阵 + 注意力机制 | — |
| | AAGCN | AGCN + 通道注意力 + 时间注意力 | — |
| | ST-GIN | 图同构卷积替代标准图卷积 | 基于 GIN |
| | ST-PGCN | 投影图卷积到可学习中心节点 | — |
| | ST-PGCN-Pool | ST-PGCN + 图层级池化下采样 | — |
| **频谱图 CNN** | ResNet+Radar | 骨骼数据 → 虚拟雷达频谱图 → ResNet18 | — |

### 3.2 图构造

人体骨架被建模为时空图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中：

- 节点集 $\mathcal{V}$：25 个身体关节
- 空间边 $\mathcal{E}_S$：基于人体自然连接的骨骼结构
- 时间边 $\mathcal{E}_T$：连接同一关节在连续帧间的对应关系

邻接矩阵 $\mathbf{A} \in \mathbb{R}^{K \times V \times V}$ 按空间配置策略（spatial configuration）划分为 3 个子集：自连接（关节自身）、向心连接（朝向躯干中心）、离心连接（远离躯干中心）。

$$\mathbf{A} = [\mathbf{A}_{\text{self}}, \mathbf{A}_{\text{inward}}, \mathbf{A}_{\text{outward}}]$$

### 3.3 ST-GCN（时空图卷积网络）

ST-GCN 是最经典的骨架动作识别图神经网络，由 Yan et al. (AAAI 2018) 提出。其核心是交替应用空间图卷积和时间卷积。

**空间图卷积**：对每个时间帧独立进行图卷积：

$$\mathbf{f}_{\text{out}}(v_i) = \sum_{v_j \in \mathcal{B}(v_i)} \frac{1}{Z_{ij}} \mathbf{f}_{\text{in}}(v_j) \cdot \mathbf{w}(l_{ij})$$

其中 $\mathcal{B}(v_i)$ 为节点 $v_i$ 的邻居集合，$Z_{ij}$ 为归一化因子，$\mathbf{w}$ 为根据分区标签 $l_{ij}$ 选择的权重函数。

具体实现为 1×1 卷积后接爱因斯坦求和：

$$\mathbf{X}' = \text{einsum}(\text{'nkctv,kvw→nctw'}, \text{Conv}_{1\times 1}(\mathbf{X}), \mathbf{A})$$

**时间卷积**：在时间轴上使用 $K_t \times 1$ 卷积核（本文取 $K_t = 9$）：

$$\mathbf{X}' = \text{Conv}_{K_t \times 1}(\mathbf{X})$$

**网络结构**：10 个 ST-GCN 块，通道数 [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]，在第 5 和第 8 块进行时间步长 2 下采样。每块包含：空间 GCN → BN → ReLU → 时间 Conv → BN → 残差连接 → ReLU。

**边重要性加权**：为邻接矩阵的每条边引入可学习权重 $\mathbf{M} \in \mathbb{R}^{K \times V \times V}$：

$$\mathbf{A}' = \mathbf{A} \odot \mathbf{M}$$

### 3.4 AGCN（自适应图卷积网络）

AGCN 在 ST-GCN 的基础上引入**自适应邻接矩阵**和**空间注意力机制**，使得图结构能够根据输入数据动态调整。

**自适应邻接矩阵**：在固定的物理邻接矩阵 $\mathbf{A}$ 基础上增加可学习的全局邻接矩阵 $\mathbf{P}_A$：

$$\mathbf{A}_{\text{adp}} = \mathbf{A} + \mathbf{P}_A$$

**空间自注意力**：通过两个 1×1 卷积 $\theta(\cdot)$ 和 $\phi(\cdot)$ 计算节点间的相似度，生成输入相关的注意力图：

$$\mathbf{A}_{\text{attn}} = \text{softmax}\left(\frac{\theta(\mathbf{X}) \cdot \phi(\mathbf{X})^T}{\sqrt{d}}\right)$$

最终邻接矩阵为三项之和：$\mathbf{A}_{\text{final}} = \mathbf{A} + \mathbf{P}_A + \mathbf{A}_{\text{attn}}$

### 3.5 AAGCN（双流注意力自适应图卷积网络）

AAGCN 在 AGCN 基础上进一步引入**时空通道注意力**（STC-Attention），对空间、时间和通道三个维度的特征同时进行注意力加权。

### 3.6 ST-GIN（时空图同构网络）

ST-GIN 将标准图卷积替换为**图同构卷积（Graph Isomorphism Convolution）**。与标准 GCN 使用均值聚合不同，GIN 对每个邻接矩阵分区使用独立的 MLP 处理，并引入可学习的自连接权重 $\epsilon$：

$$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) \cdot \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}\right)$$

多个分区的输出求和得到最终节点表示。GIN 的判别能力理论上等价于 Weisfeiler-Lehman 图同构测试，比标准 GCN 具有更强的图表达力。

### 3.7 ST-PGCN（时空投影图卷积网络）

ST-PGCN 在 ST-GCN 中插入**投影图卷积层（ProjectionGraphConv）**。该层学习一组可学习的"中心节点"（本文取 32 个），将原始 25 个关节通过软分配投影到中心节点：

$$q_{ij} = \text{softmax}\left(-\frac{1}{2}\left\|\frac{\mathbf{x}_i - \mathbf{c}_j}{\sigma(\mathbf{v}_j)}\right\|^2\right)$$

在投影空间计算新的邻接矩阵 $\mathbf{A}_{\text{proj}} = \mathbf{z}^T \mathbf{z}$，进行图卷积后再投影回原空间。核心思想是学习一个更紧凑的图表示空间。

### 3.8 ST-PGCN-Pool（时空投影图池化网络）

ST-PGCN-Pool 将投影机制扩展为**图池化操作**：在 ST-GCN 编码后，依次通过两个投影池化层将节点数从 25 → 512 → 256 进行层级下采样，最后通过全局平均池化得到特征向量。这种层次化的图粗化策略类似于 CNN 中的空间池化。

### 3.9 ResNet+VirtualRadar

将骨架序列转化为**雷达频谱图**，然后使用标准 ResNet18 进行分类。

**VirtualRadar 层**：模拟雷达系统，将骨架的每个骨骼段建模为椭球体，计算雷达后向散射截面（RCS）：

$$RCS = \frac{\pi a^4}{(\sin^2\theta \cos^2\phi + \sin^2\theta \sin^2\phi + a^2\cos^2\theta)^2}$$

其中 $a$ 为骨骼段长度，$\theta, \phi$ 为雷达视线与骨骼段之间的角度。所有骨骼段的回波叠加后通过 STFT 生成频谱图。

输入 $(N, 3, T, V, M)$ → VirtualRadar → 频谱图 $(N, F, L)$ → Resize → ResNet18 → 分类。

### 3.10 模型复杂度对比

| 模型 | 参数量 | 输入维度 | 关键算子 |
|------|--------|---------|---------|
| ST-GCN | ~3.1M | (N, 3, T, 25, 2) | GraphConvTD + TCN |
| AGCN | [待填充] | [待填充] | unit_gcn + unit_tcn |
| AAGCN | [待填充] | [待填充] | AGCN + STC-Attention |
| ST-GIN | [待填充] | (N, 3, T, 25, 2) | GraphIsoConvTD + TCN |
| ST-PGCN | [待填充] | (N, 3, T, 25, 2) | GraphConvTD + ProjectionGraphConv |
| ST-PGCN-Pool | [待填充] | (N, 3, T, 25, 2) | GraphConvTD + ProjectionGraphPool |
| ResNet+Radar | [待填充] | (N, 3, T, 25, 2) | VirtualRadar + ResNet18 |

---

## 4. 实验设计

### 4.1 训练配置

所有深度学习模型使用统一的训练配置以保证公平对比：

| 超参数 | 值 |
|--------|-----|
| 优化器 | SGD + Nesterov 动量 |
| 初始学习率 | 0.1 |
| 学习率衰减 | 第 20、30 epoch 各衰减 10 倍 |
| 训练轮数 | 40 |
| 批大小 | 4 |
| 权重衰减 | 0.0001 |
| 损失函数 | 交叉熵损失 (CrossEntropyLoss) |
| 图构建策略 | spatial configuration |
| 数据增强 | 关闭（random_choose/shift/move=False） |

### 4.2 实验矩阵

为了全面评估各方法，我们设计了以下实验矩阵：

| 实验编号 | 模型 | 模态 | 配置文件 |
|---------|------|------|---------|
| DL-01 | ST-GCN | Joint | `config/neu/train_joint_stgcn.yaml` |
| DL-02 | ST-GCN | Bone | `config/neu/train_bone_stgcn.yaml` |
| DL-03 | AGCN | Joint | `config/neu/train_joint.yaml` |
| DL-04 | AGCN | Bone | `config/neu/train_bone.yaml` |
| DL-05 | AAGCN | Joint | `config/neu/train_joint_aagcn.yaml` |
| DL-06 | AAGCN | Bone | `config/neu/train_bone_aagcn.yaml` |
| DL-07 | ST-GIN | Joint | `config/neu/train_joint_stgin.yaml` |
| DL-08 | ST-GIN | Bone | `config/neu/train_bone_stgin.yaml` |
| DL-09 | ST-PGCN | Joint | `config/neu/train_joint_stpgcn.yaml` |
| DL-10 | ST-PGCN | Bone | `config/neu/train_bone_stpgcn.yaml` |
| DL-11 | ST-PGCN-Pool | Joint | `config/neu/train_joint_stpgcnp.yaml` |
| DL-12 | ST-PGCN-Pool | Bone | `config/neu/train_bone_stpgcnp.yaml` |
| DL-13 | ResNet+Radar | Joint | 待创建 |
| EN-01 | ST-GCN + AGCN | Joint+Bone Ensemble | `skeletal-ensemble` |

### 4.3 评估指标

- **Top-1 准确率**：$\text{Acc} = \frac{N_{\text{correct}}}{N_{\text{total}}}$
- **Top-5 准确率**：真实标签是否在概率最高的 5 个预测之中
- **混淆矩阵**：各类别间的混淆模式
- **参数量**：模型总参数量（百万）
- **推理时间**：单样本平均推理耗时（ms）

### 4.4 消融实验设计

以 ST-GCN 为基线，分析各组件的贡献：

| 消融对比 | 分析内容 |
|---------|---------|
| ST-GCN → AGCN | 自适应邻接矩阵 + 空间注意力的贡献 |
| ST-GCN → ST-GIN | 图同构卷积 vs 标准图卷积 |
| ST-GCN → ST-PGCN | 投影图卷积的贡献 |
| ST-PGCN → ST-PGCN-Pool | 图池化下采样的贡献 |
| Joint → Bone | 不同模态对准确率的影响 |

---

## 5. 实验结果

### 5.1 综合对比：深度学习模型

| 模型 | 模态 | Top-1 Acc | Top-5 Acc | 参数量 | 训练时间 |
|------|------|----------|----------|--------|---------|
| ST-GCN | Joint | [待填充] | [待填充] | ~3.1M | [待填充] |
| ST-GCN | Bone | [待填充] | [待填充] | ~3.1M | [待填充] |
| AGCN | Joint | [待填充] | [待填充] | [待填充] | [待填充] |
| AGCN | Bone | [待填充] | [待填充] | [待填充] | [待填充] |
| AAGCN | Joint | [待填充] | [待填充] | [待填充] | [待填充] |
| AAGCN | Bone | [待填充] | [待填充] | [待填充] | [待填充] |
| ST-GIN | Joint | [待填充] | [待填充] | [待填充] | [待填充] |
| ST-GIN | Bone | [待填充] | [待填充] | [待填充] | [待填充] |
| ST-PGCN | Joint | [待填充] | [待填充] | [待填充] | [待填充] |
| ST-PGCN | Bone | [待填充] | [待填充] | [待填充] | [待填充] |
| ST-PGCN-Pool | Joint | [待填充] | [待填充] | [待填充] | [待填充] |
| ST-PGCN-Pool | Bone | [待填充] | [待填充] | [待填充] | [待填充] |
| ResNet+Radar | Joint | [待填充] | [待填充] | [待填充] | [待填充] |

### 5.2 集成融合

| 融合方案 | Top-1 Acc | Top-5 Acc |
|---------|----------|----------|
| Best Joint 单模型 | [待填充] | [待填充] |
| Best Bone 单模型 | [待填充] | [待填充] |
| Joint + Bone Ensemble (α=1.0) | [待填充] | [待填充] |

### 5.3 与 ML 方法对比

| 类别 | 方法 | 测试准确率 |
|------|------|-----------|
| 传统 ML | SVM (RBF) | 0.8033 |
| 传统 ML | Random Forest | 0.7867 |
| 传统 ML | SVM (Linear) | 0.7533 |
| 传统 ML | AdaBoost | 0.6767 |
| DL-GCN | ST-GCN (Joint) | [待填充] |
| DL-GCN | AGCN (Joint) | [待填充] |
| DL-GCN | Best GCN | [待填充] |
| DL-CNN | ResNet+Radar | [待填充] |
| Ensemble | Joint+Bone Fusion | [待填充] |

### 5.4 各类别性能分析（[最佳模型]）

| 动作类别 | 精确率 | 召回率 | F1-score |
|---------|--------|--------|----------|
| drink water | [待填充] | [待填充] | [待填充] |
| eat meal | [待填充] | [待填充] | [待填充] |
| brush teeth | [待填充] | [待填充] | [待填充] |
| brush hair | [待填充] | [待填充] | [待填充] |
| drop | [待填充] | [待填充] | [待填充] |
| pick up | [待填充] | [待填充] | [待填充] |
| throw | [待填充] | [待填充] | [待填充] |
| sit down | [待填充] | [待填充] | [待填充] |
| stand up | [待填充] | [待填充] | [待填充] |
| clapping | [待填充] | [待填充] | [待填充] |

### 5.5 消融实验结果

| 对比 | 基线模型 | 改进模型 | Δ Acc | 分析 |
|------|---------|---------|-------|------|
| GCN 算子 | ST-GCN | AGCN | [待填充] | 自适应注意力 vs 固定邻接矩阵 |
| GCN 算子 | ST-GCN | ST-GIN | [待填充] | GIN 卷积 vs 标准图卷积 |
| 图结构 | ST-GCN | ST-PGCN | [待填充] | 投影图卷积的贡献 |
| 图池化 | ST-PGCN | ST-PGCN-Pool | [待填充] | 层次池化的贡献 |
| 模态 | Joint | Bone | [待填充] | 骨骼向量 vs 关节坐标 |

---

## 6. 结果分析

### 6.1 DL vs ML 分析

[待填充：以下为分析框架]

1. **深度学习的优势**：端到端学习自动提取时空特征，避免了手工特征工程中的信息瓶颈。图神经网络原生地建模了骨架的图拓扑结构，能够捕获关节间的空间依赖和帧间的时间依赖。

2. **GCN vs 手工特征**：[待填充：GCN 是否全面超越了 ML 方法？如果有模型不如 SVM RBF，分析原因]

3. **小数据集挑战**：仅 700 训练样本对深度学习模型而言相对不足，可能存在过拟合。相对地，手工特征 + ML 方案在少量数据下更为稳健。

### 6.2 GCN 模型间分析

[待填充：比较不同 GCN 变体的性能差异，分析图卷积算子的影响]

### 6.3 模态分析

[待填充：Joint vs Bone 的性能对比，分析两种模态的互补性]

### 6.4 混淆分析

[待填充：参考 DOC1.md §5.3 的格式，分析 drink/eat/brush 等易混淆类别在 DL 模型中的表现，与 ML 方法对比]

### 6.5 效率分析

[待填充：参数量、推理速度与精度的 trade-off 分析]

---

## 7. 结论

本文在 NTU RGB-D 的 10 类骨架动作识别任务上系统对比了 7 种深度学习模型和 4 种传统机器学习方法，得到以下主要结论：

1. **[待填充：DL vs ML 的总体结论]**
2. **[待填充：GCN 变体的贡献排序]**
3. **[待填充：模态选择的建议]**
4. **[待填充：集成策略的效果]**
5. **[待填充：对实际应用的启示]**

---

## 8. 代码结构（深度学习部分）

```
NEU_skeletal-ml/
├── config/neu/                        # 训练配置文件
│   ├── train_joint.yaml               # AGCN joint (原始)
│   ├── train_bone.yaml                # AGCN bone (原始)
│   ├── train_joint_stgcn.yaml         # ST-GCN joint
│   ├── train_bone_stgcn.yaml          # ST-GCN bone
│   ├── train_joint_stgin.yaml         # ST-GIN joint
│   ├── train_bone_stgin.yaml          # ST-GIN bone
│   ├── train_joint_stpgcn.yaml        # ST-PGCN joint
│   ├── train_bone_stpgcn.yaml         # ST-PGCN bone
│   ├── train_joint_stpgcnp.yaml       # ST-PGCN-Pool joint
│   ├── train_bone_stpgcnp.yaml        # ST-PGCN-Pool bone
│   ├── train_joint_aagcn.yaml         # AAGCN joint
│   └── train_bone_aagcn.yaml          # AAGCN bone
├── src/skeletal_dl/
│   ├── model/
│   │   ├── agcn.py                    # AGCN (自适应图卷积)
│   │   ├── aagcn.py                   # AAGCN (双流注意力AGCN)
│   │   ├── stgcn.py                   # ST-GCN (mmskeleton AAAI'18 移植)
│   │   ├── stgin.py                   # ST-GIN (图同构网络)
│   │   ├── stpgcn.py                  # ST-PGCN (投影图卷积)
│   │   ├── stpgcnp.py                 # ST-PGCN-Pool (投影图池化)
│   │   ├── resnet.py                  # ResNet+VirtualRadar
│   │   ├── resnet18.py                # ResNet18 主干
│   │   ├── virtual_radar.py           # VirtualRadar 层
│   │   └── gcn_blocks.py             # 共享 GCN 算子
│   ├── graph/
│   │   ├── ntu_rgb_d.py              # NTU 25关节骨架图定义
│   │   └── tools.py                   # 邻接矩阵构建与归一化
│   ├── feeders/feeder.py              # PyTorch DataLoader
│   ├── trainer.py                     # 训练/测试管线
│   ├── ensemble.py                    # Joint+Bone 集成
│   └── data_gen/
│       ├── neu_gendata.py             # 生成 joint 数据
│       └── gen_neu_bone.py            # 生成 bone 数据
└── work_dir/neu/                      # 训练输出（日志/权重/配置快照）
```

---

## 9. 实验复现命令

### 9.1 环境准备

```bash
uv sync
```

### 9.2 数据准备

```bash
# 生成 joint 数据（如未生成）
uv run skeletal-gendata

# 生成 bone 数据
uv run skeletal-genbone
```

### 9.3 训练

```bash
# ST-GCN (Joint)
uv run skeletal-train-dl --config config/neu/train_joint_stgcn.yaml

# ST-GCN (Bone)
uv run skeletal-train-dl --config config/neu/train_bone_stgcn.yaml

# AGCN (Joint)
uv run skeletal-train-dl --config config/neu/train_joint.yaml

# AGCN (Bone)
uv run skeletal-train-dl --config config/neu/train_bone.yaml

# AAGCN (Joint)
uv run skeletal-train-dl --config config/neu/train_joint_aagcn.yaml

# ST-GIN (Joint)
uv run skeletal-train-dl --config config/neu/train_joint_stgin.yaml

# ST-PGCN (Joint)
uv run skeletal-train-dl --config config/neu/train_joint_stpgcn.yaml

# ST-PGCN-Pool (Joint)
uv run skeletal-train-dl --config config/neu/train_joint_stpgcnp.yaml
```

### 9.4 测试

```bash
# 以 ST-GCN joint 为例
uv run skeletal-train-dl --config config/neu/train_joint_stgcn.yaml \
    --phase test \
    --weights runs/neu_stgcn_joint-<epoch>-<step>.pt
```

### 9.5 集成

```bash
uv run skeletal-ensemble --datasets neu --alpha 1.0
```

### 9.6 运行 ML 基线

```bash
uv run skeletal-eval
```

---

## 参考文献

[1] Yan, S., Xiong, Y., & Lin, D. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition." AAAI 2018.

[2] Shi, L., Zhang, Y., Cheng, J., & Lu, H. "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition." CVPR 2019.

[3] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. "How Powerful are Graph Neural Networks?" ICLR 2019. (GIN)

[4] Shahroudy, A., et al. "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis." CVPR 2016.

[5] He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.

[6] Mahafza, B. "Radar Systems Analysis and Design Using MATLAB." Chapman & Hall/CRC 2000.

---

> **提示**：本文档为报告模板，所有 `[待填充]` 字段需在完成对应实验后填入实际数据。  
> 可视化输出（混淆矩阵、模型对比柱状图）保存在 `output/` 目录下，训练日志和权重保存在 `work_dir/` 和 `runs/` 目录下。
