# 基于手工特征与集成学习的人体骨架动作识别

> 东北大学 人工智能课程设计报告  
> 使用传统机器学习方法（SVM、随机森林、AdaBoost）在 NTU RGB-D 骨架数据上进行 10 类人体动作识别

---

## 摘要

本文研究基于 3D 人体骨架数据的动作识别问题。我们从 NTU RGB-D 数据集中提取了 10 类常见人体动作的骨架序列（共 1000 个样本），设计了 8 组手工特征将变长骨架序列编码为定长特征向量，并分别使用支持向量机（SVM）、随机森林（Random Forest）和 AdaBoost 三种经典机器学习算法进行分类。实验结果表明，RBF 核 SVM 在测试集上取得 80.33% 的准确率，优于随机森林（78.67%）、线性 SVM（75.33%）和 AdaBoost（67.67%）。

---

## 1. 问题定义

给定一段人体动作的骨架序列 $\mathcal{S} = \{\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_T\}$，其中第 $t$ 帧的骨架为 25 个关节点的 3D 坐标：

$$\mathbf{X}_t = \{(\mathbf{j}_t^{(1)}, \mathbf{j}_t^{(2)}, \dots, \mathbf{j}_t^{(25)})\}, \quad \mathbf{j}_t^{(k)} = (x_t^{(k)}, y_t^{(k)}, z_t^{(k)}) \in \mathbb{R}^3$$

目标是学习一个分类函数 $f: \mathcal{S} \to \{1, 2, \dots, 10\}$，将骨架序列映射到 10 个动作类别之一。

动作类别包括：drink water (A1), eat meal (A2), brush teeth (A3), brush hair (A4), drop (A5), pick up (A6), throw (A7), sit down (A8), stand up (A9), clapping (A10)。

---

## 2. 数据集

### 2.1 数据来源与划分

数据取自 NTU RGB-D 数据集，使用 Microsoft Kinect v2 传感器采集，包含 3 个不同角度摄像机的深度、RGB、红外和骨架数据。本文仅使用骨架信息，随机抽取前 10 种动作（A1–A10），每种动作 100 个骨架文件，共计 1000 个样本。

| 划分 | 样本数 | 每类样本数 |
|------|--------|-----------|
| 训练集 | 700 | 70 |
| 测试集 | 300 | 30 |

### 2.2 数据格式

每个 `.skeleton` 文件包含多帧骨架序列，帧数范围为 $[46, 188]$，均值约 81 帧。每帧包含 25 个关节点，每个关节点有 12 个特征值，本文仅使用前 3 个 $(x, y, z)$ 空间坐标。Kinect v2 的 25 关节骨架结构如下：

```
         Head(3)
           |
     Neck(2) — ShoulderRight(8) — ElbowRight(9) — WristRight(10) — HandRight(11)
        |           /                                              /       \
SpineShoulder(20)  ShoulderLeft(4) — ElbowLeft(5) — WristLeft(6) — HandLeft(7)
        |          \                                                \       /
     SpineMid(1)    \                                           HandTipRight(23)
        |             \                                            /
    SpineBase(0)       \                                      ThumbRight(24)
      /     \
 HipLeft(12) HipRight(16)
      |          |
 KneeLeft(13) KneeRight(17)
      |          |
AnkleLeft(14) AnkleRight(18)
      |          |
 FootLeft(15) FootRight(19)
```

---

## 3. 特征工程

手工特征设计是传统机器学习方法的核心。我们从骨架序列中提取了 8 组特征，总计 863 维。

### 3.1 数据预处理

**中心归一化**：以脊柱底端关节（SpineBase, 索引 0）为参考点进行平移归一化，消除不同拍摄距离带来的整体位移：

$$\mathbf{\hat{j}}_t^{(k)} = \mathbf{j}_t^{(k)} - \mathbf{j}_t^{(0)}, \quad \forall k \in \{1,\dots,25\}$$

### 3.2 特征组设计

**（1）关节位置统计量（375 维）**

直接将中心化后的 75 维坐标（25 关节 × 3 坐标）在时间轴上提取 5 种统计量：

$$\phi_{\text{pos}} = [\mu(\mathbf{P}), \sigma(\mathbf{P}), \min(\mathbf{P}), \max(\mathbf{P}), \gamma_1(\mathbf{P})]$$

其中 $\mathbf{P} = \{ \mathbf{\hat{j}}_t^{(k)} \}_{t=1}^T$ 为某关节某坐标分量的时间序列，$\mu$ 为均值，$\sigma$ 为标准差，$\gamma_1$ 为偏度（skewness）。

**（2）关键关节对距离（105 维）**

选取 21 对信息量最大的关节，计算欧氏距离并做统计聚合：

$$d_t(j_a, j_b) = \|\mathbf{\hat{j}}_t^{(j_a)} - \mathbf{\hat{j}}_t^{(j_b)}\|_2 = \sqrt{\sum_{c \in \{x,y,z\}} (\hat{j}_{t,c}^{(j_a)} - \hat{j}_{t,c}^{(j_b)})^2}$$

选取的关节对包括：手—手距离、手—头距离、手—脚距离、脚—脚距离、肘—膝关节距离等。

**（3）骨骼长度统计量（120 维）**

计算 24 个骨骼段的长度（邻接关节间的欧氏距离）：

$$l_t^{(b)} = \|\mathbf{\hat{j}}_t^{(u_b)} - \mathbf{\hat{j}}_t^{(v_b)}\|_2$$

其中 $(u_b, v_b)$ 为第 $b$ 根骨骼的两个端点关节索引。

**（4）关节速度统计量（125 维）**

计算帧间关节位移的模长来表示运动速度：

$$v_t^{(k)} = \|\mathbf{\hat{j}}_{t+1}^{(k)} - \mathbf{\hat{j}}_t^{(k)}\|_2, \quad t = 1, \dots, T-1$$

**（5）关节总位移（75 维）**

序列首末帧的关节位置差，反映动作的整体运动幅度：

$$\Delta\mathbf{j}^{(k)} = \mathbf{\hat{j}}_T^{(k)} - \mathbf{\hat{j}}_1^{(k)}$$

**（6）全局运动量（1 维）**

所有关节轨迹长度的总和：

$$E_{\text{motion}} = \sum_{k=1}^{25} \sum_{t=1}^{T-1} \|\mathbf{\hat{j}}_{t+1}^{(k)} - \mathbf{\hat{j}}_t^{(k)}\|_2$$

**（7）骨骼角度统计量（40 维）**

计算肘关节、膝关节、肩关节等 8 个关键铰链关节的角度：

$$\theta_t^{(i)} = \arccos\left( \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|} \right)$$

其中 $\mathbf{v}_1, \mathbf{v}_2$ 分别为近端→顶点和远端→顶点的向量。

**（8）关键关节均值位置（27 维）**

选取 Head、双肩、双手、双腕、双脚共 9 个关键关节在整个序列中的 3D 均值位置。

### 3.3 标准化与降维

在分类器输入前，对特征矩阵进行 Z-score 标准化：

$$x' = \frac{x - \mu}{\sigma}$$

对于 SVM 和 AdaBoost，进一步使用 PCA 降至 100–120 维，保留约 90% 的方差：

$$\mathbf{Z} = \mathbf{X} \mathbf{W}_k$$

其中 $\mathbf{W}_k$ 为前 $k$ 个最大特征值对应的特征向量组成的投影矩阵。

---

## 4. 分类模型

### 4.1 支持向量机（SVM）

**RBF 核 SVM** 通过非线性映射将数据投影到高维空间，寻找最大化分类间隔的超平面：

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{N} \xi_i$$

$$\text{s.t. } y_i(\mathbf{w}^T \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

RBF 核函数：

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

超参数：$C = 10$, $\gamma = \text{scale}$（即 $\gamma = \frac{1}{D \cdot \text{Var}(X)}$）。

**线性 SVM** 直接在高维特征空间中使用线性决策边界，采用 LinearSVC 实现（liblinear 求解器）。

### 4.2 随机森林（Random Forest）

随机森林是 Bagging 集成的代表性方法，通过自助采样（Bootstrap）构建 $B$ 棵决策树的集合：

$$f_{\text{RF}}(\mathbf{x}) = \arg\max_{y} \sum_{b=1}^{B} \mathbb{I}[f_b(\mathbf{x}) = y]$$

每棵树在分裂节点时随机选取 $m = \lfloor \sqrt{D} \rfloor$ 个特征子集进行最优分裂，基尼不纯度作为分裂准则：

$$Gini(t) = 1 - \sum_{c=1}^{C} p_c^2(t)$$

超参数：$B = 100$, 最大深度 $= 10$, 最小分裂样本数 $= 5$。

### 4.3 AdaBoost

AdaBoost 通过迭代训练弱分类器并调整样本权重，逐步关注被误分类的样本：

1. 初始化样本权重：$w_i^{(1)} = \frac{1}{N}$
2. 对于 $m = 1$ 到 $M$：
   - 训练弱分类器 $h_m$，计算加权误差 $\epsilon_m = \sum_i w_i^{(m)} \mathbb{I}[h_m(\mathbf{x}_i) \neq y_i]$
   - 计算分类器权重 $\alpha_m = \frac{1}{2}\ln\left(\frac{1-\epsilon_m}{\epsilon_m}\right)$
   - 更新样本权重 $w_i^{(m+1)} = w_i^{(m)} \exp(-\alpha_m y_i h_m(\mathbf{x}_i))$ 并归一化
3. 最终分类器：$H(\mathbf{x}) = \text{sign}\left(\sum_m \alpha_m h_m(\mathbf{x})\right)$

超参数：弱学习器 $=$ 深度为 3 的决策树, $M = 300$, 学习率 $\eta = 1.0$。

---

## 5. 实验与评估

### 5.1 实验设置

- 评估方式：在训练集上使用 5 折分层交叉验证进行超参数搜索，在独立测试集上报告最终性能
- 评价指标：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1-score

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### 5.2 实验结果

| 模型 | CV 准确率 | 测试准确率 | 训练时间 |
|------|----------|-----------|---------|
| SVM (RBF) | 0.7757 | **0.8033** | 21s |
| Random Forest | 0.7929 | 0.7867 | 27s |
| SVM (Linear) | 0.7557 | 0.7533 | 61s |
| AdaBoost | 0.6600 | 0.6767 | 76s |

### 5.3 各类别分析（SVM RBF，最佳模型）

| 动作类别 | 精确率 | 召回率 | F1-score |
|---------|--------|--------|----------|
| drink water | 0.588 | 0.667 | 0.625 |
| eat meal | 0.643 | 0.600 | 0.621 |
| brush teeth | 0.625 | 0.667 | 0.645 |
| brush hair | 0.700 | 0.700 | 0.700 |
| drop | 0.889 | 0.800 | 0.842 |
| pick up | 0.857 | 1.000 | 0.923 |
| throw | 0.967 | 0.967 | 0.967 |
| sit down | 1.000 | 0.867 | 0.929 |
| stand up | 0.967 | 0.967 | 0.967 |
| clapping | 0.857 | 0.800 | 0.828 |

### 5.4 结果分析

1. **SVM RBF 最优**：RBF 核的非线性映射能力有效捕捉了骨架动作中的复杂模式，80.33% 的准确率在 10 类问题上表现良好。

2. **随机森林接近 SVM**：随机森林的集成特性表现稳健，且训练速度更快，不需要 PCA 降维。

3. **"吃/喝"类别混淆严重**：drink water 和 eat meal 的 F1 分数最低（0.625 和 0.621），因为它们都涉及手部靠近头部的动作，骨架差异微小。两者之间的混淆是合理的——都是上肢+头部的交互动作。

4. **大幅度动作识别效果好**：throw (0.967), stand up (0.967), sit down (0.929) 等涉及全身大幅度运动的动作几乎完美识别。

5. **AdaBoost 表现差**：AdaBoost 对特征噪声和异常值敏感，在 863 维高维特征空间上易过拟合。PCA 降维后虽有改善但仍不及 SVM 和随机森林。

6. **线性 SVM 的局限**：线性决策边界无法分离骨架姿态中的非线性模式，表明动作识别是一个内在非线性的分类问题。

### 5.5 案例分析

使用 `predict.py` 脚本对单个骨架文件进行可视化，同时显示模型预测标签、真实标签及 Top-5 概率分布。

**正例：sit down 被正确识别**

```bash
$ uv run skeletal-predict data/test/S010C001P018R001A008.skeleton
```

```
  Prediction: [7] sit down
  True label: [7] sit down
  Top-5 probabilities:
    1. [7] sit down        0.813 ←
    2. [1] eat meal        0.037
    3. [5] pick up         0.030
    4. [8] stand up        0.028
    5. [4] drop            0.027
```

模型以 81.3% 的高置信度正确识别出 sit down，其他类别概率均低于 4%。sit down 动作涉及从站立到坐下的全身大幅度姿态变化，骨架的整体重心下降和下肢弯曲模式非常独特，模型能够可靠地捕获。

**反例：eat meal 被自信地误判为 brush hair**

```bash
$ uv run skeletal-predict data/test/S008C003P025R002A002.skeleton
```

```
  Prediction: [3] brush hair
  True label: [1] eat meal
  Top-5 probabilities:
    1. [3] brush hair      0.911 ←
    2. [5] pick up         0.021
    3. [2] brush teeth     0.020
    4. [9] clapping        0.016
    5. [0] drink water     0.009
```

打开文件查看后发现，这是文件本身的错误。这个特定文件 `S008C003P025R002A002.skeleton` 中的所有 79 帧的 `body` 计数都是 **0**——即在采集过程中，Kinect 传感器完全没有检测到人体。

解析函数 `parse_skeleton` 在 `data_loader.py:30` 按帧循环时为 `num_body=0` 时跳过所有关节读取，因此 `(79, 25, 3)` 的坐标矩阵保持全零。中心化后依旧全是 `(0,0,0)`，所以 25 个关节全都挤在原点上，肉眼只看到一个点。

**反例：drink water 边缘正确**

```bash
$ uv run skeletal-predict data/test/S009C002P016R001A001.skeleton
```

```
  Prediction: [0] drink water
  True label: [0] drink water
  Top-5 probabilities:
    1. [0] drink water     0.496 ←
    2. [2] brush teeth     0.311
    3. [1] eat meal        0.068
    4. [6] throw           0.033
    5. [9] clapping        0.027
```

虽然最终预测正确，但置信度仅 49.6%，brush teeth 以 31.1% 紧随其后。drink water、brush teeth 和 eat meal 三个"手—头"交互动作构成了一个易混淆的子群，也是整个数据集中识别难度最大的类别组。

eat meal 和 brush hair 的动作模式高度相似：两者都是手部在头部附近做往复运动（吃饭时手从桌面到嘴边，梳头时手在头部附近移动）。纯统计特征（均值/方差/偏度）将这两种运动模式的时序统计量压缩为相似的值，丢失了关键的时序顺序和运动方向信息。这提示后续改进方向：引入时序建模（如 DTW 距离特征）或频域特征（傅里叶描述子）来捕获运动的先后顺序。

---

## 6. 代码结构

```
NEU_legacy_solution/
├── data_loader.py          # 骨架数据解析（.skeleton → numpy）
├── feature_extraction.py   # 863 维手工特征提取
├── train_ml.py             # 网格搜索训练管线
├── evaluate.py             # 最优参数评估 + 可视化
├── visualization.py        # 混淆矩阵 / 对比柱状图
├── show_skeleton.py        # 骨架 2D/3D 可视化（遗留）
├── show_predict.py         # 单样本推理 + 标签可视化
├── confusion_matrices.png  # 生成的四模型混淆矩阵
└── model_comparison.png    # 模型对比柱状图
```

---

## 7. 结论

本文在 NTU RGB-D 骨架数据上，通过手工设计的 8 组运动学特征（关节位置、关节对距离、骨骼长度、关节速度、关节位移、骨骼角度等），结合三种传统机器学习算法完成了 10 类人体动作识别任务。

主要发现：
- **SVM + RBF 核** 取得了最高的测试准确率（80.33%），证明了非线性核方法在骨架动作识别中的有效性
- **随机森林** 以 78.67% 的准确率紧随其后，且训练效率高、不需要 PCA 降维
- 手工特征 + 传统分类器的方案在仅有 700 训练样本的条件下表现可靠，说明精心设计的运动学特征能够捕获动作的判别性信息
- 主要挑战在于相似上肢动作（如 eat/drink/brush）的区分，需进一步考虑更精细的关节运动时序模式

---

## 参考文献

[1] Shahroudy, A., et al. "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis." CVPR 2016.

[2] Vemulapalli, R., et al. "Human Action Recognition by Representing 3D Skeletons as Points in a Lie Group." CVPR 2014.

[3] Cortes, C., & Vapnik, V. "Support-Vector Networks." Machine Learning, 1995.

[4] Breiman, L. "Random Forests." Machine Learning, 2001.

[5] Freund, Y., & Schapire, R. E. "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting." JCSS, 1997.
