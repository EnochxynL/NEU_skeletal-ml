# skeletal-ml — NTU RGB-D 人体骨架动作识别

基于**手工运动学特征**与**传统机器学习**（SVM / 随机森林 / AdaBoost）的 10 类人体动作识别。

## 目录结构

```
NEU_legacy_solution/
├── pyproject.toml          # 项目配置 + 依赖 + 入口点
├── uv.lock
├── README.md
├── .gitignore
├── src/
│   └── skeletal_ml/            # 核心库（可 import）
│       ├── __init__.py
│       ├── data_loader.py      # 解析 .skeleton → numpy
│       ├── feature_extraction.py  # 863 维手工特征
│       ├── visualization.py    # 混淆矩阵 / 对比图
│       ├── paths.py            # 数据集与输出路径解析
│       └── cli/                # 命令行入口
│           ├── train.py        # 网格搜索调参
│           ├── evaluate.py     # 最优参数训练 + 出图
│           ├── predict.py      # 单样本推理 + 骨架可视化
│           └── show_skeleton.py  # 遗留骨架动画
├── tests/
│   └── test_data_loader.py
└── outputs/                 # 生成的图片（git 忽略）
```

## 环境准备

```bash
uv sync              # 安装依赖 + 以可编辑模式安装 skeletal_ml
```

数据集默认位于项目上一级的 `../NEU_data/实验数据/{train,test}`。
如需自定义路径，设置环境变量：

```bash
export NEU_DATA_DIR=/path/to/实验数据
```

## 使用

三个入口点（`uv sync` 后可用）：

```bash
# 1. 复现全部结果：训练 4 个模型 + 生成混淆矩阵/对比图（→ outputs/）
uv run skeletal-eval

# 2. 网格搜索超参数调优（一次性，耗时较长）
uv run skeletal-train

# 3. 单样本推理 + 骨架可视化（显示预测/真实标签）
uv run skeletal-predict ../NEU_data/实验数据/test/S001C001P001R002A001.skeleton
uv run skeletal-predict <file.skeleton> --gif   # 保存 GIF 动画
```

也可直接运行模块：

```bash
uv run python -m skeletal_ml.cli.evaluate
```

## 测试

```bash
uv run pytest
```

## 结果概览

| 模型 | 测试准确率 |
|------|-----------|
| SVM (RBF) | **80.33%** |
| Random Forest | 78.67% |
| SVM (Linear) | 75.33% |
| AdaBoost | 67.67% |

完整分析见项目根目录的 `NEU_report.md`。
