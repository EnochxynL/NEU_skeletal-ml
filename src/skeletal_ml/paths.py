"""统一的路径解析。

无论从哪个目录运行脚本，都能正确定位数据集和输出目录。
支持通过环境变量 NEU_DATA_DIR 覆盖数据集路径。
"""
import os
from pathlib import Path

# 本文件: <root>/src/skeletal_ml/paths.py
# 项目根: parents[2] = NEU_legacy_solution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据集默认在项目根的上一级: ../NEU_data/实验数据
_DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# 允许环境变量覆盖
DATA_DIR = Path(os.environ.get("NEU_DATA_DIR", _DEFAULT_DATA_DIR))

# 输出目录（图片等生成物）
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


def ensure_output_dir() -> Path:
    """确保输出目录存在并返回其路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def check_data() -> None:
    """校验数据集路径是否存在，缺失时给出清晰报错。"""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"数据集目录不存在: {DATA_DIR}\n"
            f"请确认 NEU_data/实验数据 位于 {PROJECT_ROOT.parent}，"
            f"或设置环境变量 NEU_DATA_DIR 指向正确路径。"
        )
