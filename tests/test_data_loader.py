"""data_loader 与 feature_extraction 的冒烟测试。

运行:
    uv run pytest
"""
import numpy as np
import pytest

from skeletal_ml.data_loader import parse_skeleton, load_dataset, BONES, JOINT_NAMES
from skeletal_ml.feature_extraction import extract_features, extract_features_batch
from skeletal_ml.paths import DATA_DIR, TRAIN_DIR, TEST_DIR

# 数据集是否可用（缺失则跳过依赖数据的测试）
DATA_AVAILABLE = DATA_DIR.is_dir()
requires_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="数据集不可用")


def test_constants():
    """骨架常量的基本正确性。"""
    assert len(JOINT_NAMES) == 25
    # 所有骨骼端点索引应在 [0, 24] 内
    for j1, j2 in BONES:
        assert 0 <= j1 < 25
        assert 0 <= j2 < 25


@requires_data
def test_parse_single_skeleton():
    """解析单个骨架文件，形状应为 (T, 25, 3)。"""
    files = list(TRAIN_DIR.glob("*.skeleton"))
    assert len(files) > 0
    coords = parse_skeleton(str(files[0]))
    assert coords.ndim == 3
    assert coords.shape[1] == 25
    assert coords.shape[2] == 3
    assert coords.shape[0] > 0


@requires_data
def test_load_dataset_counts():
    """训练/测试集样本数与标签范围校验。"""
    X_tr, y_tr, X_te, y_te = load_dataset(str(DATA_DIR))
    assert len(X_tr) == len(y_tr) == 700
    assert len(X_te) == len(y_te) == 300
    # 标签应在 0-9
    assert set(np.unique(y_tr)).issubset(set(range(10)))
    # 每类应均衡
    assert np.bincount(y_tr).tolist() == [70] * 10
    assert np.bincount(y_te).tolist() == [30] * 10


@requires_data
def test_feature_extraction_shape_and_nan():
    """特征提取应产生固定维度且无 NaN。"""
    files = list(TRAIN_DIR.glob("*.skeleton"))[:5]
    seqs = [parse_skeleton(str(f)) for f in files]
    feats = extract_features_batch(seqs)
    assert feats.shape[0] == 5
    assert feats.shape[1] == 863  # 当前特征总维数
    assert not np.isnan(feats).any()


@requires_data
def test_single_feature_vector():
    """单序列特征向量维度与批量一致。"""
    files = list(TRAIN_DIR.glob("*.skeleton"))
    coords = parse_skeleton(str(files[0]))
    feat = extract_features(coords)
    assert feat.ndim == 1
    assert feat.shape[0] == 863
