"""
特征工程模块
从骨架序列中提取手工特征，用于传统机器学习分类器。

特征组：
  1. 关节位置统计量 — 中心化后的坐标均值/标准差
  2. 关节对距离 — 关键关节间欧氏距离的统计量
  3. 骨骼长度 — 各骨骼段长度的统计量
  4. 关节速度 — 帧间位移的统计量
  5. 关节运动范围 — 首帧到末帧的总位移
  6. 骨骼角度 — 关键关节角度（如肘、膝弯曲角）

所有特征先按帧计算，再用统计聚合（mean, std, min, max, skew）
将变长序列压缩为定长向量。
"""
import numpy as np
from scipy.stats import skew
from typing import List
from .data_loader import BONES


def temporal_stats(seq: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    沿时间轴计算统计聚合特征。
    
    Args:
        seq: shape (T, D) 的时序数据
    Returns:
        np.ndarray shape (5 * D,) — [mean, std, min, max, skew]
    """
    if seq.shape[0] <= 1:
        return np.zeros(5 * seq.shape[1], dtype=np.float32)
    mean = np.mean(seq, axis=0)       # (D,)
    std = np.std(seq, axis=0)         # (D,)
    _min = np.min(seq, axis=0)        # (D,)
    _max = np.max(seq, axis=0)        # (D,)
    sk = skew(seq, axis=0, bias=False)  # (D,)
    # 处理 NaN（常数序列的 skewness 为 NaN）
    sk = np.nan_to_num(sk, nan=0.0)
    return np.concatenate([mean, std, _min, _max, sk]).astype(np.float32)


def center_normalize(coords: np.ndarray) -> np.ndarray:
    """
    以脊柱底端（关节0）为中心归一化，实现平移不变性。
    
    Args:
        coords: shape (T, 25, 3)
    Returns:
        np.ndarray shape (T, 25, 3)，SpineBase 始终为原点
    """
    spine_base = coords[:, 0:1, :]  # (T, 1, 3)
    return coords - spine_base


def compute_velocities(coords: np.ndarray) -> np.ndarray:
    """
    计算关节速度（帧间差分）。
    
    Args:
        coords: shape (T, 25, 3)
    Returns:
        np.ndarray shape (T-1, 25)，每个关节的速度模长
    """
    diff = np.diff(coords, axis=0)  # (T-1, 25, 3)
    return np.linalg.norm(diff, axis=2)  # (T-1, 25)


def compute_bone_lengths(coords: np.ndarray) -> np.ndarray:
    """
    计算各骨骼段的长度。
    
    Args:
        coords: shape (T, 25, 3)
    Returns:
        np.ndarray shape (T, num_bones)
    """
    num_bones = len(BONES)
    lengths = np.zeros((coords.shape[0], num_bones), dtype=np.float32)
    for i, (j1, j2) in enumerate(BONES):
        vec = coords[:, j2, :] - coords[:, j1, :]  # (T, 3)
        lengths[:, i] = np.linalg.norm(vec, axis=1)  # (T,)
    return lengths


def compute_bone_angles(coords: np.ndarray) -> np.ndarray:
    """
    计算关键骨骼夹角。
    
    选取人体动作中最有区分度的关节角度：
      - 左肘 (ShoulderLeft-ElbowLeft 与 ElbowLeft-WristLeft)
      - 右肘 (ShoulderRight-ElbowRight 与 ElbowRight-WristRight)
      - 左膝 (HipLeft-KneeLeft 与 KneeLeft-AnkleLeft)
      - 右膝 (HipRight-KneeRight 与 KneeRight-AnkleRight)
      - 左右肩 (Neck-ShoulderLeft 与 SpineShoulder-Neck)
      - 左右髋 (SpineBase-HipLeft 与 SpineMid-SpineBase)
      - 头-颈 (Neck-Head 与 SpineShoulder-Neck)
    
    Args:
        coords: shape (T, 25, 3)，需已中心化
    Returns:
        np.ndarray shape (T, num_angles)
    """
    # 定义角度: (顶点关节, 近端关节, 远端关节)
    angle_defs = [
        # 左肘: ShoulderLeft(4) - ElbowLeft(5) - WristLeft(6)
        (5, 4, 6),
        # 右肘: ShoulderRight(8) - ElbowRight(9) - WristRight(10)
        (9, 8, 10),
        # 左膝: HipLeft(12) - KneeLeft(13) - AnkleLeft(14)
        (13, 12, 14),
        # 右膝: HipRight(16) - KneeRight(17) - AnkleRight(18)
        (17, 16, 18),
        # 左肩: SpineShoulder(20) - ShoulderLeft(4) - ElbowLeft(5)
        (4, 20, 5),
        # 右肩: SpineShoulder(20) - ShoulderRight(8) - ElbowRight(9)
        (8, 20, 9),
        # 头-颈: Neck(2) - Head(3) /* vs */ Neck(2) - SpineShoulder(20)
        (2, 3, 20),
        # 脊柱弯曲: SpineMid(1) - SpineShoulder(20) vs SpineBase(0) - SpineMid(1)
        (1, 20, 0),   # 上脊柱与下脊柱夹角
    ]
    
    T = coords.shape[0]
    num_angles = len(angle_defs)
    angles = np.zeros((T, num_angles), dtype=np.float32)
    
    for i, (vertex, prox, dist) in enumerate(angle_defs):
        v1 = coords[:, prox, :] - coords[:, vertex, :]   # 近端 -> 顶点
        v2 = coords[:, dist, :] - coords[:, vertex, :]   # 远端 -> 顶点
        # 余弦角
        dot = np.sum(v1 * v2, axis=1)  # (T,)
        norm1 = np.linalg.norm(v1, axis=1)  # (T,)
        norm2 = np.linalg.norm(v2, axis=1)  # (T,)
        cos_angle = dot / (norm1 * norm2 + 1e-10)
        cos_angle = np.clip(cos_angle, -1, 1)
        angles[:, i] = np.arccos(cos_angle)  # 弧度
    
    return angles


# 选取信息量最大的关节对距离（手工筛选减少维数）
KEY_JOINT_PAIRS = [
    # 手-手距离
    (7, 11),   # HandLeft - HandRight
    (6, 10),   # WristLeft - WristRight
    # 手-头距离
    (7, 3),    # HandLeft - Head
    (11, 3),   # HandRight - Head
    # 手-脚距离
    (7, 15),   # HandLeft - FootLeft
    (11, 19),  # HandRight - FootRight
    # 脚-脚距离
    (15, 19),  # FootLeft - FootRight
    (14, 18),  # AnkleLeft - AnkleRight
    # 手-髋距离
    (6, 12),   # WristLeft - HipLeft
    (10, 16),  # WristRight - HipRight
    # 头-脚距离
    (3, 0),    # Head - SpineBase
    (3, 14),   # Head - AnkleLeft
    (3, 18),   # Head - AnkleRight
    # 肘-膝
    (5, 13),   # ElbowLeft - KneeLeft
    (9, 17),   # ElbowRight - KneeRight
    # 肩-髋
    (4, 12),   # ShoulderLeft - HipLeft
    (8, 16),   # ShoulderRight - HipRight
    # 手-脊柱
    (7, 1),    # HandLeft - SpineMid
    (11, 1),   # HandRight - SpineMid
    # 膝-膝
    (13, 17),  # KneeLeft - KneeRight
]


def extract_features(sequence: np.ndarray) -> np.ndarray:
    """
    从骨架序列中提取完整特征向量。
    
    Args:
        sequence: shape (T, 25, 3) 的骨架序列
        
    Returns:
        np.ndarray shape (D,) 的特征向量
    """
    # 1. 中心归一化
    coords = center_normalize(sequence)  # (T, 25, 3)
    T, J, _ = coords.shape
    
    features = []
    
    # ---- 特征组1: 关节位置统计量 ----
    # 展平为 (T, 75)
    pos_flat = coords.reshape(T, -1)
    pos_stats = temporal_stats(pos_flat)  # (5*75=375,)
    features.append(pos_stats)
    
    # ---- 特征组2: 关键关节对距离 ----
    pair_dists = np.zeros((T, len(KEY_JOINT_PAIRS)), dtype=np.float32)
    for k, (j1, j2) in enumerate(KEY_JOINT_PAIRS):
        diff = coords[:, j1, :] - coords[:, j2, :]  # (T, 3)
        pair_dists[:, k] = np.linalg.norm(diff, axis=1)  # (T,)
    pair_stats = temporal_stats(pair_dists)  # (5*21=105,)
    features.append(pair_stats)
    
    # ---- 特征组3: 骨骼长度统计量 ----
    bone_lengths = compute_bone_lengths(coords)  # (T, num_bones)
    bone_stats = temporal_stats(bone_lengths)  # (5*24=120,)
    features.append(bone_stats)
    
    # ---- 特征组4: 关节速度 ----
    if T >= 2:
        velocities = compute_velocities(coords)  # (T-1, 25)
        vel_stats = temporal_stats(velocities)   # (5*25=125,)
    else:
        vel_stats = np.zeros(5 * J, dtype=np.float32)
    features.append(vel_stats)
    
    # ---- 特征组5: 首帧到末帧的关节总位移 ----
    displacement = coords[-1] - coords[0]  # (25, 3)
    features.append(displacement.ravel())   # (75,)
    
    # ---- 特征组6: 全局运动量 ----
    # 所有关节的总运动距离（轨迹长度）
    if T >= 2:
        total_motion = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=2))  # scalar
    else:
        total_motion = 0.0
    features.append(np.array([total_motion], dtype=np.float32))
    
    # ---- 特征组7: 骨骼角度统计量 ----
    bone_angles = compute_bone_angles(coords)  # (T, num_angles)
    angle_stats = temporal_stats(bone_angles)  # (5*8=40,)
    features.append(angle_stats)
    
    # ---- 特征组8: 关键关节位置 (相对于脊柱中心) 的均值 ----
    # 选取几个关键关节的3D均值位置
    key_joints = [3, 4, 8, 7, 11, 6, 10, 15, 19]  # Head, Shoulders, Hands, Wrists, Feet
    key_positions = np.mean(coords[:, key_joints, :], axis=0)  # (9, 3)
    features.append(key_positions.ravel())  # (27,)
    
    feat = np.concatenate(features)
    return feat


def extract_features_batch(sequences: List[np.ndarray]) -> np.ndarray:
    """
    批量提取特征。
    
    Args:
        sequences: list of np.ndarray, 每个 shape (T_i, 25, 3)
    Returns:
        np.ndarray shape (N, D)
    """
    feature_list = [extract_features(seq) for seq in sequences]
    X = np.stack(feature_list, axis=0)
    return X


if __name__ == '__main__':
    import os
    from skeletal_ml.data_loader import load_dataset
    from skeletal_ml.paths import DATA_DIR

    data_path = str(DATA_DIR)
    if os.path.isdir(data_path):
        X_train_seq, y_train, X_test_seq, y_test = load_dataset(data_path)
        
        print('提取训练集特征...')
        X_train = extract_features_batch(X_train_seq)
        print('提取测试集特征...')
        X_test = extract_features_batch(X_test_seq)
        
        print(f'训练特征矩阵: {X_train.shape}')  # (700, D)
        print(f'测试特征矩阵: {X_test.shape}')   # (300, D)
        
        # 检查 NaN
        print(f'Train NaN count: {np.isnan(X_train).sum()}')
        print(f'Test  NaN count: {np.isnan(X_test).sum()}')
