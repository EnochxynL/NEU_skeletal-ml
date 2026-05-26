"""
骨架数据加载器
解析 NTU RGB-D .skeleton 文件，提取 3D 关节坐标
"""
import os
import re
import numpy as np
from glob import glob
from typing import Dict, Tuple


def parse_skeleton(filepath: str) -> np.ndarray:
    """
    解析单个 .skeleton 文件，返回关节坐标数组。
    
    文件格式（每帧）:
        帧数
        body数（通常为1）
        body元信息（跳过）
        关节数（25）
        25行，每行12个特征: x y z depthX depthY colorX colorY orientW orientX orientY orientZ trackingState
    
    Returns:
        np.ndarray of shape (num_frames, 25, 3) — x,y,z 坐标
    """
    with open(filepath, 'r') as f:
        num_frames = int(f.readline().strip())
        coords = np.zeros((num_frames, 25, 3), dtype=np.float32)
        
        for t in range(num_frames):
            num_body = int(f.readline().strip())
            for m in range(num_body):
                # 跳过 body 元信息行
                f.readline()
                num_joint = int(f.readline().strip())
                for j in range(num_joint):
                    parts = f.readline().strip().split()
                    if j < 25:
                        coords[t, j, 0] = float(parts[0])  # x
                        coords[t, j, 1] = float(parts[1])  # y
                        coords[t, j, 2] = float(parts[2])  # z
        return coords


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载训练集和测试集。
    
    Args:
        data_dir: NEU_data/实验数据 的路径
        
    Returns:
        X_train: list of np.ndarray, 每个元素 shape (T_i, 25, 3)
        y_train: np.ndarray of int, shape (N_train,), label 0-9
        X_test: 同上
        y_test: 同上
    """
    label_map = {
        'A001': 0, 'A002': 1, 'A003': 2, 'A004': 3, 'A005': 4,
        'A006': 5, 'A007': 6, 'A008': 7, 'A009': 8, 'A010': 9,
    }
    
    pattern = re.compile(r'A(\d{3})')
    
    def _load_split(split):
        folder = os.path.join(data_dir, split)
        files = sorted(glob(os.path.join(folder, '*.skeleton')))
        X, y = [], []
        for f in files:
            m = pattern.search(f)
            if m:
                X.append(parse_skeleton(f))
                y.append(label_map[f'A{m.group(1)}'])
        return X, np.array(y, dtype=np.int32)
    
    X_train, y_train = _load_split('train')
    X_test, y_test = _load_split('test')
    
    return X_train, y_train, X_test, y_test


# 人体骨架关节索引与连接关系（Kinect v2, 25 joint）
JOINT_NAMES = [
    'SpineBase',     # 0
    'SpineMid',      # 1
    'Neck',          # 2
    'Head',          # 3
    'ShoulderLeft',  # 4
    'ElbowLeft',     # 5
    'WristLeft',     # 6
    'HandLeft',      # 7
    'ShoulderRight', # 8
    'ElbowRight',    # 9
    'WristRight',    # 10
    'HandRight',     # 11
    'HipLeft',       # 12
    'KneeLeft',      # 13
    'AnkleLeft',     # 14
    'FootLeft',      # 15
    'HipRight',      # 16
    'KneeRight',     # 17
    'AnkleRight',    # 18
    'FootRight',     # 19
    'SpineShoulder', # 20
    'HandTipLeft',   # 21
    'ThumbLeft',     # 22
    'HandTipRight',  # 23
    'ThumbRight',    # 24
]

# 骨架连接关系（骨骼段）
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),    # 脊柱 -> 头
    (20, 4), (4, 5), (5, 6), (6, 7),      # 左臂
    (7, 21), (7, 22),                       # 左手
    (20, 8), (8, 9), (9, 10), (10, 11),    # 右臂
    (11, 23), (11, 24),                     # 右手
    (0, 12), (12, 13), (13, 14), (14, 15), # 左腿
    (0, 16), (16, 17), (17, 18), (18, 19), # 右腿
]


if __name__ == '__main__':
    # 快速测试
    import sys
    data_path = '../NEU_data/实验数据'
    if os.path.isdir(data_path):
        X_train, y_train, X_test, y_test = load_dataset(data_path)
        print(f'Train: {len(X_train)} sequences, labels: {np.bincount(y_train)}')
        print(f'Test : {len(X_test)} sequences, labels: {np.bincount(y_test)}')
        lens = [x.shape[0] for x in X_train]
        print(f'Train frames: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}')
    else:
        print(f'Data dir not found: {data_path}')
