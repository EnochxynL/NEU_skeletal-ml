"""
骨架动作可视化 —— 显示 3D 骨架 + 预测标签 + 真实标签

用法:
    uv run skeletal-predict <path/to/file.skeleton>
    uv run skeletal-predict <path/to/file.skeleton> --gif
    # 或
    uv run python scripts/show_predict.py <path/to/file.skeleton>
"""
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from skeletal_ml.data_loader import parse_skeleton, BONES
from skeletal_ml.feature_extraction import extract_features
from skeletal_ml.paths import DATA_DIR, ensure_output_dir


# ---- 动作名称 ----
ACTION_NAMES = [
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop',
    'pick up', 'throw', 'sit down', 'stand up', 'clapping'
]

# ---- 加载最优模型 ----
def load_model():
    """加载训练好的 SVM RBF 模型"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline

    from skeletal_ml.data_loader import load_dataset
    from skeletal_ml.feature_extraction import extract_features_batch

    data_dir = str(DATA_DIR)
    print('加载训练数据...', end=' ', flush=True)
    X_train_seq, y_train, _, _ = load_dataset(data_dir)
    X_train = extract_features_batch(X_train_seq)
    print(f'({X_train.shape[0]} samples, {X_train.shape[1]} features)')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=120, random_state=42)),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
    ])
    print('训练 SVM RBF...', end=' ', flush=True)
    pipe.fit(X_train, y_train)
    print('done.')
    return pipe


def predict_single(model, filepath):
    """对单个骨架文件预测，返回 (pred_label, pred_name, probs)"""
    seq = parse_skeleton(filepath)
    feat = extract_features(seq).reshape(1, -1)
    pred = model.predict(feat)[0]
    probs = model.predict_proba(feat)[0]
    return pred, ACTION_NAMES[pred], probs


def show_skeleton_3d(filepath, model, save_gif=False):
    """
    3D 骨架动画，标题显示预测标签和真实标签。
    """
    # 解析骨架
    coords = parse_skeleton(filepath)  # (T, 25, 3)
    T = coords.shape[0]

    # 预测
    pred_label, pred_name, probs = predict_single(model, filepath)

    # 真实标签
    m = re.search(r'A(\d{3})', os.path.basename(filepath))
    true_label = int(m.group(1)) - 1 if m else -1
    true_name = ACTION_NAMES[true_label] if true_label >= 0 else 'Unknown'

    # 是否正确
    correct = '✓' if pred_label == true_label else '✗'

    # 中心化
    coords = coords - coords[:, 0:1, :]
    coords = coords[..., [0, 2, 1]]    # NTU→显示: y→-z, z→y
    coords[..., 1] *= -1

    # 坐标范围
    xmin, xmax = coords[:, :, 0].min(), coords[:, :, 0].max()
    ymin, ymax = coords[:, :, 1].min(), coords[:, :, 1].max()
    zmin, zmax = coords[:, :, 2].min(), coords[:, :, 2].max()
    pad = 0.3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    lines = []
    scatter = None

    def init():
        nonlocal scatter
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_zlim(zmin - pad, zmax + pad)
        # NTU RGB-D 坐标系: x=左右, y=上下, z=深度(朝向相机)
        # elev=10 azim=-90 从侧面平视，人"站立"而非"躺倒"
        ax.view_init(elev=10, azim=-90)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        title = (f'Pred: [{pred_label}] {pred_name} | '
                 f'True: [{true_label}] {true_name} {correct}')
        # 如果预测正确,标题绿色;错误则红色
        color = 'green' if correct == '✓' else 'red'
        ax.set_title(title, fontsize=13, color=color, fontweight='bold')

        scatter = ax.scatter([], [], [], c='red', s=30)
        for _ in BONES:
            line, = ax.plot([], [], [], c='steelblue', lw=1.5)
            lines.append(line)
        return [scatter] + lines

    def update(frame):
        pts = coords[frame]  # (25, 3)
        scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for i, (j1, j2) in enumerate(BONES):
            lines[i].set_data_3d(
                [pts[j1, 0], pts[j2, 0]],
                [pts[j1, 1], pts[j2, 1]],
                [pts[j1, 2], pts[j2, 2]]
            )
        # 更新帧号
        ax.set_title(
            f'Pred: [{pred_label}] {pred_name} | '
            f'True: [{true_label}] {true_name} {correct} | '
            f'Frame {frame+1}/{T}',
            fontsize=13,
            color='green' if correct == '✓' else 'red',
            fontweight='bold'
        )
        return [scatter] + lines

    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        interval=50, blit=False, repeat=True)

    if save_gif:
        base = os.path.basename(filepath).replace('.skeleton', '_pred.gif')
        out = str(ensure_output_dir() / base)
        ani.save(out, writer='pillow', fps=20, dpi=80)
        print(f'GIF saved: {out}')
    else:
        plt.show()


def show_skeleton_static(filepath, model, num_frames=4):
    """
    静态多帧对比：挑选若干关键帧并排展示。
    """
    coords = parse_skeleton(filepath)
    T = coords.shape[0]
    coords = coords - coords[:, 0:1, :]
    coords = coords[..., [0, 2, 1]]    # NTU→显示: y→-z, z→y
    coords[..., 1] *= -1

    pred_label, pred_name, probs = predict_single(model, filepath)

    m = re.search(r'A(\d{3})', os.path.basename(filepath))
    true_label = int(m.group(1)) - 1 if m else -1
    true_name = ACTION_NAMES[true_label] if true_label >= 0 else 'Unknown'
    correct = '✓ CORRECT' if pred_label == true_label else '✗ WRONG'

    # 均匀选帧
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 5),
                             subplot_kw={'projection': '3d'})

    color = 'green' if correct == '✓ CORRECT' else 'red'
    fig.suptitle(f'Pred: [{pred_label}] {pred_name}  |  '
                 f'True: [{true_label}] {true_name}  |  {correct}',
                 fontsize=14, color=color, fontweight='bold')

    xall = coords[:, :, 0].ravel()
    yall = coords[:, :, 1].ravel()
    zall = coords[:, :, 2].ravel()
    pad = 0.2

    for ax, idx in zip(axes, indices):
        pts = coords[idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=20)
        for j1, j2 in BONES:
            ax.plot([pts[j1, 0], pts[j2, 0]],
                    [pts[j1, 1], pts[j2, 1]],
                    [pts[j1, 2], pts[j2, 2]], c='steelblue', lw=1.5)
        ax.set_xlim(xall.min() - pad, xall.max() + pad)
        ax.set_ylim(yall.min() - pad, yall.max() + pad)
        ax.set_zlim(zall.min() - pad, zall.max() + pad)
        ax.view_init(elev=10, azim=-90)
        ax.set_title(f'Frame {idx+1}/{T}', fontsize=10)

    # 右下角放概率图
    plt.tight_layout()

    # 保存静态图
    base = os.path.basename(filepath).replace('.skeleton', '_pred.png')
    out = str(ensure_output_dir() / base)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Static saved: {out}')

    # Top-5 概率
    print(f'\n  Prediction: [{pred_label}] {pred_name}')
    print(f'  True label: [{true_label}] {true_name}')
    print(f'  Top-5 probabilities:')
    top5 = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(top5):
        marker = ' ←' if idx == pred_label else ''
        print(f'    {rank+1}. [{idx}] {ACTION_NAMES[idx]:<15} {probs[idx]:.3f}{marker}')
    print()


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python show_predict.py <path/to/file.skeleton>')
        print('       uv run python show_predict.py <path/to/file.skeleton> --gif')
        sys.exit(1)

    filepath = sys.argv[1]
    save_gif = '--gif' in sys.argv

    if not os.path.exists(filepath):
        print(f'File not found: {filepath}')
        sys.exit(1)

    model = load_model()
    show_skeleton_static(filepath, model)
    show_skeleton_3d(filepath, model, save_gif=save_gif)


if __name__ == '__main__':
    main()
