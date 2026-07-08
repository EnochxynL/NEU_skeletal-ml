"""
可视化与评估工具
生成混淆矩阵图、对比柱状图、特征重要性分析
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Dict

ACTION_NAMES = [
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop',
    'pick up', 'throw', 'sit down', 'stand up', 'clapping'
]


def plot_confusion_matrices(results: List[Dict], save_path: str = 'confusion_matrices.png'):
    """
    为每个模型生成混淆矩阵子图。
    
    Args:
        results: list of dict, 每个含 'name', 'y_true', 'y_pred'
        save_path: 保存路径
    """
    n = len(results)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, r in enumerate(results):
        cm = confusion_matrix(r['y_true'], r['y_pred'], normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=ACTION_NAMES)
        disp.plot(ax=axes[i], cmap='Blues', xticks_rotation=45, values_format='.2f')
        axes[i].set_title(f'{r["name"]} (Acc: {r["test_acc"]:.3f})', fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'混淆矩阵已保存: {save_path}')


def plot_comparison(results: List[Dict], save_path: str = 'model_comparison.png'):
    """
    绘制模型对比柱状图（Test accuracy，如有 CV 也展示）。
    """
    names = [r['name'] for r in results]
    test_scores = [r['test_acc'] for r in results]
    has_cv = all('cv_score' in r for r in results)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    if has_cv:
        cv_scores = [r['cv_score'] for r in results]
        bars1 = ax.bar(x - width/2, cv_scores, width, label='CV Accuracy', color='steelblue')
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)

    bars2 = ax.bar(x + width/2 if has_cv else x, test_scores,
                   width if has_cv else 0.6, label='Test Accuracy', color='coral')

    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'模型对比图已保存: {save_path}')


def plot_feature_importance(model, feature_names: List[str] = None,
                            top_k: int = 20, save_path: str = 'feature_importance.png'):
    """
    绘制特征重要性（仅对 Random Forest / AdaBoost）。
    """
    if not hasattr(model, 'feature_importances_'):
        print('该模型不支持特征重要性')
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]

    if feature_names is None:
        feature_names = [f'F{i}' for i in range(len(importances))]

    top_names = [feature_names[i] for i in indices]
    top_imps = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_k), top_imps[::-1], color='steelblue')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'特征重要性已保存: {save_path}')


if __name__ == '__main__':
    print('可视化模块加载成功。在 train_ml.py 中调用这些函数。')
