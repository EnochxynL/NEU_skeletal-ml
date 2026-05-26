"""
最终评估脚本 — 用最优超参数重新训练并生成可视化
"""
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from data_loader import load_dataset
from feature_extraction import extract_features_batch
from visualization import plot_confusion_matrices, plot_comparison

ACTION_NAMES = [
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop',
    'pick up', 'throw', 'sit down', 'stand up', 'clapping'
]

def main():
    data_dir = '../NEU_data/实验数据'
    print('加载数据/提取特征...')
    Xtr_seq, y_train, Xte_seq, y_test = load_dataset(data_dir)
    X_train = extract_features_batch(Xtr_seq)
    X_test = extract_features_batch(Xte_seq)
    print(f'特征: {X_train.shape}')

    models = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=120, random_state=42)),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
        ]),
        'SVM (Linear)': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=100, random_state=42)),
            ('clf', LinearSVC(C=0.01, dual=False, max_iter=5000, random_state=42)),
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1)),
        ]),
        'AdaBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=100, random_state=42)),
            ('clf', AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=3, random_state=42),
                n_estimators=300, learning_rate=1.0, random_state=42)),
        ]),
    }

    results = []
    for name, pipe in models.items():
        print(f'\n训练: {name}...', end=' ', flush=True)
        t0 = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f'Test Acc: {acc:.4f} ({elapsed:.1f}s)')
        results.append({
            'name': name, 'test_acc': acc, 'y_true': y_test, 'y_pred': pred,
            'best_model': pipe, 'time': elapsed
        })

    # 汇总
    print(f'\n{"="*55}')
    print(f'  {"Model":<18} {"Test Acc":>8}')
    print(f'  {"-"*26}')
    for r in results:
        print(f'  {r["name"]:<18} {r["test_acc"]:>8.4f}')
    print(f'{"="*55}')

    # 每个模型的分类报告
    for r in results:
        print(f'\n--- {r["name"]} ---')
        print(classification_report(r['y_true'], r['y_pred'],
              target_names=ACTION_NAMES, digits=3))

    # 生成可视化
    print('\n生成可视化...')
    plot_confusion_matrices(results, 'confusion_matrices.png')
    plot_comparison(results, 'model_comparison.png')
    print('Done!')

    return results


if __name__ == '__main__':
    main()
