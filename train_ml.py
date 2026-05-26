"""
训练管线: SVM / Random Forest / Adaboost
精简版 v2 — LinearSVC 代替 SVC(kernel='linear') 大幅加速
"""
import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from data_loader import load_dataset
from feature_extraction import extract_features_batch

ACTION_NAMES = [
    'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop',
    'pick up', 'throw', 'sit down', 'stand up', 'clapping'
]


def evaluate_model(name, pipe, param_grid, X_train, y_train, X_test, y_test, cv=5):
    print(f'\n{"="*55}\n  {name}\n{"="*55}', flush=True)

    t0 = time.time()
    search = GridSearchCV(
        pipe, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    cv_score = search.best_score_
    test_pred = search.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f'  CV: {cv_score:.4f} | Test: {test_acc:.4f} | '
          f'Time: {elapsed:.1f}s | Best: {search.best_params_}', flush=True)
    print(classification_report(y_test, test_pred, target_names=ACTION_NAMES, digits=3))

    return {
        'name': name, 'cv_score': cv_score, 'test_acc': test_acc,
        'best_params': search.best_params_, 'best_model': search.best_estimator_,
        'y_pred': test_pred, 'y_true': y_test, 'time': elapsed
    }


def main():
    data_dir = '../NEU_data/实验数据'
    print('=== 加载数据 ===')
    X_train_seq, y_train, X_test_seq, y_test = load_dataset(data_dir)
    print(f'Train: {len(X_train_seq)}, Test: {len(X_test_seq)}')

    print('=== 提取特征 ===')
    t0 = time.time()
    X_train = extract_features_batch(X_train_seq)
    X_test = extract_features_batch(X_test_seq)
    print(f'维度: {X_train.shape[1]}, 耗时: {time.time()-t0:.1f}s')
    assert not np.isnan(X_train).any()

    results = []

    # ---- 1. SVM RBF ----
    svm_rbf = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100, random_state=42)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42)),
    ])
    svm_rbf_grid = {
        'pca__n_components': [80, 120],
        'clf__C': [1, 10, 100],
        'clf__gamma': ['scale', 0.01],
    }
    results.append(evaluate_model('SVM (RBF)', svm_rbf, svm_rbf_grid,
                                   X_train, y_train, X_test, y_test))

    # ---- 2. SVM Linear (LinearSVC — 快很多) ----
    svm_lin = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100, random_state=42)),
        ('clf', LinearSVC(dual=False, max_iter=3000, random_state=42)),
    ])
    svm_lin_grid = {
        'pca__n_components': [60, 100, 150],
        'clf__C': [0.01, 0.1, 1, 10],
    }
    results.append(evaluate_model('SVM (Linear)', svm_lin, svm_lin_grid,
                                   X_train, y_train, X_test, y_test))

    # ---- 3. Random Forest ----
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])
    rf_grid = {
        'clf__n_estimators': [100, 300],
        'clf__max_depth': [10, 20, 30],
        'clf__min_samples_split': [2, 5],
    }
    results.append(evaluate_model('Random Forest', rf, rf_grid,
                                   X_train, y_train, X_test, y_test))

    # ---- 4. AdaBoost ----
    base_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    ada = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', AdaBoostClassifier(estimator=base_dt, random_state=42)),
    ])
    ada_grid = {
        'clf__n_estimators': [100, 300, 500],
        'clf__learning_rate': [0.1, 0.5, 1.0],
    }
    results.append(evaluate_model('AdaBoost', ada, ada_grid,
                                   X_train, y_train, X_test, y_test))

    # ---- 汇总 ----
    print(f'\n{"="*55}')
    print(f'  {"Model":<18} {"CV":>6} {"Test":>6} {"Time":>6}')
    print(f'  {"-"*36}')
    best_name, best_acc = '', 0
    for r in results:
        print(f'  {r["name"]:<18} {r["cv_score"]:.4f} {r["test_acc"]:.4f} '
              f'{r["time"]:>5.0f}s')
        if r['test_acc'] > best_acc:
            best_acc = r['test_acc']
            best_name = r['name']
    print(f'  {"-"*36}')
    print(f'  Best: {best_name} ({best_acc:.4f})')
    print(f'{"="*55}')

    return results


if __name__ == '__main__':
    main()
