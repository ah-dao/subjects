"""
XGBoost 基线（OPTIMIZATION_PATHS.md 8.2 阶段 2）。

用 28 维特征验证特征质量：5 折交叉验证输出 AUC，并检查水位特征是否排前列
（验证"消落带水位触发"假设）。AUC > 0.7 再继续上图模型。

用法：
    python baseline_xgb.py [--folds 5] [--method spatial_kmeans|random] [--seed 42]
输出：
    results/baseline_xgb.json   （各折 AUC、特征重要性）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (FEATURES_CSV, study_shp_path, RESULT_DIR,
                        K_FOLDS, FOLD_METHOD, SEED, ALL_FEATURES)
from src.dataset import (load_features, load_centroids,
                         spatial_folds, random_folds, fold_indices,
                         minmax_fit, minmax_apply)
from src.metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=K_FOLDS)
    parser.add_argument('--method', default=FOLD_METHOD,
                        choices=['spatial_kmeans', 'random'])
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f'未找到特征表: {FEATURES_CSV}（先运行 tills 提取脚本 + merge_features.py）')

    import xgboost as xgb

    unit_id, X, y = load_features(FEATURES_CSV)
    centroids = load_centroids(study_shp_path())
    assert len(unit_id) == len(centroids), '特征表与 shp 行数不一致'

    if args.method == 'spatial_kmeans':
        fold_id = spatial_folds(centroids, n_folds=args.folds, seed=args.seed)
    else:
        fold_id = random_folds(len(y), n_folds=args.folds, seed=args.seed)

    print(f'XGBoost 基线 | {args.folds} 折 | 方法: {args.method} | 正样本: {int(y.sum())}/{len(y)}')
    aucs, recalls, importances = [], [], []
    for fold in range(args.folds):
        tr_idx, va_idx = fold_indices(fold_id, fold)
        min_, max_ = minmax_fit(X[tr_idx])
        Xn = minmax_apply(X, min_, max_)
        model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric='auc', random_state=args.seed + fold,
                                  n_jobs=-1)
        model.fit(Xn[tr_idx], y[tr_idx])
        prob = model.predict_proba(Xn[va_idx])[:, 1]
        a, r = summarize(y[va_idx], prob)
        aucs.append(a)
        recalls.append(r)
        importances.append(model.feature_importances_)
        print(f'  Fold {fold + 1}: AUC {a:.4f} | Recall@Top10% {r:.4f}')

    valid = [a for a in aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float('nan')
    std_auc = float(np.std(valid)) if valid else float('nan')
    print(f'\n平均 AUC: {mean_auc:.4f} ± {std_auc:.4f}')

    imp_mean = np.mean(importances, axis=0)
    imp_df = pd.DataFrame({'feature': ALL_FEATURES, 'importance': imp_mean}).sort_values('importance', ascending=False)
    print('\n特征重要性 Top10:')
    print(imp_df.head(10).to_string(index=False))

    result = {
        'mean_auc': mean_auc, 'std_auc': std_auc,
        'fold_aucs': aucs, 'fold_recalls': recalls,
        'feature_importance': imp_df.to_dict('records'),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / 'baseline_xgb.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {out}')


if __name__ == '__main__':
    main()
