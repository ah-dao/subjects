"""
基线测试可视化（阶段汇报用）。

基于现有结果文件生成 7 张图到 results/figures/：
  1. roc.png              ROC 曲线（5 折 + 平均 AUC）
  2. feature_importance.png   XGBoost 特征重要性 Top10（事件窗口 20 维）
  3. univariate_auc.png       单变量 AUC Top10
  4. k_sensitivity.png        K=1..6 敏感性（平均 AUC ± std 误差棒）
  5. baseline_compare.png     基线对比（静态 22 维 / 去 recent_2yr / 事件窗口 20 维）
  6. event_year_dist.png      正样本事件年 vs 负样本伪事件年分布（频率匹配）
  7. top10_recall.png         Recall@Top10% 对比（静态 vs 事件窗口，各折）

用法：
    python visualize_baseline.py [--features-csv features/event_window_features_k2_v30.csv]
                                 [--folds 5] [--seed 42] [--out results/figures]
依赖：results/ 下已有各实验 json（缺失的图自动跳过）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import STUDY_UNITS_COUNT_CSV, study_shp_path
from src.dataset import load_features, load_centroids, spatial_folds, fold_indices, \
    minmax_fit, minmax_apply


def setup_chinese_font():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP'):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            break
    return plt


def load_json(name):
    """优先 results/，历史结果回退 results/archive/。"""
    for base in (ROOT / 'results', ROOT / 'results' / 'archive'):
        p = base / name
        if p.exists():
            return json.load(open(p, encoding='utf-8'))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-csv', default=str(ROOT / 'features' / 'event_window_features_k2_v30.csv'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=str(ROOT / 'results' / 'figures'))
    args = parser.parse_args()
    plt = setup_chinese_font()

    feat_csv = Path(args.features_csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = [c for c in pd.read_csv(feat_csv, nrows=0).columns if c not in ('unit_id', 'label')]
    unit_id, X, y = load_features(feat_csv, features=feat_cols)
    centroids = load_centroids(study_shp_path())
    fold_id = spatial_folds(centroids, n_folds=args.folds, seed=args.seed)

    # ============ 1. ROC 曲线（重训 5 折收集验证集概率） ============
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    y_trues, y_scores = [], []
    print('重训 XGBoost 5 折收集 ROC 数据...')
    for fold in range(args.folds):
        tr_idx, va_idx = fold_indices(fold_id, fold)
        min_, max_ = minmax_fit(X[tr_idx])
        Xn = minmax_apply(X, min_, max_)
        m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric='auc', random_state=args.seed + fold, n_jobs=-1)
        m.fit(Xn[tr_idx], y[tr_idx])
        prob = m.predict_proba(Xn[va_idx])[:, 1]
        y_trues.append(y[va_idx])
        y_scores.append(prob)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for f, (yt, ys) in enumerate(zip(y_trues, y_scores)):
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = __import__('sklearn.metrics', fromlist=['roc_curve']).roc_curve(yt, ys)
        a = roc_auc_score(yt, ys)
        ax.plot(fpr, tpr, lw=1.0, alpha=0.55, label=f'Fold {f + 1} (AUC={a:.3f})')
    all_yt = np.concatenate(y_trues)
    all_ys = np.concatenate(y_scores)
    fpr, tpr, _ = __import__('sklearn.metrics', fromlist=['roc_curve']).roc_curve(all_yt, all_ys)
    mean_auc = roc_auc_score(all_yt, all_ys)
    ax.plot(fpr, tpr, lw=2.6, color='#C0392B',
            label=f'合并 OOF (AUC={mean_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('假阳性率 (FPR)'); ax.set_ylabel('真阳性率 (TPR)')
    ax.set_title('XGBoost 基线 ROC（事件窗口 19 维，5 折空间 K-Fold）')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / 'roc.png', dpi=150); plt.close(fig)
    print('已生成 roc.png')

    # ============ 2. 特征重要性（全部 20 个，按类别着色） ============
    from src.config import STATIC_TERRAIN_FEATURES, GEOMETRY_FEATURES
    CAT_COLORS = {'静态地形': '#2E86AB', '静态几何': '#16A085',
                  '淹没交互': '#E67E22', '事件前窗口': '#C0392B'}

    def category_of(f):
        if f.startswith(('k2_', 'ant_')) or f == 'wet_season_frac':
            return '事件前窗口'
        if f in STATIC_TERRAIN_FEATURES:
            return '静态地形'
        if f in GEOMETRY_FEATURES:
            return '静态几何'
        return '淹没交互'

    k2 = load_json('baseline_xgb_ew_feat_k2.json')
    if k2:
        imp = pd.DataFrame(k2['feature_importance']).sort_values('importance').iloc[::-1]
        # 保持数据顺序为 importance 降序，仅对画布用 bottom-up 排列
        imp = imp.iloc[::-1]
        colors = [CAT_COLORS[category_of(f)] for f in imp['feature']]
        fig, ax = plt.subplots(figsize=(9.5, 9.5))
        ax.barh(imp['feature'], imp['importance'], color=colors, edgecolor='white', linewidth=0.4)
        ax.set_xlabel('XGBoost 特征重要性')
        ax.set_title('特征重要性（全部 19 维；红=事件前窗口，橙=淹没，蓝=地形，绿=几何）')
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=lab) for lab, c in CAT_COLORS.items()]
        ax.legend(handles=handles, loc='lower right', fontsize=9)
        for idx, (f, v) in enumerate(zip(imp['feature'], imp['importance'])):
            ax.annotate(f'{v:.4f}', (v, idx), textcoords='offset points',
                        xytext=(3, 0), va='center', fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / 'feature_importance.png', dpi=150); plt.close(fig)
        print('已生成 feature_importance.png（全部 19 维，按类别着色）')

    # ============ 3. 单变量 AUC Top10 ============
    if k2 and k2.get('univariate_auc'):
        uni = pd.DataFrame(k2['univariate_auc']).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#C0392B' if 'k2_' in f else '#2E86AB' for f in uni['feature']]
        ax.barh(uni['feature'], uni['univar_auc'], color=colors)
        ax.axvline(0.5, color='k', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('单变量 AUC'); ax.set_title('单变量 AUC Top10（红=事件前窗口特征）')
        ax.set_xlim(0.45, 0.70)
        fig.tight_layout(); fig.savefig(out_dir / 'univariate_auc.png', dpi=150); plt.close(fig)
        print('已生成 univariate_auc.png')

    # ============ 4. K 敏感性 ============
    ks, aucs, stds = [], [], []
    for k in range(1, 7):
        d = load_json(f'baseline_xgb_ew_feat_k{k}.json')
        if d:
            ks.append(k); aucs.append(d['mean_auc']); stds.append(d['std_auc'])
    if ks:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(ks, aucs, yerr=stds, fmt='o-', capsize=5, color='#2E86AB',
                    ecolor='#AAAAAA', elinewidth=1.5, markersize=7)
        for x, y in zip(ks, aucs):
            ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=9)
        ax.axhline(0.6944, color='gray', ls='--', lw=1, label='静态全窗口 22 维对照 (0.694)')
        ax.set_xticks(ks)
        ax.set_xlabel('事件前窗口长度 K（年）'); ax.set_ylabel('平均 AUC')
        ax.set_title('K 敏感性分析（1997-2021 完整窗口，5 折空间 CV）')
        ax.legend(fontsize=9)
        ax.set_ylim(0.60, 0.75)
        fig.tight_layout(); fig.savefig(out_dir / 'k_sensitivity.png', dpi=150); plt.close(fig)
        print('已生成 k_sensitivity.png')

    # ============ 5. 基线对比 ============
    comp = [
        ('静态全窗口 18 维（历史对照）', 'baseline_xgb.json', '#95A5A6'),
        ('去 recent_2yr（历史消融）', 'baseline_xgb_excl_recent_2yr_ndvi_drop_recent_2yr_maxdaily.json', '#E67E22'),
        ('事件窗口 19 维 (K=2)', 'baseline_xgb_ew_feat_k2.json', '#C0392B'),
    ]
    rows = []
    for label, fname, color in comp:
        d = load_json(fname)
        if d:
            rows.append((label, d['mean_auc'], d['std_auc'], color))
    if rows:
        labels = [r[0] for r in rows]; means = [r[1] for r in rows]; stds = [r[2] for r in rows]
        colors = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.9, width=0.55)
        for b, m, s in zip(bars, means, stds):
            ax.annotate(f'{m:.4f}\n±{s:.4f}', (b.get_x() + b.get_width() / 2, m + s + 0.004),
                        ha='center', fontsize=9)
        ax.axhline(0.70, color='green', ls='--', lw=1.2, label='AUC=0.70 门槛')
        ax.set_ylabel('平均 AUC（5 折空间 CV）')
        ax.set_title('基线方案对比')
        ax.legend(fontsize=9)
        ax.set_ylim(0.60, 0.78)
        fig.tight_layout(); fig.savefig(out_dir / 'baseline_compare.png', dpi=150); plt.close(fig)
        print('已生成 baseline_compare.png')

    # ============ 6. 事件年 vs 伪事件年分布（频率匹配） ============
    su = pd.read_csv(STUDY_UNITS_COUNT_CSV)
    d = pd.to_datetime(su['study_first_landslide_date'], errors='coerce')
    pos_years = d.dropna().dt.year.astype(int)
    rng = np.random.RandomState(args.seed)
    neg_years = pd.Series(rng.choice(pos_years.values, size=(len(su) - len(pos_years)), replace=True))
    all_years = list(range(2003, 2022))
    pos_cnt = pos_years.value_counts().reindex(all_years, fill_value=0)
    neg_cnt = neg_years.value_counts().reindex(all_years, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    ax.bar([y - width / 2 for y in all_years], pos_cnt.values, width, label='正样本事件年 (662)', color='#C0392B')
    ax.bar([y + width / 2 for y in all_years], neg_cnt.values, width, label='负样本伪事件年 (25222)', color='#95A5A6')
    ax.set_xticks(all_years[::2]); ax.set_xticklabels(all_years[::2])
    ax.set_xlabel('年份'); ax.set_ylabel('单元数')
    ax.set_title('频率匹配：负样本伪事件年分布 = 正样本事件年分布（无泄漏关键）')
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / 'event_year_dist.png', dpi=150); plt.close(fig)
    print('已生成 event_year_dist.png')

    # ============ 7. Recall@Top10% 对比 ============
    static_d = load_json('baseline_xgb.json')
    if static_d and k2:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(5)
        ax.bar(x - 0.18, static_d['fold_recalls'], 0.36, label='静态全窗口 22 维', color='#95A5A6')
        ax.bar(x + 0.18, k2['fold_recalls'], 0.36, label='事件窗口 19 维 (K=2)', color='#C0392B')
        ax.set_xticks(x); ax.set_xticklabels([f'Fold {i + 1}' for i in range(5)])
        ax.set_ylabel('Recall@Top10%'); ax.set_title('Recall@Top10%（各折对比）')
        ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / 'top10_recall.png', dpi=150); plt.close(fig)
        print('已生成 top10_recall.png')

    print(f'\n全部图片已保存到: {out_dir}')


if __name__ == '__main__':
    main()
