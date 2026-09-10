"""
TabPFN v2 基线（B1：表格基础模型对照，协议与 baseline_xgb.py 完全对齐）。

关键差异处理：TabPFN 的 fit() 不支持 sample_weight（API 已核对），因此"软采样"
以等效子采样实现——全部正样本保留 + 4km 内负样本全保留（权重 1.0）+ 远区负样本按
λ 概率保留（权重 0.2 的重采样等价，importance weighting → resampling）。
上下文上限 max_train（TabPFN 预训练限制 ~1 万行）内再随机降采样远区。

用法：
    python baseline_tabpfn.py                       # 默认 admin×soft(4km, λ=0.2)，3 种子
    python baseline_tabpfn.py --seeds 5 --neg-lam 0.2
输出：
    results/baseline_tabpfn_madmin_soft0.2.json（双 AUC + recall，与 XGB 同构）
依赖：
    pip install tabpfn        （torch>=2.5，pip 会自动解析；国内先 export HF_ENDPOINT=https://hf-mirror.com）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES_CSV, EVENT_WINDOW_FEATURES,
                        study_shp_path, RESULT_DIR, COUNTY_UNITS_CSV, SEED)
from src.dataset import load_features, load_centroids_utm, admin_folds, \
    spatial_folds, random_folds, fold_indices, proximity_mask
from src.metrics import summarize
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--method', default='admin', choices=['admin', 'spatial_kmeans', 'random'])
    ap.add_argument('--neg-km', type=float, default=4.0, help='软采样邻域半径（km）')
    ap.add_argument('--neg-lam', type=float, default=0.2, help='远区负样本保留概率（等效 λ）')
    ap.add_argument('--max-train', type=int, default=12000,
                    help='TabPFN 上下文上限（含正样本；实测近区负样本可达 ~9k，勿低于 11k）')
    ap.add_argument('--seeds', type=int, default=3, help='子采样重复次数（报告 mean±std）')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--ignore-limits', action='store_true',
                    help='放开 TabPFN 预训练规模限制（>10k 上下文时用）')
    ap.add_argument('--device', default=None, help='默认自动：cuda 可用则 cuda')
    ap.add_argument('--model-path', default='tabpfn-v2-classifier-finetuned-zk73skhh.ckpt',
                    help='权重文件名；默认为 TabPFN v2 官方权重（Nature 2025 版本，免 HF license 门控；'
                         '勿用含 "v3"/"v2.5" 字样的名字，那些是 gated 仓库）')
    args = ap.parse_args()

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    from tabpfn import TabPFNClassifier
    print(f'[TabPFN] device={device} | 上下文上限 {args.max_train} | '
          f'等效软采样 {args.neg_km}km/λ={args.neg_lam} | 种子重复 {args.seeds}')

    unit_id, X, y = load_features(EVENT_WINDOW_FEATURES_CSV, features=list(EVENT_WINDOW_FEATURES))
    cent_utm = load_centroids_utm(study_shp_path())
    n_pos_total = int(y.sum())
    print(f'特征表: {len(unit_id)} 单元 × {X.shape[1]} 维 | 正样本 {n_pos_total}')

    if args.method == 'admin':
        fold_id = admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=args.folds)
    elif args.method == 'spatial_kmeans':
        fold_id = spatial_folds(cent_utm, n_folds=args.folds, seed=args.seed)
    else:
        fold_id = random_folds(len(y), n_folds=args.folds, seed=args.seed)

    all_aucs, all_pools, all_recalls, all_ctx = [], [], [], []
    per_seed = []
    for s in range(args.seeds):
        rng = np.random.RandomState(args.seed + 1000 * s)
        aucs, pools, recalls = [], [], []
        for fold in range(args.folds):
            tr_idx, va_idx = fold_indices(fold_id, fold)
            tr_pos = tr_idx[y[tr_idx] == 1]
            va_pos = va_idx[y[va_idx] == 1]

            # ---- 软采样的子采样等价（按折计算，防跨折泄漏）----
            near_tr = proximity_mask(cent_utm, y, tr_pos, args.neg_km, candidate_idx=tr_idx)
            neg_tr = tr_idx[y[tr_idx] == 0]
            is_near = near_tr[neg_tr]
            near_neg, far_neg = neg_tr[is_near], neg_tr[~is_near]
            keep_far = far_neg[rng.rand(len(far_neg)) < args.neg_lam]
            near_keep, far_keep = near_neg, keep_far
            keep = np.concatenate([tr_pos, near_neg, keep_far]).astype(int)

            # 上下文保护：正样本全保留 → 近区负样本尽量保留 → 远区按 λ 保留；
            # 仍超限时才对近区按比例精确降采样（正样本绝不动）
            if len(keep) > args.max_train:
                budget = max(args.max_train - len(tr_pos), 0)
                n_near_keep = min(len(near_neg), budget)
                near_keep = near_neg if n_near_keep == len(near_neg) else \
                    near_neg[rng.permutation(len(near_neg))[:n_near_keep]]
                n_far_keep = min(len(keep_far), max(budget - n_near_keep, 0))
                far_keep = keep_far if n_far_keep == len(keep_far) else \
                    rng.choice(keep_far, size=n_far_keep, replace=False)
                keep = np.concatenate([tr_pos, near_keep, far_keep]).astype(int)
                if len(keep) > args.max_train:
                    print(f'    [警告] 上下文 {len(keep)} 仍超 {args.max_train}'
                          '（近区负样本过多）——建议调大 --max-train 或用 --ignore-limits')
            all_ctx.append(len(keep))

            clf = TabPFNClassifier(device=device, random_state=int(args.seed + fold),
                                   model_path=args.model_path,
                                   ignore_pretraining_limits=bool(args.ignore_limits or len(keep) > 10000),
                                   show_progress_bar=False)
            clf.fit(X[keep], y[keep])
            prob = clf.predict_proba(X[va_idx])[:, 1]
            a, r = summarize(y[va_idx], prob)
            aucs.append(a); recalls.append(r)

            # 采样池 AUC：验证折正样本 + 其 4km 邻域（与 baseline_xgb 口径一致）
            near_va = proximity_mask(cent_utm, y, va_pos, args.neg_km, candidate_idx=va_idx)
            pool_mask = np.zeros(len(y), dtype=bool)
            pool_mask[va_pos] = True
            pool_mask[near_va] = True
            pool_pos = np.flatnonzero(pool_mask[va_idx])
            pools.append(float(roc_auc_score(y[va_idx][pool_pos], prob[pool_pos])))

            print(f'  [seed {s}] Fold {fold + 1}: 全单元 AUC {a:.4f} | 池 AUC {pools[-1]:.4f} '
                  f'| recall@10% {r:.4f} | 上下文 {len(keep)}（正 {len(tr_pos)} '
                  f'/ 近区负 {len(near_neg)}→{len(near_keep)} '
                  f'/ 远区负 {len(far_neg)}→{len(far_keep)}）')
        per_seed.append({'seed_index': s, 'mean_auc': float(np.mean(aucs)),
                         'mean_auc_pool': float(np.mean(pools)),
                         'mean_recall': float(np.mean(recalls))})
        all_aucs += aucs; all_pools += pools; all_recalls += recalls

    result = {
        'model': 'TabPFN-v2',
        'model_path': args.model_path,
        'fold_method': args.method, 'folds': args.folds,
        'neg_km': args.neg_km, 'neg_lam': args.neg_lam,
        'max_train': args.max_train, 'seeds': args.seeds,
        'n_context_mean': float(np.mean(all_ctx)),
        'n_features': X.shape[1],
        'fold_aucs': all_aucs, 'fold_aucs_pool': all_pools, 'fold_recalls': all_recalls,
        'mean_auc': float(np.mean(all_aucs)), 'std_auc': float(np.std(all_aucs)),
        'mean_auc_pool': float(np.mean(all_pools)), 'std_auc_pool': float(np.std(all_pools)),
        'mean_recall': float(np.mean(all_recalls)),
        'per_seed': per_seed,
    }
    tag = ''
    if args.method != 'spatial_kmeans':
        tag += f'_m{args.method}'
    tag += f'_soft{args.neg_lam}'
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f'baseline_tabpfn{tag}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n===== TabPFN 汇总（{args.seeds} 种子 × {args.folds} 折）=====')
    print(f'平均 AUC（全单元）: {result["mean_auc"]:.4f} ± {result["std_auc"]:.4f}')
    print(f'平均 AUC（采样池）: {result["mean_auc_pool"]:.4f} ± {result["std_auc_pool"]:.4f}')
    print(f'recall@Top10%     : {result["mean_recall"]:.4f}')
    print(f'平均上下文规模    : {result["n_context_mean"]:.0f}')
    print(f'结果已保存: {out}')


if __name__ == '__main__':
    main()
