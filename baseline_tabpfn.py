"""
TabPFN v2 基线（表格基础模型线：与 baseline_xgb.py **同口径**对照）。

2026-10 改版要点（v2 主线）：
  1. 不再硬编码 v1 特征表：新增 --features-csv / --features，默认用 38 维主线
     `features/v2/features_v2_train_38.csv`（不存在时回退 34 维 v2 表）。
  2. 新增 --context matched：病例-对照匹配上下文（与 baseline_xgb --neg-sampling matched 同一套规则），
     并输出与 XGB 一致的平衡诊断（SMD / 残留混杂 AUC），供"上下文构造 ↔ 目标人群等价性"（方案 T1）使用。
  3. JSON 结构对齐 baseline_xgb（fold_aucs / pool / recalls / used_features / match_diag），
     便于 tools/paired_compare.py 直接做配对检验。
  4. 输出落到 results/runs/tabpfn/（不再污染 results/ 顶层）。

关键差异（不可回避）：TabPFN 的 fit() 不支持 sample_weight，因此"软负采样"用
**等效子采样**实现——正样本全保留 + 4km 内负样本全保留（权重 1.0）+ 远区负样本按 λ 概率保留
（importance weighting → resampling）。
CPU 实测（16 线程，38 维）：上下文 1000 时预测 400 行约 32s、上下文 2000 时约 89s
→ **正式实验必须上 GPU**；CPU 只用于冒烟测试。

用法：
    python baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv \
        --method spatial_kmeans --context soft --seeds 3
    python baseline_tabpfn.py --context matched --match-k 5 --method spatial_kmeans --seeds 3
    python baseline_tabpfn.py --smoke          # 极小子集冒烟（1 折 1 种子，CPU 可跑）
输出：
    results/runs/tabpfn/baseline_tabpfn_<tags>.json
依赖：
    pip install tabpfn        # torch>=2.5；国内可 export HF_ENDPOINT=https://hf-mirror.com
    （或离线：权重已放在 models/tabpfn/，用 --model-path 指向本地 ckpt，并 export HF_HUB_OFFLINE=1）
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.deps'))          # 本地离线依赖（沙箱内 pip 不通时的兜底）

from src.config import (FEATURES_CSV, FEATURES_CSV_38, EVENT_WINDOW_FEATURES_CSV,
                        FEATURES_DIR, study_shp_path, RESULT_DIR, COUNTY_UNITS_CSV, SEED)
from src.dataset import (load_features, load_centroids_utm, admin_folds, spatial_folds,
                         random_folds, fold_indices, proximity_mask, matched_negatives)
from src.metrics import summarize
from sklearn.metrics import roc_auc_score

RUNS_DIR = Path(RESULT_DIR) / 'runs' / 'tabpfn'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-csv', default=None,
                    help='特征表；默认 38 维主线 features/v2/features_v2_train_38.csv')
    ap.add_argument('--features', default='', help='逗号分隔的显式特征列（默认取表中全部非 unit_id/label 列）')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--method', default='admin', choices=['admin', 'spatial_kmeans', 'random'])
    # ---- 上下文构造 ----
    ap.add_argument('--context', default='soft', choices=['soft', 'matched', 'full'],
                    help='soft=软采样等效子采样；matched=病例-对照匹配；full=全域（仅对照用）')
    ap.add_argument('--neg-km', type=float, default=4.0, help='soft: 邻域半径（km）')
    ap.add_argument('--neg-lam', type=float, default=0.2, help='soft: 远区负样本保留概率（等效 λ）')
    ap.add_argument('--match-radius-km', type=float, default=4.0, help='matched: 缓冲区半径')
    ap.add_argument('--match-k', type=int, default=5, help='matched: 每个正样本匹配的负样本数')
    ap.add_argument('--match-elev-bin', type=float, default=20.0)
    ap.add_argument('--match-slope-bin', type=float, default=10.0)
    ap.add_argument('--match-far-frac', type=float, default=0.0)
    # ---- 规模与运行 ----
    ap.add_argument('--max-train', type=int, default=12000, help='上下文上限（TabPFN 预训练 ~1 万行）')
    ap.add_argument('--seeds', type=int, default=3, help='上下文子采样重复次数（提高检验力用；可到 10-20）')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--ignore-limits', action='store_true', help='放开 TabPFN 规模限制')
    ap.add_argument('--device', default=None, help='默认自动：cuda 可用则 cuda')
    ap.add_argument('--n-estimators', type=int, default=None, help='TabPFN 内部集成数（默认库默认；调小可加速）')
    ap.add_argument('--model-path', default=None,
                    help='权重路径；默认用 models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt（本地已备）')
    ap.add_argument('--smoke', action='store_true', help='冒烟：1 折 1 种子 + 小上下文')
    ap.add_argument('--save-oof', default='',
                    help='逐单元 OOF 概率写到该 CSV（用于集成/校准/共形预测）；多种子时同时给种子间标准差作为单元级不确定性')
    args = ap.parse_args()

    csv_path = Path(args.features_csv) if args.features_csv else (
        FEATURES_CSV_38 if Path(FEATURES_CSV_38).exists() else Path(EVENT_WINDOW_FEATURES_CSV))
    if not csv_path.exists():
        raise FileNotFoundError(f'未找到特征表: {csv_path}')
    if args.features:
        use_features = [f.strip() for f in args.features.split(',') if f.strip()]
    elif args.features_csv or csv_path != Path(FEATURES_CSV):
        use_features = [c for c in pd.read_csv(csv_path, nrows=0).columns
                        if c not in ('unit_id', 'label')]
    else:
        use_features = None
    if args.smoke:
        args.folds, args.seeds, args.max_train = 2, 1, 1200

    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        import tabpfn as _tp
        tpv = getattr(_tp, '__version__', '?')
    except Exception:
        tpv = '?'
    print(f'[版本] tabpfn {tpv} | torch {torch.__version__} | cuda {torch.cuda.is_available()}'
          f'  ← 比较实验必须固定 tabpfn 版本（2.x 与 8.x 的默认集成数不同，数值会有小差异）')
    ckpt = args.model_path or str(ROOT / 'models' / 'tabpfn' /
                                  'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
    if not os.path.exists(ckpt) and args.model_path is None:
        ckpt = 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'      # 交给 tabpfn 自己从 HF 取
    print(f'[TabPFN] device={device} | 上下文模式 {args.context} | 上限 {args.max_train} '
          f'| 种子重复 {args.seeds} | 权重 {ckpt}')
    print(f'[特征表] {csv_path.name} | {len(use_features) if use_features else "?"} 维')

    unit_id, X, y = load_features(csv_path, features=use_features)
    used = use_features or [c for c in pd.read_csv(csv_path, nrows=0).columns
                            if c not in ('unit_id', 'label')]
    cent_utm = load_centroids_utm(study_shp_path())
    assert len(unit_id) == len(cent_utm), '特征表与 shp 行数不一致'
    print(f'单元 {len(y)} | 正样本 {int(y.sum())}')

    # ---- 匹配口径所需的分层键与混杂变量 ----
    match_keys = m_elev = m_slope = m_inund = None
    if args.context == 'matched':
        aux = pd.DataFrame({'unit_id': unit_id})
        _t = pd.read_csv(FEATURES_DIR / 'v2' / 'groups' / 'terrain.csv',
                         usecols=['unit_id', 'elevation_mean', 'slope_mean'])
        _w = pd.read_csv(FEATURES_DIR / 'v2' / 'groups' / 'water.csv',
                         usecols=['unit_id', 'inundation_fraction'])
        aux = aux.merge(_t, on='unit_id', how='left').merge(_w, on='unit_id', how='left')
        m_elev = aux['elevation_mean'].values.astype(np.float64)
        m_slope = aux['slope_mean'].values.astype(np.float64)
        m_inund = aux['inundation_fraction'].values.astype(np.float64)
        for v in (m_elev, m_slope, m_inund):
            bad = ~np.isfinite(v)
            if bad.any():
                v[bad] = np.nanmean(v)
        match_keys = (np.floor(m_elev / args.match_elev_bin).astype('int64') * 100000
                      + np.floor(m_slope / args.match_slope_bin).astype('int64'))
        print(f'[匹配分层] 高程 {args.match_elev_bin:g}m × 坡度 {args.match_slope_bin:g}° | '
              f'半径 {args.match_radius_km:g}km | k={args.match_k} | 全域占比 {args.match_far_frac:g}')

    if args.method == 'admin':
        fold_id = admin_folds(COUNTY_UNITS_CSV, unit_id, y, n_folds=args.folds)
    elif args.method == 'spatial_kmeans':
        fold_id = spatial_folds(cent_utm, n_folds=args.folds, seed=args.seed)
    else:
        fold_id = random_folds(len(y), n_folds=args.folds, seed=args.seed)

    all_aucs, all_pools, all_recalls, all_ctx, per_seed, match_diag = [], [], [], [], [], []
    oof_all = []                                              # 每种子的逐单元 OOF（用于集成/不确定性）
    for s in range(args.seeds):
        rng = np.random.RandomState(args.seed + 1000 * s)
        aucs, pools, recalls = [], [], []
        oof_seed = np.full(len(y), np.nan) if args.save_oof else None
        for fold in range(args.folds):
            tr_idx, va_idx = fold_indices(fold_id, fold)
            if args.smoke:                       # 冒烟：验证集也要缩小（CPU 上每行预测约 0.08~0.2s）
                _rs = np.random.RandomState(0)
                _vp = va_idx[y[va_idx] == 1]
                _vn = va_idx[y[va_idx] == 0]
                _vn = _rs.choice(_vn, size=min(300, len(_vn)), replace=False)
                va_idx = np.concatenate([_vp, _vn]).astype(int)
            tr_pos = tr_idx[y[tr_idx] == 1]
            va_pos = va_idx[y[va_idx] == 1]

            if args.context == 'matched':
                # ---------- 病例-对照匹配上下文（规则与 baseline_xgb 完全一致）----------
                picked, st = matched_negatives(
                    cent_utm, y, tr_pos, match_keys, d_km=args.match_radius_km,
                    exclude_m=150.0, k=args.match_k, far_frac=args.match_far_frac,
                    seed=42 + fold, candidate_idx=tr_idx)
                keep = np.concatenate([tr_pos, picked]).astype(int)
                if len(keep) > args.max_train:                  # 极端情况下才降采样负样本
                    neg = picked[rng.permutation(len(picked))[:args.max_train - len(tr_pos)]]
                    keep = np.concatenate([tr_pos, neg]).astype(int)
                if len(picked) > 0:                             # 平衡诊断（与 XGB 同构）
                    def _smd(a, b):
                        a, b = np.asarray(a, float), np.asarray(b, float)
                        return float((a.mean() - b.mean()) /
                                     np.sqrt((a.var() + b.var()) / 2 + 1e-12))
                    smd = {'elevation': _smd(m_elev[tr_pos], m_elev[picked]),
                           'slope': _smd(m_slope[tr_pos], m_slope[picked]),
                           'inundation': _smd(m_inund[tr_pos], m_inund[picked])}
                    from sklearn.linear_model import LogisticRegression
                    Zm = np.column_stack([m_elev[keep], m_slope[keep], m_inund[keep]])
                    lr = LogisticRegression(max_iter=1000).fit(Zm, y[keep])
                    ca = float(roc_auc_score(y[keep], lr.predict_proba(Zm)[:, 1]))
                    if s == 0:
                        match_diag.append({'fold': fold, 'smd': smd, 'confound_auc': ca,
                                           'n_train': int(len(keep)), 'n_neg': int(len(picked))})
                    if s == 0 and fold == 0:
                        print(f'    [匹配] 池内候选 {st["core_pool"]} | 匹配成功正样本 '
                              f'{st["matched_pos"]}/落空 {st["dropped_pos"]} | 负样本 {st["matched"]}'
                              f'+{st["far"]} | SMD 高程 {smd["elevation"]:+.3f} 坡度 {smd["slope"]:+.3f} '
                              f'淹没 {smd["inundation"]:+.3f} | 混杂AUC {ca:.4f}')
                near_neg = far_neg = keep_far = near_keep = np.array([], dtype=int)
            elif args.context == 'full':
                keep = tr_idx.astype(int)
                near_neg = far_neg = keep_far = near_keep = np.array([], dtype=int)
            else:
                # ---------- 软采样等效子采样（正样本全保留 + 近区全留 + 远区按 λ）----------
                near_tr = proximity_mask(cent_utm, y, tr_pos, args.neg_km, candidate_idx=tr_idx)
                neg_tr = tr_idx[y[tr_idx] == 0]
                is_near = near_tr[neg_tr]
                near_neg, far_neg = neg_tr[is_near], neg_tr[~is_near]
                keep_far = far_neg[rng.rand(len(far_neg)) < args.neg_lam]
                near_keep = near_neg
                keep = np.concatenate([tr_pos, near_neg, keep_far]).astype(int)
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
                        print(f'    [警告] 上下文 {len(keep)} 仍超 {args.max_train}')
            all_ctx.append(len(keep))

            from tabpfn import TabPFNClassifier
            kwargs = dict(device=device, random_state=int(args.seed + fold * 100 + s))
            if os.path.exists(ckpt):
                kwargs['model_path'] = ckpt
            if args.n_estimators:
                kwargs['n_estimators'] = args.n_estimators
            cpu_large = (device == 'cpu' and len(keep) > 1000)
            kwargs['ignore_pretraining_limits'] = bool(args.ignore_limits or cpu_large
                                                       or len(keep) > 10000)
            # 兼容两代 API：tabpfn 2.x 要求 torch>=2.1，6.x/8.x 要求 torch>=2.5。
            # 只传构造器签名里存在的参数，避免在小版本上 TypeError。
            try:
                import inspect
                _sig = set(inspect.signature(TabPFNClassifier.__init__).parameters)
                dropped = [k for k in list(kwargs) if k not in _sig]
                for k in dropped:
                    kwargs.pop(k)
                if dropped and not globals().get('_warned_dropped'):
                    print('  [兼容] 当前 tabpfn 版本不支持参数 %s，已忽略（2.x 与 8.x 的差异）'
                          % dropped)
                    globals()['_warned_dropped'] = True
            except Exception:
                pass
            clf = TabPFNClassifier(**kwargs)
            clf.fit(X[keep], y[keep])
            prob = clf.predict_proba(X[va_idx])[:, 1]
            if args.save_oof:                                  # 收集逐单元 OOF 概率
                oof_seed[va_idx] = prob
            a, r = summarize(y[va_idx], prob)
            aucs.append(float(a)); recalls.append(float(r))

            near_va = proximity_mask(cent_utm, y, va_pos, args.neg_km, candidate_idx=va_idx)
            pool_mask = np.zeros(len(y), dtype=bool)
            pool_mask[va_pos] = True
            pool_mask[near_va] = True
            pool_pos = np.flatnonzero(pool_mask[va_idx])
            pools.append(float(roc_auc_score(y[va_idx][pool_pos], prob[pool_pos])))
            print(f'  [seed {s}] Fold {fold + 1}: 全单元 AUC {a:.4f} | 池 AUC {pools[-1]:.4f}'
                  f' | recall@10% {r:.4f} | 上下文 {len(keep)}（正 {len(tr_pos)}'
                  f' / 近区负 {len(near_neg)}→{len(near_keep)} / 远区负 {len(far_neg)}→{len(keep_far)}）')
        per_seed.append({'seed_index': s, 'mean_auc': float(np.mean(aucs)),
                         'mean_auc_pool': float(np.mean(pools)),
                         'mean_recall': float(np.mean(recalls))})
        all_aucs += aucs; all_pools += pools; all_recalls += recalls
        if oof_seed is not None:
            oof_all.append(oof_seed)
        if args.smoke:
            break

    if args.save_oof and oof_all:
        M = np.vstack(oof_all)
        p_mean, p_sd = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
        out_p = args.save_oof if os.path.isabs(args.save_oof) else os.path.join(ROOT, args.save_oof)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        pd.DataFrame({'unit_id': unit_id, 'label': y, 'p_tabpfn': p_mean,
                      'p_tabpfn_sd': p_sd}).to_csv(out_p, index=False, encoding='utf-8-sig')
        print(f'[OOF] 已写出 {out_p}（{len(y)} 单元；p_tabpfn=种子均值，'
              f'p_tabpfn_sd=种子间标准差，可作单元级不确定性）')

    result = {
        'model': 'TabPFN-v2', 'model_path': ckpt, 'device': device,
        'tabpfn_version': tpv, 'torch_version': torch.__version__,
        'fold_method': args.method, 'folds': args.folds,
        'features_csv': str(csv_path), 'n_features': X.shape[1], 'used_features': used,
        'context_mode': args.context, 'max_train': args.max_train, 'seeds': args.seeds,
        'neg_km': args.neg_km, 'neg_lam': args.neg_lam,
        'match_radius_km': args.match_radius_km, 'match_k': args.match_k,
        'match_elev_bin': args.match_elev_bin, 'match_slope_bin': args.match_slope_bin,
        'match_far_frac': args.match_far_frac,
        'n_context_mean': float(np.mean(all_ctx)),
        'fold_aucs': all_aucs, 'fold_aucs_pool': all_pools, 'fold_recalls': all_recalls,
        'mean_auc': float(np.mean(all_aucs)), 'std_auc': float(np.std(all_aucs)),
        'mean_auc_pool': float(np.mean(all_pools)), 'std_auc_pool': float(np.std(all_pools)),
        'mean_recall': float(np.mean(all_recalls)),
        'per_seed': per_seed, 'match_diag': match_diag,
    }
    tag = ''
    if args.method != 'spatial_kmeans':
        tag += f'_m{args.method}'
    if args.context == 'matched':
        tag += f'_mt{args.match_k}f{int(round(args.match_far_frac * 100))}'
    elif args.context == 'full':
        tag += '_full'
    else:
        tag += f'_soft{args.neg_lam}'
    tag += f'_{X.shape[1]}d_s{args.seeds}'
    if args.smoke:
        tag += '_smoke'
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f'baseline_tabpfn{tag}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n===== TabPFN 汇总（{args.seeds} 种子 × {args.folds} 折，{args.context} 上下文）=====')
    print(f'平均 AUC（全单元）: {result["mean_auc"]:.4f} ± {result["std_auc"]:.4f}')
    print(f'平均 AUC（采样池）: {result["mean_auc_pool"]:.4f} ± {result["std_auc_pool"]:.4f}')
    print(f'recall@Top10%     : {result["mean_recall"]:.4f}')
    print(f'平均上下文规模    : {result["n_context_mean"]:.0f}')
    print(f'结果已保存: {out}')


if __name__ == '__main__':
    main()
