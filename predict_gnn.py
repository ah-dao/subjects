"""
全图推理 + 矢量出图（PROJECT_OVERVIEW.md，方案 C：训练人群 + 水下单元回填）。

加载最终模型 → 对训练图（25636 节点）前向 → 概率 [0,1] → 5 级易发性。
常年水下单元（432 个，未参与训练/不在图中）回填 prob=0、level=1（极低易发），
保证全量 26068 每个斜坡单元都有概率值与等级（出图无空缺）。

用法：
    python predict_gnn.py --plan B [--method fixed|quantile]
                          [--model models/best_B.pth --scaler models/scaler_B.npz]
输出：
    predictions/susceptibility_units_full.shp   （26068 全量，含 ls_prob/ls_level/is_submerged）
    predictions/gnn_probabilities.csv           （26068 全量表格：unit_id/prob/level）
    predictions/statistics.txt                  （各等级单元数统计）
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES_CSV, EVENT_WINDOW_FEATURES,
                        GRAPH_NPZ, study_shp_path, MODEL_DIR,
                        PRED_DIR, LEVEL_THRESHOLDS, LEVEL_NAMES,
                        PLAN, HIDDEN_DIM, NUM_HEADS, TRANSFORMER_LAYERS, DROPOUT)
from src.dataset import load_features, load_graph, minmax_apply
from src.model import build_model

# 水下单元名单 + 回填映射（由 rebuild_train_population.py 生成）
SUBMERGED_CSV = ROOT / 'features' / 'submerged_units_combined.csv'
BACKFILL_MAP = ROOT / 'features' / 'train_backfill_map.csv'
FULL_SHP = ROOT / 'data' / 'slope_units' / 'slope_units_fixed.shp'   # 全量 26068


def levels_from_probs(prob, method='fixed'):
    """按概率分 5 级：fixed=固定阈值，quantile=每级 20% 单元。返回 0..4。"""
    if method == 'fixed':
        thr = np.array([-np.inf] + LEVEL_THRESHOLDS + [np.inf])
        return np.digitize(prob, thr[1:-1])
    q = np.quantile(prob, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(prob, q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN, choices=['A', 'B', 'C'])
    parser.add_argument('--method', default='fixed', choices=['fixed', 'quantile'])
    parser.add_argument('--model', default=None)
    parser.add_argument('--scaler', default=None)
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else MODEL_DIR / f'best_{args.plan}.pth'
    scaler_path = Path(args.scaler) if args.scaler else MODEL_DIR / f'scaler_{args.plan}.npz'
    if not model_path.exists():
        raise FileNotFoundError(f'未找到模型: {model_path}（先运行 python train_gnn.py --plan {args.plan}）')
    if not scaler_path.exists():
        raise FileNotFoundError(f'未找到归一化参数: {scaler_path}')

    # ---------- 1. 加载训练人群数据（25636） ----------
    unit_id_tr, X_tr, _ = load_features(EVENT_WINDOW_FEATURES_CSV, features=EVENT_WINDOW_FEATURES)
    edge_index = load_graph(GRAPH_NPZ)
    print(f'训练/推理人群: {len(unit_id_tr)}（图节点 {edge_index.max() + 1}）')

    sc = np.load(scaler_path)
    Xn = minmax_apply(X_tr, sc['min_'], sc['max_'])
    ckpt = torch.load(model_path, map_location='cpu')
    model = build_model(args.plan, input_dim=ckpt['input_dim'],
                        num_nodes=ckpt['num_nodes'],
                        hidden_dim=ckpt.get('hidden_dim', HIDDEN_DIM),
                        num_heads=ckpt.get('num_heads', NUM_HEADS),
                        num_layers=ckpt.get('num_layers', TRANSFORMER_LAYERS),
                        dropout=ckpt.get('dropout', DROPOUT))
    model.load_state_dict(ckpt['model_state'])

    from src.train import predict_all, DEVICE
    model.to(DEVICE)
    x_t = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    edge_t = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
    prob_tr = predict_all(model, x_t, edge_t)          # 25636 个（训练行序）
    print(f'模型推理完成: {len(prob_tr)} 个单元（仅可滑坡坡体）')

    # ---------- 2. 水下单元回填 + 全量映射 ----------
    bm = pd.read_csv(BACKFILL_MAP)
    bm['orig_id'] = bm['orig_id'].astype(str)
    # Id 为 NaN 的水下单元不参与训练输出 join（merge 后 prob 保持 NaN → fillna(0)）
    tr_df = pd.DataFrame({'train_id': np.arange(1, len(prob_tr) + 1),
                          'prob': prob_tr})
    tr_df['train_id'] = tr_df['train_id'].astype(float)
    full = bm.merge(tr_df, left_on='Id', right_on='train_id', how='left')
    full['prob'] = full['prob'].fillna(0.0)           # 水下单元无模型输出 → 0
    full['is_submerged'] = full['is_submerged'].astype(int)
    n_sub = int(full['is_submerged'].sum())
    print(f'水下单元回填 prob=0: {n_sub}')

    # ---------- 3. 分级（只在非水下单元概率上算断点） ----------
    active = full.loc[full['is_submerged'] == 0, 'prob'].values
    if args.method == 'quantile':
        q = np.quantile(active, [0.2, 0.4, 0.6, 0.8])
        level = np.digitize(full['prob'].values, q)
    else:
        thr = np.array([-np.inf] + LEVEL_THRESHOLDS + [np.inf])
        level = np.digitize(full['prob'].values, thr[1:-1])
    level[full['is_submerged'] == 1] = 0              # 水下 → 最低级（极低易发）
    full['level'] = level
    full = full.sort_values('orig_id').reset_index(drop=True)
    print(f'全量概率表: {len(full)}（orig_id {full["orig_id"].min()}~{full["orig_id"].max()}）')

    # ---------- 4. 回填全量 shp（26068） ----------
    import geopandas as gpd
    gdf = gpd.read_file(FULL_SHP)
    gdf['_orig'] = gdf['Id'].astype(str)
    gdf = gdf.merge(full[['orig_id', 'prob', 'level', 'is_submerged']],
                    left_on='_orig', right_on='orig_id', how='left')
    gdf['ls_prob'] = gdf['prob'].fillna(0.0)
    gdf['ls_level'] = gdf['level'].fillna(0).astype(int) + 1     # 0..4 → 1..5
    gdf['is_submerged'] = gdf['is_submerged'].fillna(0).astype(int)
    gdf = gdf.drop(columns=['_orig', 'orig_id', 'prob', 'level'])

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_shp = PRED_DIR / f'susceptibility_units_{args.plan}_full.shp'
    gdf.to_file(out_shp)
    print(f'已导出（全量 26068）: {out_shp}')
    print('级别分布(1-5):', dict(pd.Series(gdf['ls_level']).value_counts().sort_index()))
    print('水下单元数:', int(gdf['is_submerged'].sum()), '| 水下级别全为1:', bool((gdf.loc[gdf['is_submerged']==1, 'ls_level']==1).all()))

    # ---------- 5. 全量表格（unit_id / prob / level） ----------
    out_csv = PRED_DIR / f'gnn_{args.plan}_probabilities.csv'
    tbl = full[['orig_id', 'prob', 'level']].copy()
    tbl.columns = ['unit_id', 'ls_prob', 'ls_level']
    tbl['ls_level'] = tbl['ls_level'] + 1     # 0..4 → 1..5
    tbl.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'表格已导出: {out_csv}（{len(tbl)} 行: unit_id/ls_prob/ls_level）')

    # ---------- 6. 统计 ----------
    counts = np.bincount(gdf['ls_level'].values - 1, minlength=5)
    stats_lines = ['滑坡易发性等级统计（全量 26068 单元）\n', '=' * 40]
    for i, (name, c) in enumerate(zip(LEVEL_NAMES, counts)):
        stats_lines.append(f'  {i + 1}: {name}: {c}（{c / 26068:.1%}）')
    stats_lines.append(f'\n  非水下概率范围: {full.loc[full["is_submerged"]==0,"prob"].min():.4f} ~ '
                       f'{full.loc[full["is_submerged"]==0,"prob"].max():.4f}')
    stats_path = PRED_DIR / f'statistics_{args.plan}.txt'
    stats_path.write_text('\n'.join(stats_lines), encoding='utf-8')
    print('\n'.join(stats_lines))

    # ---------- 7. 示意图 ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP'):
            if any(name in f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams['font.sans-serif'] = [name]
                plt.rcParams['axes.unicode_minus'] = False
                break
        colors = ['#2E8B57', '#9ACD32', '#FFD700', '#FF8C00', '#DC143C']
        fig, ax = plt.subplots(figsize=(10, 6))
        gdf.plot(ax=ax, column='ls_level', cmap=matplotlib.colors.ListedColormap(colors),
                 legend=True, categorical=True, legend_kwds={'labels': LEVEL_NAMES})
        ax.set_title('滑坡易发性分布（全量 26068，水下单元=1级）')
        ax.set_axis_off()
        fig.savefig(PRED_DIR / f'susceptibility_map_{args.plan}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'示意图已保存: susceptibility_map_{args.plan}.png')
    except Exception as e:
        print(f'[警告] 示意图生成失败: {e}')


if __name__ == '__main__':
    main()

