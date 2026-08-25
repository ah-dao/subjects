"""
全图推理 + 矢量出图（OPTIMIZATION_PATHS.md 7 节）。

加载最终模型 → 全图 2.6 万节点前向 → 概率 [0,1] → 5 级易发性 → 回填 shapefile。

用法：
    python predict_gnn.py --plan B [--method fixed|quantile]
                          [--model models/best_B.pth --scaler models/scaler_B.npz]
输出：
    predictions/susceptibility_units.shp   （含 ls_prob、ls_level 字段）
    predictions/susceptibility_map.png     （5 级易发性示意图）
    predictions/statistics.txt             （各等级单元数统计）
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES_CSV, EVENT_WINDOW_FEATURES,
                        GRAPH_NPZ, study_shp_path, MODEL_DIR,
                        PRED_DIR, LEVEL_THRESHOLDS, LEVEL_NAMES,
                        PLAN, HIDDEN_DIM, NUM_HEADS, TRANSFORMER_LAYERS, DROPOUT)
from src.dataset import load_features, load_graph, minmax_apply
from src.model import build_model


def levels_from_probs(prob, method='fixed'):
    """按概率分 5 级：fixed=固定阈值（7.1 节），quantile=每级 20% 单元。"""
    if method == 'fixed':
        thr = np.array([-np.inf] + LEVEL_THRESHOLDS + [np.inf])
        return np.digitize(prob, thr[1:-1]) - 0  # 0..4
    # quantile：按分位数每级 20%
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

    unit_id, X, y = load_features(EVENT_WINDOW_FEATURES_CSV, features=EVENT_WINDOW_FEATURES)
    edge_index = load_graph(GRAPH_NPZ)

    # 归一化 + 模型
    sc = np.load(scaler_path)
    Xn = minmax_apply(X, sc['min_'], sc['max_'])
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
    prob = predict_all(model, x_t, edge_t)
    print(f'推理完成: {len(prob)} 个单元')

    level = levels_from_probs(prob, args.method)

    # ---------- 回填 shp ----------
    import geopandas as gpd
    gdf = gpd.read_file(study_shp_path())
    assert len(gdf) == len(prob), f'shp({len(gdf)}) 与特征表({len(prob)}) 行数不一致'
    gdf['ls_prob'] = prob
    gdf['ls_level'] = level
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_shp = PRED_DIR / 'susceptibility_units.shp'
    gdf.to_file(out_shp)
    print(f'已导出: {out_shp}（字段 ls_prob / ls_level）')

    # ---------- 统计 ----------
    counts = np.bincount(level, minlength=5)
    stats_lines = ['滑坡易发性等级统计（单元数）\n', '=' * 40]
    for i, (name, c) in enumerate(zip(LEVEL_NAMES, counts)):
        stats_lines.append(f'  {i}: {name}: {c}（{c / len(prob):.1%}）')
    stats_lines.append(f'\n  概率范围: {prob.min():.4f} ~ {prob.max():.4f}')
    stats_path = PRED_DIR / 'statistics.txt'
    stats_path.write_text('\n'.join(stats_lines), encoding='utf-8')
    print('\n'.join(stats_lines))

    # ---------- 示意图 ----------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # 中文字体回退（Windows: 微软雅黑；Linux/Colab: Noto Sans CJK）
        for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP'):
            if any(name in f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams['font.sans-serif'] = [name]
                plt.rcParams['axes.unicode_minus'] = False
                break
        colors = ['#2E8B57', '#9ACD32', '#FFD700', '#FF8C00', '#DC143C']
        fig, ax = plt.subplots(figsize=(10, 6))
        gdf.plot(ax=ax, column='ls_level', cmap=matplotlib.colors.ListedColormap(colors),
                 legend=True, categorical=True, legend_kwds={'labels': LEVEL_NAMES})
        ax.set_title('滑坡易发性分布（斜坡单元）')
        ax.set_axis_off()
        fig.savefig(PRED_DIR / 'susceptibility_map.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'示意图已保存: {PRED_DIR / "susceptibility_map.png"}')
    except Exception as e:  # matplotlib 缺失等不影响主流程
        print(f'[警告] 示意图生成失败: {e}')


if __name__ == '__main__':
    main()
