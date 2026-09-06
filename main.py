"""
滑坡易发性分析（斜坡单元方案）一键流程编排。

用法：
    python main.py --stage all          # 全流程：数据 → 图 → 基线 → 训练 → 预测
    python main.py --stage data         # 仅数据准备（→ 30 维主线特征表 V30）
    python main.py --stage graph        # 仅图构建
    python main.py --stage baseline     # 仅 XGBoost 基线（30 维主线，AUC≈0.8223）
    python main.py --stage train        # 仅 GNN 训练（可加 --plan A/B/C --folds 5）
    python main.py --stage predict      # 仅全图推理出图（可加 --method quantile）

前提：data/ 下已放置斜坡单元 shp、滑坡点/水位数据、GEE 导出的 NDVI/降雨矩阵、
      CLCD 土地利用栅格、OSM 道路 shp（详见 docs/QUICKSTART.md 与 docs/FEATURES_V30.md）。
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script, extra=None):
    cmd = [PY, str(script)] + (extra or [])
    print(f'\n{"=" * 70}\n>>> 执行: {" ".join(cmd)}\n{"=" * 70}')
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all',
                        choices=['all', 'data', 'graph', 'baseline', 'train', 'predict'])
    parser.add_argument('--plan', default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--method', default='fixed', choices=['fixed', 'quantile'])
    parser.add_argument('--fold-method', default='spatial_kmeans',
                        choices=['spatial_kmeans', 'random'])
    args = parser.parse_args()

    tills = ROOT / 'tills'
    v30_features = ROOT / 'features' / 'event_window_features_k2_v30.csv'   # 30 维主线特征表

    if args.stage in ('all', 'data'):
        # ① 滑坡点与单元关联
        run(tills / 'extract_landslide_points.py')    # 滑坡点筛选（2000-2021）
        run(tills / 'join_landslide_dates.py')        # 滑坡点→单元：计数+首末日期+研究期字段
        # ② 静态特征提取（地形 / 淹没 / 水系 / 土地利用）
        run(tills / 'extract_terrain_features.py')    # 地形特征（elevation/slope/aspect/TRI…）
        run(tills / 'extract_water_features.py')      # 淹没特征 inundation_fraction
        run(tills / 'extract_water_network_features.py')  # 水系距离/密度
        run(tills / 'extract_landuse_features.py')    # CLCD 土地利用年度矩阵
        # ③ 道路特征（OSM 重庆+湖北，需已下载到 data/roads/）
        run(tills / 'extract_road_features.py')       # 距路距离/道路密度
        # ④ 事件窗口特征表（K=2 + ant_* + landuse T−1）
        run(tills / 'build_event_window_features.py',
            ['--k', '2', '--start-year', '2000', '--seed', '42'])
        # ⑤ 组装 30 维主线（+道路4 +土地利用变化3，-冗余 curvature）
        run(tills / 'build_v30_features.py', ['--k', '2'])
        print(f'\n→ 主线特征表: {v30_features}')

    if args.stage in ('all', 'graph'):
        run(tills / 'build_graph.py')

    if args.stage in ('all', 'baseline'):
        run(ROOT / 'baseline_xgb.py', ['--features-csv', str(v30_features),
                                       '--folds', str(args.folds),
                                       '--method', args.fold_method])

    if args.stage in ('all', 'train'):
        run(ROOT / 'train_gnn.py', ['--plan', args.plan, '--folds', str(args.folds),
                                    '--fold-method', args.fold_method])

    if args.stage in ('all', 'predict'):
        run(ROOT / 'predict_gnn.py', ['--plan', args.plan, '--method', args.method])

    print('\n全部完成')


if __name__ == '__main__':
    main()
