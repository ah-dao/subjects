"""
滑坡易发性分析（斜坡单元方案）一键流程编排。

用法：
    python main.py --stage all          # 全流程：数据 → 图 → 基线 → 训练 → 预测
    python main.py --stage data         # 仅数据准备（地形/时序/水位 → 特征表 + 事件窗口特征）
    python main.py --stage graph        # 仅图构建
    python main.py --stage baseline     # 仅 XGBoost 基线（事件窗口 19 维特征）
    python main.py --stage train        # 仅 GNN 训练（可加 --plan A/B/C --folds 5）
    python main.py --stage predict      # 仅全图推理出图（可加 --method quantile）

前提：data/ 下已放置斜坡单元 shp、滑坡点/水位数据、GEE 导出的 NDVI/降雨栈
（详见 docs/QUICKSTART.md）。
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
    ew_features = ROOT / 'features' / 'event_window_features_k2.csv'

    if args.stage in ('all', 'data'):
        run(tills / 'extract_landslide_points.py')    # 滑坡点筛选（2000-2021）
        run(tills / 'join_landslide_dates.py')        # 滑坡点→单元：计数+首末日期+研究期字段
        run(tills / 'filter_study_units.py')          # 研究期 2003-2021（剔除仅蓄水前滑坡单元）
        run(tills / 'extract_terrain_features.py')    # 地形 5 维
        run(tills / 'extract_temporal_features.py')   # NDVI/降雨 8 维（全窗口，静态对照口径）
        run(tills / 'extract_water_features.py')      # 淹没 6 维
        run(tills / 'merge_features.py')              # 合并 22 维特征表 + 标签（对照口径）
        run(tills / 'build_event_window_features.py',  # 事件窗口 19 维特征表（当前主线）
            ['--k', '2', '--start-year', '2000', '--seed', '42'])

    if args.stage in ('all', 'graph'):
        run(tills / 'build_graph.py')

    if args.stage in ('all', 'baseline'):
        run(ROOT / 'baseline_xgb.py', ['--features-csv', str(ew_features),
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
