"""
GraphSAGE + Transformer 训练入口（PROJECT_OVERVIEW.md）。

用法：
    python train_gnn.py --plan B --folds 5 [--epochs 200] [--patience 20]
                        [--fold-method spatial_kmeans|random] [--seed 42]
                        [--final-epochs 100]
行为：
    1. K 折空间交叉验证，输出每折 AUC + 平均 AUC±std，保存 OOF 外推预测
    2. 用全部数据训练最终模型（供 predict_gnn.py 出图），保存权重与归一化参数
输出：
    models/best_<plan>.pth / models/scaler_<plan>.npz
    features/oof_predictions.csv
    results/train_gnn_<plan>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (EVENT_WINDOW_FEATURES_CSV, EVENT_WINDOW_FEATURES,
                        GRAPH_NPZ, study_shp_path, MODEL_DIR,
                        RESULT_DIR, OOF_PREDICTIONS_CSV, STUDY_UNITS_COUNT_CSV,
                        PLAN, HIDDEN_DIM, NUM_HEADS, TRANSFORMER_LAYERS, DROPOUT,
                        LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, EARLY_STOP_PATIENCE,
                        K_FOLDS, FOLD_METHOD, SEED, INPUT_DIM,
                        NEG_SAMPLING, NEG_KM, NEG_K, NEG_LAM, NEG_SEED)
from src.dataset import load_sample_weights
from src.train import run_cv, train_final, DEVICE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN, choices=['A', 'B', 'C'])
    parser.add_argument('--folds', type=int, default=K_FOLDS)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--patience', type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument('--fold-method', default=FOLD_METHOD,
                        choices=['spatial_kmeans', 'random', 'admin'],
                        help='分折方法：spatial_kmeans / admin（按县，需 features/county_units.csv）/ random')
    parser.add_argument('--weight-scheme', default='none', choices=['none', 'count'],
                        help='逐样本权重：none=不加权；count=正样本按研究期滑坡次数加权')
    parser.add_argument('--final-epochs', type=int, default=0,
                        help='最终模型训练轮数（0=与 --epochs 相同）')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--neg-sampling', default=NEG_SAMPLING,
                        choices=['none', 'proximity', 'soft'],
                        help='负样本口径：none=全域；proximity=时空邻近硬采样；'
                             'soft=软负采样（邻近权重1、远区λ，λ=0 退化硬采样、λ=1 退化全域）')
    parser.add_argument('--neg-km', type=float, default=NEG_KM, help='时空邻近半径（km）')
    parser.add_argument('--neg-k', type=int, default=NEG_K, help='每正样本抽取负样本数')
    parser.add_argument('--neg-lam', type=float, default=NEG_LAM, help='软负采样远区负样本权重')
    parser.add_argument('--neg-seed', type=int, default=NEG_SEED, help='负采样种子')
    args = parser.parse_args()

    cfg = {
        'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM, 'num_heads': NUM_HEADS,
        'num_layers': TRANSFORMER_LAYERS, 'dropout': DROPOUT,
        'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'epochs': args.epochs, 'patience': args.patience,
        'k_folds': args.folds, 'fold_method': args.fold_method,
        'neg_sampling': args.neg_sampling, 'neg_km': args.neg_km,
        'neg_k': args.neg_k, 'neg_lam': args.neg_lam, 'neg_seed': args.neg_seed,
    }
    sample_weight = load_sample_weights(EVENT_WINDOW_FEATURES_CSV, STUDY_UNITS_COUNT_CSV,
                                        scheme=args.weight_scheme)
    n_pos = int((sample_weight > 1).sum())
    if args.weight_scheme == 'count':
        print(f'[权重] count 方案: {n_pos} 个正样本按滑坡次数加权（1/2/3/4）')
    print(f'设备: {DEVICE}')
    print(f'方案: {args.plan} | {args.folds} 折 | epochs {args.epochs} | 方法 {args.fold_method}'
          f' | 负样本: {args.neg_sampling}'
          + (f'（{args.neg_km}km × k={args.neg_k}）' if args.neg_sampling == 'proximity' else '')
          + (f'（{args.neg_km}km, λ={args.neg_lam}）' if args.neg_sampling == 'soft' else ''))

    fold_aucs, fold_aucs_pool, fold_recalls, _ = run_cv(
        EVENT_WINDOW_FEATURES_CSV, GRAPH_NPZ, study_shp_path(),
        cfg, plan=args.plan, seed=args.seed,
        save_oof=OOF_PREDICTIONS_CSV,
        sample_weight=sample_weight,
        features=EVENT_WINDOW_FEATURES)

    # 最终模型（全数据；负采样时全局采样一次）
    model_path = MODEL_DIR / f'best_{args.plan}.pth'
    scaler_path = MODEL_DIR / f'scaler_{args.plan}.npz'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_epochs = args.final_epochs or args.epochs
    print(f'\n训练最终模型（全数据, {final_epochs} epochs）...')
    train_final(EVENT_WINDOW_FEATURES_CSV, GRAPH_NPZ, cfg, plan=args.plan, seed=args.seed,
                model_path=model_path, scaler_path=scaler_path,
                epochs=final_epochs, sample_weight=sample_weight,
                features=EVENT_WINDOW_FEATURES, shp_path=study_shp_path())

    valid = [a for a in fold_aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float('nan')
    std_auc = float(np.std(valid)) if valid else float('nan')
    valid_pool = [a for a in fold_aucs_pool if not np.isnan(a)]
    mean_pool = float(np.mean(valid_pool)) if valid_pool else float('nan')
    std_pool = float(np.std(valid_pool)) if valid_pool else float('nan')
    result = {
        'plan': args.plan, 'fold_method': args.fold_method,
        'fold_aucs': fold_aucs, 'fold_aucs_pool': fold_aucs_pool,
        'fold_recalls': fold_recalls,
        'mean_auc': mean_auc, 'std_auc': std_auc,
        'mean_auc_pool': mean_pool, 'std_auc_pool': std_pool,
        'neg_sampling': args.neg_sampling, 'neg_km': args.neg_km,
        'neg_k': args.neg_k, 'neg_seed': args.neg_seed,
        'model_path': str(model_path), 'scaler_path': str(scaler_path),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f'train_gnn_{args.plan}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {out}')


if __name__ == '__main__':
    main()
