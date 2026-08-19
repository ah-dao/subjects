"""
GraphSAGE + Transformer 训练入口（OPTIMIZATION_PATHS.md 8.2 阶段 3）。

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

from src.config import (FEATURES_CSV, GRAPH_NPZ, study_shp_path, MODEL_DIR,
                        RESULT_DIR, OOF_PREDICTIONS_CSV,
                        PLAN, HIDDEN_DIM, NUM_HEADS, TRANSFORMER_LAYERS, DROPOUT,
                        LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, EARLY_STOP_PATIENCE,
                        K_FOLDS, FOLD_METHOD, SEED, INPUT_DIM)
from src.train import run_cv, train_final, DEVICE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=PLAN, choices=['A', 'B', 'C'])
    parser.add_argument('--folds', type=int, default=K_FOLDS)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--patience', type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument('--fold-method', default=FOLD_METHOD,
                        choices=['spatial_kmeans', 'random'])
    parser.add_argument('--final-epochs', type=int, default=0,
                        help='最终模型训练轮数（0=与 --epochs 相同）')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    cfg = {
        'input_dim': INPUT_DIM, 'hidden_dim': HIDDEN_DIM, 'num_heads': NUM_HEADS,
        'num_layers': TRANSFORMER_LAYERS, 'dropout': DROPOUT,
        'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'epochs': args.epochs, 'patience': args.patience,
        'k_folds': args.folds, 'fold_method': args.fold_method,
    }
    print(f'设备: {DEVICE}')
    print(f'方案: {args.plan} | {args.folds} 折 | epochs {args.epochs} | 方法 {args.fold_method}')

    fold_aucs, fold_recalls, _ = run_cv(FEATURES_CSV, GRAPH_NPZ, study_shp_path(),
                                        cfg, plan=args.plan, seed=args.seed,
                                        save_oof=OOF_PREDICTIONS_CSV)

    # 最终模型（全数据）
    model_path = MODEL_DIR / f'best_{args.plan}.pth'
    scaler_path = MODEL_DIR / f'scaler_{args.plan}.npz'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_epochs = args.final_epochs or args.epochs
    print(f'\n训练最终模型（全数据, {final_epochs} epochs）...')
    train_final(FEATURES_CSV, GRAPH_NPZ, cfg, plan=args.plan, seed=args.seed,
                model_path=model_path, scaler_path=scaler_path,
                epochs=final_epochs)

    valid = [a for a in fold_aucs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float('nan')
    std_auc = float(np.std(valid)) if valid else float('nan')
    result = {
        'plan': args.plan, 'fold_method': args.fold_method,
        'fold_aucs': fold_aucs, 'fold_recalls': fold_recalls,
        'mean_auc': mean_auc, 'std_auc': std_auc,
        'model_path': str(model_path), 'scaler_path': str(scaler_path),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f'train_gnn_{args.plan}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {out}')


if __name__ == '__main__':
    main()
