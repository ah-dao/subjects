"""评估指标：AUC 与 Recall@Top10%（6/7 节）。"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, y_score):
    """ROC-AUC。全零/全一标签时返回 nan（该折无法计算）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def recall_at_top10(y_true, y_score):
    """按概率降序取前 10% 样本，计算其中正样本的召回率（Recall@Top10%）。

    含义：如果按易发性排序优先排查前 10% 单元，能命中多大比例的真实滑坡单元。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = max(1, int(round(len(y_score) * 0.10)))
    top_idx = np.argsort(-y_score)[:k]
    pos = y_true.sum()
    if pos == 0:
        return float('nan')
    return float(y_true[top_idx].sum() / pos)


def summarize(y_true, y_score):
    """返回 (auc, recall_top10) 元组。"""
    return auc(y_true, y_score), recall_at_top10(y_true, y_score)
