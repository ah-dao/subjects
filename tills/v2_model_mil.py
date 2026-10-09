# -*- coding: utf-8 -*-
"""M2 模型骨架：单元=像元袋的多示例学习（MIL）+ 与主模型的**残差融合**。

定位（docs/当前模型方案.md §4）：
    现有 38 维全是单元级聚合量；机制发生在岸坡面尺度（"某处 2-3 个像元的软岩恰在水线附近"），
    单元聚合把"在哪里"平均掉了。M2 让模型直接看**单元内的像元集合**，并学出"关键带"注意力。

结构：
    像元编码器 (C -> H)  ->  门控注意力池化 (Ilse et al. 2018, gated attention)  ->  单元表示 h
    logit = logit_base + g([h, 表格特征])        ← 残差融合：base 来自 TabPFN/XGB 的 OOF，冻结
    注意力权重 a_p 即"关键带"图（可导出成栅格，替代 SHAP 做空间解释）

两种模式：
    --smoke-synthetic     合成数据冒烟（不需要任何数据文件，CPU 可跑）——验证形状/反传/池化/融合
    --bags <npz> --tab <csv> [--base-col p_soft02]   真数据训练（服务器上跑）

用法：
    python tills/v2_model_mil.py --smoke-synthetic
    python tills/v2_model_mil.py --bags features/v2/unit_bags.npz \
        --tab features/v2/features_v2_train_38.csv --epochs 5 --k 64
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'


def build_model(C, H=64, tab_dim=38, dropout=0.2):
    import torch
    import torch.nn as nn

    class GatedAttentionMIL(nn.Module):
        """门控注意力池化（Ilse et al. 2018）：a_p = softmax(w·(tanh(Vh) ⊙ σ(Uh)))"""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(C, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
            self.V = nn.Linear(H, H)
            self.U = nn.Linear(H, H)
            self.w = nn.Linear(H, 1)
            self.head = nn.Sequential(nn.Linear(H + tab_dim, H), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(H, 1))

        def forward(self, x, tab, logit_base, mask=None):
            """x: (B, K, C) 像元；tab: (B, tab_dim)；logit_base: (B,)；mask: (B, K) 有效像元"""
            h = self.enc(x)                                   # B,K,H
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)   # B,K
            if mask is not None:
                a = a.masked_fill(~mask, -1e9)
            a = torch.softmax(a, dim=1)
            z = torch.einsum('bk,bkh->bh', a, h)               # 单元表示
            delta = self.head(torch.cat([z, tab], dim=1)).squeeze(-1)
            return logit_base + delta, a                       # 残差融合 + 注意力权重

    return GatedAttentionMIL()


def smoke_synthetic(args):
    import torch
    torch.manual_seed(0)
    C, N, K = 5, 400, args.k
    model = build_model(C, H=args.hidden, tab_dim=args.tab_dim)
    x = torch.randn(N, K, C)
    mask = torch.rand(N, K) > 0.25                            # 模拟小单元（像元不足 -> 掩码）
    tab = torch.randn(N, args.tab_dim)
    base = torch.randn(N) * 0.5                               # 模拟冻结的 base logit
    y = (torch.rand(N) > 0.8).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    print('MIL 冒烟（合成数据）| 通道 %d | 单元 %d | 每袋 %d 像元 | 掩码比例 %.0f%%'
          % (C, N, K, 100 * (~mask).float().mean()))
    t0 = time.time()
    for ep in range(3):
        opt.zero_grad()
        logit, a = model(x, tab, base, mask)
        loss = lossf(logit, y)
        loss.backward()
        opt.step()
        print('  epoch %d | loss %.4f | logit 范围 %.2f~%.2f | 注意力每袋和 %.4f（应=1）'
              % (ep + 1, loss.item(), logit.min().item(), logit.max().item(),
                 a.sum(dim=1).mean().item()))
    print('  前向+反传 3 epoch 用时 %.1fs（CPU）' % (time.time() - t0))
    print('  参数量 %d' % sum(p.numel() for p in model.parameters()))


def load_real(args):
    """真数据：读像元袋 npz + 表格特征，按单元组装 (bag, tab, base, y)。"""
    import pandas as pd
    import torch
    z = np.load(args.bags, allow_pickle=True)
    uid_pix, X = z['uid'], z['X'].astype(np.float32)
    bands = [str(b) for b in z['bands']]
    tab = pd.read_csv(args.tab)
    feats = [c for c in tab.columns if c not in ('unit_id', 'label', 'p_soft02', 'p_matched5')]
    tab_dim = len(feats)
    order = np.argsort(uid_pix, kind='stable')                # 按单元聚合像元
    uid_pix, X = uid_pix[order], X[order]
    starts = np.searchsorted(uid_pix, np.arange(1, uid_pix.max() + 2))
    print('像元袋：%d 像元 / %d 单元 / 波段 %s | 表格 %d 维'
          % (len(uid_pix), len(starts) - 1, bands, tab_dim))

    class BagDataset(torch.utils.data.Dataset):
        def __init__(self, ids):
            self.ids = ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            u = int(self.ids[i])
            s, e = starts[u - 1], starts[u]
            idx = np.arange(s, e)
            if len(idx) == 0:
                x = np.zeros((args.k, X.shape[1]), np.float32); m = np.zeros(args.k, bool)
            else:
                pick = np.random.choice(idx, size=min(args.k, len(idx)), replace=False)
                x = np.zeros((args.k, X.shape[1]), np.float32); m = np.zeros(args.k, bool)
                x[:len(pick)] = X[pick]; m[:len(pick)] = True
            row = tab.iloc[u - 1]
            return (torch.from_numpy(x), torch.from_numpy(m),
                    torch.tensor(row[feats].values.astype(np.float32)),
                    torch.tensor(float(row.get(args.base_col, 0.0))),
                    torch.tensor(float(row['label'])))
    return BagDataset, tab_dim


def train_real(args):
    import torch
    BagDataset, tab_dim = load_real(args)
    ds = BagDataset(np.arange(1, len(np.unique(np.load(args.bags)['uid'])) + 1))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    C = np.load(args.bags)['X'].shape[1]
    model = build_model(C, H=args.hidden, tab_dim=tab_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    print('训练 MIL：单元 %d | 每袋 %d 像元 | 通道 %d | 表格 %d 维 | batch %d'
          % (len(ds), args.k, C, tab_dim, args.batch))
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for x, m, tb, base, y in dl:
            opt.zero_grad()
            logit, _ = model(x, tb, base, m)
            loss = lossf(logit, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        print('  epoch %d | loss %.4f' % (ep + 1, tot / max(n, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke-synthetic', action='store_true')
    ap.add_argument('--bags', default=os.path.join(ROOT, 'features', 'v2', 'unit_bags.npz'))
    ap.add_argument('--tab', default=os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv'))
    ap.add_argument('--base-col', default='p_soft02',
                    help='冻结的 base logit 来源列（出图脚本产出的 OOF 概率，会被转成 logit）')
    ap.add_argument('--k', type=int, default=64, help='每袋抽多少像元')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--tab-dim', type=int, default=38)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    if args.smoke_synthetic:
        smoke_synthetic(args)
    else:
        train_real(args)


if __name__ == '__main__':
    main()
