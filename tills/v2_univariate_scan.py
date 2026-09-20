# -*- coding: utf-8 -*-
"""单变量 AUC 扫描: 新表(v35) vs 旧表(v34), 用于判断 AUC 跃升来源与泄漏排查。"""
import io
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = r'C:\Users\dollars\code\subjects'
buf = io.StringIO()
new = pd.read_csv(ROOT + r'\features\v2\\features_v2_train.csv')
old = pd.read_csv(ROOT + r'\archive\v1\features\event_window_features_k2_v34_train.csv')
FEATS = [c for c in new.columns if c not in ('unit_id', 'label')]

rows = []
for c in FEATS:
    r = {'feature': c}
    for nm, d in [('new', new), ('old', old)]:
        if c in d.columns:
            x = d[c].astype(float).values
            y = d['label'].values
            m = np.isfinite(x)
            r[f'{nm}_auc'] = roc_auc_score(y[m], x[m]) if (m.sum() > 100 and 0 < y[m].sum() < m.sum()) else np.nan
        else:
            r[f'{nm}_auc'] = np.nan
    rows.append(r)
t = pd.DataFrame(rows)
t['|new-0.5|'] = (t['new_auc'] - 0.5).abs()
t = t.sort_values('|new-0.5|', ascending=False)
print('=== 单变量 AUC（按新表判别力降序）===', file=buf)
print(t[['feature', 'new_auc', 'old_auc']].round(4).to_string(index=False), file=buf)

print(f'\n新表 正样本 {int(new.label.sum())}/{len(new)} | 旧表 正样本 {int(old.label.sum())}/{len(old)}', file=buf)
print(f'新表 AUC>0.8 的单变量数: {int((t["new_auc"] > 0.8).sum())} | >0.9: {int((t["new_auc"] > 0.9).sum())}', file=buf)
print(f'旧表 AUC>0.8 的单变量数: {int((t["old_auc"] > 0.8).sum())} | >0.9: {int((t["old_auc"] > 0.9).sum())}', file=buf)
open(ROOT + r'\results\v2_univariate_auc.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('ok')
