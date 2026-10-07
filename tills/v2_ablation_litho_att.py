# -*- coding: utf-8 -*-
"""岩性(新源软硬4级) + 产状(删列后8列) 消融: 确定保留哪些特征。

背景(2026-09 换源后):
  - 岩性 lithology.csv 已换源中国岩性分布图(软硬分级), 新列 4占比+weak+grade+dom+cov
  - 产状 attitude.csv 已删 dip_mean 与 oblique/anti 占比, 剩 8 列
  - 旧结论(NEW_FACTORS_V2.md): 旧岩性有益; 旧产状组主动有害 —— 需对新列重验

变体(基线 34 维 + 增列, admin x soft 5折 x 3 seeds, 基线 AUC=0.9090):
  litho4     4 类占比(和=1, 共线)
  litho_gw    grade + weak(软硬指数 + 软弱岩占比)
  litho_grade1 仅 grade | litho_weak1 仅 weak
  att_all     产状 8 列全量
  att_dir     倾向 sin/cos + 方向一致性
  att_stype   平地 + 顺向坡占比(SlopeType 0/1)
  att_dip1    仅顺向坡占比
  att_c1      仅方向一致性(旧实验唯一非有害产状列)
  noiseN      同列数噪声对照
判据: Δ噪声对照 > +0.003 真增益; 接近 0 中性; < 0 有害。
输出: results/v2_ablation_litho_att.txt / .json
"""
import contextlib
import glob
import io
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = r'C:\Users\dollars\code\subjects'
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tills'))
from shp_io import read_polygons                                    # noqa: E402
from pyproj import CRS, Transformer                                 # noqa: E402
import src.dataset as ds                                            # noqa: E402


def _c(shp_path):
    g, _, _ = read_polygons(shp_path)
    return np.array([[x.centroid.x, x.centroid.y] for x in g], dtype=np.float64)


def _cu(shp_path, utm_epsg='EPSG:32649'):
    g, _, w = read_polygons(shp_path)
    tf = Transformer.from_crs(CRS.from_wkt(w), CRS.from_epsg(int(str(utm_epsg).split(':')[1])),
                              always_xy=True)
    p = [x.centroid for x in g]
    x, y = tf.transform([q.x for q in p], [q.y for q in p])
    return np.column_stack([x, y]).astype(np.float64)


ds.load_centroids = _c
ds.load_centroids_utm = _cu

ATT8 = ['dipdir_sin_mean', 'dipdir_cos_mean', 'dipdir_consistency',
        'rta_mean', 'apparent_dip_mean', 'stype_flat_frac',
        'stype_dip_slope_frac', 'att_cov']
VARIANTS = [
    ('litho4', ['litho_hard_frac', 'litho_medhard_frac',
                'litho_medsoft_frac', 'litho_soft_frac']),
    ('litho_gw', ['litho_grade', 'litho_weak_frac']),
    ('litho_grade1', ['litho_grade']),
    ('litho_weak1', ['litho_weak_frac']),
    ('att_all', ATT8),
    ('att_dir', ['dipdir_sin_mean', 'dipdir_cos_mean', 'dipdir_consistency']),
    ('att_stype', ['stype_flat_frac', 'stype_dip_slope_frac']),
    ('att_dip1', ['stype_dip_slope_frac']),
    ('att_c1', ['dipdir_consistency']),
]
NOISE_N = [1, 2, 3, 4, 8]
SEEDS = [42, 43, 44]

base = pd.read_csv(ROOT + r'\features\v2\features_v2_train.csv')
base['unit_id'] = base['unit_id'].astype(int)
FEATS0 = [c for c in base.columns if c not in ('unit_id', 'label')]
G = ROOT + r'\features\v2\groups'
extra = pd.DataFrame({'unit_id': base['unit_id'].values})
for f in ('lithology', 'attitude'):
    d = pd.read_csv(f'{G}\\{f}.csv')
    d['unit_id'] = d['unit_id'].astype(int)
    extra = extra.merge(d, on='unit_id', how='left')
rng = np.random.default_rng(20260921)
for i in range(1, 9):
    extra[f'rz{i}'] = rng.normal(size=len(extra))

outdir = ROOT + r'\features\v2\ablation'
os.makedirs(outdir, exist_ok=True)
paths = {}
ALLV = VARIANTS + [(f'noise{k}', [f'rz{i}' for i in range(1, k + 1)]) for k in NOISE_N]
for name, cols in ALLV:
    df = pd.concat([base[['unit_id'] + FEATS0 + ['label']].reset_index(drop=True),
                    extra[cols].reset_index(drop=True)], axis=1)
    p = f'{outdir}\\la_{name}.csv'
    df.to_csv(p, index=False, encoding='utf-8-sig')
    paths[name] = p
print(f'已生成 {len(paths)} 个变体表')

BASE = ROOT + r'\baseline_xgb.py'
RES = ROOT + r'\results'
summary = []
for name, cols in ALLV:
    per = []
    for s in SEEDS:
        before = set(glob.glob(RES + r'\baseline_xgb*.json'))
        argv = ['baseline_xgb.py', '--features-csv', paths[name], '--method', 'admin',
                '--neg-sampling', 'soft', '--folds', '5', '--seed', str(s)]
        old = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                import runpy
                runpy.run_path(BASE, run_name='__main__')
        except SystemExit:
            pass
        finally:
            sys.argv = old
        newest = max(set(glob.glob(RES + r'\baseline_xgb*.json')) - before or before,
                     key=os.path.getmtime)
        j = json.load(open(newest, encoding='utf-8'))
        per.append({'auc': j['mean_auc'], 'pool': j['mean_auc_pool'],
                    'recall': float(np.nanmean(j['fold_recalls'])), 'folds': j['fold_aucs'],
                    'imp': {d['feature']: d['importance'] for d in j['feature_importance']}})
    summary.append({'variant': name, 'n_add': len(cols), 'cols': cols,
                    'auc': float(np.mean([x['auc'] for x in per])),
                    'std': float(np.std([x['auc'] for x in per])),
                    'pool': float(np.mean([x['pool'] for x in per])),
                    'recall': float(np.mean([x['recall'] for x in per])),
                    'folds': [a for x in per for a in x['folds']],
                    'imp': {k: float(np.mean([x['imp'].get(k, 0) for x in per])) for k in cols}})
    print(f'  {name:12s} +{len(cols):2d} AUC {summary[-1]["auc"]:.4f} | recall {summary[-1]["recall"]:.4f}',
          flush=True)

BASE_AUC = 0.9090
noise = {s['n_add']: s['auc'] for s in summary if s['variant'].startswith('noise')}
buf = io.StringIO()
print('=' * 96, file=buf)
print('岩性(新源) + 产状(8列) 消融 | admin x soft | 5 折 x 3 seeds | 基线 v2 = 0.9090', file=buf)
print('=' * 96, file=buf)
print(f'{"变体":12s} {"增列":>4s} {"AUC":>8s} {"±seed":>7s} {"Δbase":>8s} {"Δ噪声对照":>10s} '
      f'{"池AUC":>8s} {"recall@10%":>10s} {"新增列重要性":>10s}', file=buf)
for s in summary:
    nz = noise.get(s['n_add'])
    dn = (s['auc'] - nz) if nz else float('nan')
    tot = sum(s['imp'].values()) or 1.0
    top = sorted(s['imp'].items(), key=lambda x: -x[1])[:3]
    imp = ' | '.join(f'{k.replace("_frac","").replace("litho_","").replace("_mean","")} {v/tot*100:.0f}%'
                     for k, v in top)
    print(f'{s["variant"]:12s} {s["n_add"]:4d} {s["auc"]:8.4f} {s["std"]:7.4f} '
          f'{s["auc"]-BASE_AUC:+8.4f} {dn:+10.4f} {s["pool"]:8.4f} {s["recall"]:10.4f} {imp:>10s}',
          file=buf)
print('\n判据: Δ噪声对照 > +0.003 才算真增益; 接近 0 为中性; < 0 为有害。', file=buf)
json.dump(summary, open(RES + r'\v2_ablation_litho_att.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
open(RES + r'\v2_ablation_litho_att.txt', 'w', encoding='utf-8').write(buf.getvalue())
print('已写出 results/v2_ablation_litho_att.txt')
