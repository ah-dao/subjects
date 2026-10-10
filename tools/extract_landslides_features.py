# -*- coding: utf-8 -*-
"""把外部滑坡清单（features/other/landslides.xlsx）按**原始 Id** 对齐到 v2 斜坡单元的 38 维主线特征。

背景
    features/other/landslides.xlsx（133 行 × 36 列）里的 `Id` 是**项目最初版斜坡单元文件的 Id**，
    与 v2 重编号后的 `unit_id` 不同。对应关系已由 `features/v2/unit_id_map_v2.csv` 维护：
        src_fixed_Id  原始斜坡单元 Id（本表要用的键）
        unit_id       v2 当前编号（1..25939）
        src_train_Id  训练人口内的原始 Id；is_train / is_submerged / in_xiaoluoqu
    实测：133 个 Id **全部**命中 src_fixed_Id，且全部落在训练人口（is_train=1），
          也没有"被合并单元"（merged_from）的情况。

产物
    --out             Id + unit_id + 38 维特征（+ is_train 等标记）的干净表
    --out-full        再把滑坡清单原有 36 列一并带上（不丢信息）
    --report          匹配/缺失/NaN 体检报告

用法
    python tools/extract_landslides_features.py
    python tools/extract_landslides_features.py --xlsx features/other/landslides.xlsx \\
        --features-csv features/v2/features_v2_train_38.csv --out deliverables/landslides_v2_38.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, 'features', 'other', 'landslides.xlsx')
MAP = os.path.join(ROOT, 'features', 'v2', 'unit_id_map_v2.csv')
FEAT = os.path.join(ROOT, 'features', 'v2', 'features_v2_train_38.csv')
FEAT_FULL = os.path.join(ROOT, 'features', 'v2', 'features_v2_38.csv')
OUT = os.path.join(ROOT, 'deliverables', 'landslides_v2_38.csv')
OUT_FULL = os.path.join(ROOT, 'deliverables', 'landslides_v2_38_full.csv')
REPORT = os.path.join(ROOT, 'results', 'landslides_v2_38_report.txt')
BUF = []


def log(s=''):
    s = str(s)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'), flush=True)
    BUF.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xlsx', default=XLSX)
    ap.add_argument('--map', default=MAP)
    ap.add_argument('--features-csv', default=FEAT)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--out-full', default=OUT_FULL)
    ap.add_argument('--report', default=REPORT)
    ap.add_argument('--fill-nan', action='store_true', default=True,
                    help='用训练人口列均值填补 NaN（与出图脚本口径一致）')
    args = ap.parse_args()

    L = pd.read_excel(args.xlsx, sheet_name=0)
    log('滑坡清单：%s | %d 行 × %d 列 | 键列 Id 唯一值 %d 个'
        % (os.path.basename(args.xlsx), len(L), L.shape[1], L['Id'].nunique()))
    dup = L['Id'][L['Id'].duplicated()].tolist()
    if dup:
        log('  ⚠ Id 有重复：%s' % dup[:10])

    M = pd.read_csv(args.map, encoding='utf-8-sig')
    keep = ['unit_id', 'src_fixed_Id', 'is_train', 'is_submerged', 'in_xiaoluoqu']
    if 'merged_from' in M.columns:
        keep.append('merged_from')
    M = M[[c for c in keep if c in M.columns]]
    log('ID 对照表：%s | %d 行' % (os.path.basename(args.map), len(M)))

    mm = L.merge(M, left_on='Id', right_on='src_fixed_Id', how='left', validate='m:1')
    n_hit = int(mm['unit_id'].notna().sum())
    log('匹配：%d / %d 行命中 src_fixed_Id' % (n_hit, len(L)))
    if n_hit < len(L):
        miss = mm.loc[mm['unit_id'].isna(), 'Id'].tolist()
        log('  ⚠ 未命中的 Id：%s' % miss[:30])
        log('    可能原因：该单元在 v2 被并入相邻单元（见 merged_from）或落在研究区外。')

    F = pd.read_csv(args.features_csv)
    feats = [c for c in F.columns if c not in ('unit_id', 'label')]
    log('特征表：%s | %d 行 | %d 维' % (os.path.basename(args.features_csv), len(F), len(feats)))

    # ⚠ 滑坡表与特征表存在同名列（如 area）→ merge 会产生 _x/_y 后缀并让后续取列失败，
    #   先给滑坡表的同名列加 ls_ 前缀（本项目反复踩过的坑）。
    coll = [c for c in L.columns if c in feats]
    if coll:
        L = L.rename(columns={c: 'ls_' + c for c in coll})
        log('  ⚠ 与特征重名的列已加前缀 ls_：%s' % coll)
        mm = L.merge(M, left_on='Id', right_on='src_fixed_Id', how='left', validate='m:1')

    out = mm.merge(F[['unit_id'] + feats], on='unit_id', how='left', validate='m:1')
    n_feat = int(out[feats[0]].notna().sum()) if feats else 0
    log('接入特征：%d / %d 行拿到 %d 维特征' % (n_feat, len(out), len(feats)))
    if n_feat < n_hit:
        log('  ⚠ 有 %d 行在特征表里找不到（可能不在训练人口）' % (n_hit - n_feat))

    # NaN 体检与填补（litho_T2/J2/T1 在源数据缺口处为 NaN）
    nan_n = out[feats].isna().sum()
    if int(nan_n.sum()):
        log('')
        log('NaN 情况（源数据缺口）：%s' % nan_n[nan_n > 0].to_dict())
        if args.fill_nan:
            means = pd.read_csv(args.features_csv)[feats].mean()
            out[feats] = out[feats].fillna(means)
            log('  已用**训练人口列均值**填补（与 tills/v2_predict_map.py 口径一致）')
    else:
        log('NaN 情况：无')

    # 与全量表核对（若存在）
    if os.path.exists(FEAT_FULL):
        FF = pd.read_csv(FEAT_FULL)
        j = out[['unit_id'] + feats].merge(FF[['unit_id'] + feats], on='unit_id', suffixes=('_t', '_f'))
        if len(j):
            d = np.nanmax(np.abs(j[[c + '_t' for c in feats]].values -
                                  j[[c + '_f' for c in feats]].values))
            log('与全量表 features_v2_38.csv 的最大逐列差异：%.2e（应为 0）' % d)

    out['label'] = 1                       # 本清单 = 已知滑坡，直接可作预测/评估的真值

    # ---- 落盘 ----
    front = ['Id', 'unit_id', 'label'] + [c for c in ('滑坡名', '经度', '纬度') if c in out.columns]
    tag = [c for c in ('is_train', 'is_submerged', 'in_xiaoluoqu') if c in out.columns]
    only = out[front + tag + feats]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    only.to_csv(args.out, index=False, encoding='utf-8-sig')
    full = out.drop(columns=[c for c in ('src_fixed_Id',) if c in out.columns])
    os.makedirs(os.path.dirname(args.out_full), exist_ok=True)
    full.to_csv(args.out_full, index=False, encoding='utf-8-sig')
    log('')
    log('写出 %s（%d 行 × %d 列：Id + unit_id + %d 维特征）'
        % (args.out, len(only), only.shape[1], len(feats)))
    log('写出 %s（%d 行 × %d 列：额外保留滑坡清单原列）'
        % (args.out_full, len(full), full.shape[1]))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    open(args.report, 'w', encoding='utf-8').write('\n'.join(BUF))
    log('DONE -> %s' % args.report)


if __name__ == '__main__':
    main()
