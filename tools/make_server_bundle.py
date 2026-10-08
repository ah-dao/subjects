# -*- coding: utf-8 -*-
"""打包"服务器最小上传集"：代码 + 最小数据 + 离线 wheel + TabPFN 权重 → 一个 zip。

为什么需要：
  全量 `data/` 是 21 GB，绝大多数在服务器上用不到（地质栅格/道路/地形原始影像都已提完特征）。
  本脚本按 `docs/当前模型方案.md` §7.1 的清单打包，控制在 ~250 MB。

用法：
  python tools/make_server_bundle.py                 # 默认：模型线（M1/M3）所需
  python tools/make_server_bundle.py --with-terrain  # 额外带上 30 m 地形栈（M2 用，+76 MB）
  python tools/make_server_bundle.py --with-gee      # 额外带 GEE 矩阵缓存（重算 GEE 特征用，+175 MB）
输出：
  dist/server_bundle_<日期>.zip   （内含 SETUP.md，服务器上照它执行）
"""
import argparse
import datetime
import os
import sys
import zipfile

ROOT = r'C:\Users\dollars\code\subjects'
DIST = os.path.join(ROOT, 'dist')

CODE = ['src', 'tills', 'tools', 'docs']
CODE_FILES = ['baseline_xgb.py', 'baseline_tabpfn.py', 'requirements.txt', '.gitignore',
              'README.md']
DATA = [  # (相对路径, 说明, 是否可选)
    ('features/v2/features_v2_train_38.csv', '38 维主线（训练 25,509）', False),
    ('features/v2/features_v2_38.csv', '38 维主线（全量 25,939，出图用）', False),
    ('features/v2/features_v2_train.csv', '34 维对照', False),
    ('features/v2/features_v2.csv', '34 维对照（全量）', False),
    ('features/v2/unit_id_map_v2.csv', 'ID 对照表', False),
    ('features/v2/county_units_v2_train.csv', 'admin 分折用', False),
    ('features/v2/county_units_v2.csv', 'admin 分折用（全量）', False),
    ('features/v2/groups', '分组特征（地形/水系/道路/土壤/岩性/产状/力学等）', False),
    ('features/v2/sources/daily_station_levels.csv', 'M1 逐日库水位', False),
    ('data/slope_units', '斜坡单元 shp + 计数表（质心/几何/ID）', False),
    ('models/tabpfn', 'TabPFN v2 权重（29 MB，来自 ModelScope，服务器免 HF）', False),
    ('data/terrain/Terrain_MultiBand.tif', '30 m 地形栈（M2 用）', True),
    ('features/v2/sources/ndvi_unit_matrix.csv', 'GEE NDVI 矩阵（可选）', True),
    ('features/v2/sources/rain_unit_matrix.csv', 'GEE 降雨矩阵（可选）', True),
]
SKIP_DIRS = {'__pycache__', '.git', 'archive', 'dist', '.deps'}


def add_path(zf, rel, optional=False):
    p = os.path.join(ROOT, rel.replace('/', os.sep))
    if not os.path.exists(p):
        print('  %-56s %s' % (rel, '（跳过：不存在）' if optional else '⚠ 缺失'))
        return 0
    n = 0
    if os.path.isfile(p):
        zf.write(p, rel)
        n = 1
    else:
        for dirpath, dirnames, files in os.walk(p):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in files:
                if f.endswith(('.pyc',)):
                    continue
                full = os.path.join(dirpath, f)
                zf.write(full, os.path.relpath(full, ROOT).replace(os.sep, '/'))
                n += 1
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p) for f in fs) \
        if os.path.isdir(p) else os.path.getsize(p)
    print('  %-56s %5d 个文件 %8.1f MB' % (rel, n, sz / 1e6))
    return n


def setup_md(with_terrain, with_gee):
    return f"""# 服务器运行包（模型线：M1 / M2 / M3）

> 生成于 {datetime.date.today()}　｜　由 `tools/make_server_bundle.py` 打包
> 详细方案见 `docs/当前模型方案.md`；本文件只讲**怎么在服务器上跑起来**。

## 0. 目录约定

解压到 GPU 实例的数据盘（AutoDL: `/root/autodl-tmp/`；恒源云: `/hy-tmp/` 或你的数据盘）：

```bash
cd /root/autodl-tmp && unzip server_bundle_*.zip -d subjects && cd subjects
export PYTHONPATH=$PWD/.deps            # 若用离线 wheel 安装，见下
```

## 1. 环境（两条路，任选）

**A. 在线安装（推荐，服务器网络通常可用）**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128      # >=2.7
pip install xgboost scikit-learn pandas numpy scipy shapely pyproj rasterio pyshp
pip install ./tools/wheels/tabpfn.whl    # 或 pip install tabpfn
export HF_ENDPOINT=https://hf-mirror.com # 若还要自己拉权重；本包已带权重，可跳过
```

**B. 离线安装（服务器也不能联网时）**

```bash
mkdir -p .deps && cd .deps
for w in ../tools/wheels/*.whl; do unzip -o -q "$w"; done && cd ..
export PYTHONPATH=$PWD/.deps
```

## 2. 权重（已随包携带，无需外网）

`models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`（29 MB，来自 ModelScope 的
`Prior-Labs/TabPFN-v2-clf`；本机 huggingface.co / hf-mirror 均不可达，故走魔搭）。

## 3. 冒烟（1 分钟，确认链路）

```bash
export HF_HUB_OFFLINE=1 TABPFN_ALLOW_CPU_LARGE_DATASET=1
python baseline_tabpfn.py --smoke --method spatial_kmeans --context matched --n-estimators 2
python baseline_xgb.py --features-csv features/v2/features_v2_train_38.csv \\
    --method spatial_kmeans --seed 42 --neg-sampling matched --neg-k 5
```

## 4. 入场实验（方案 §5.2 的 2×2：模型 × 人群）

```bash
# ① soft 等效人群（远区按 λ=0.2 子采样，上下文上限 12k）
python baseline_xgb.py   --features-csv features/v2/features_v2_train_38.csv \\
    --method spatial_kmeans --neg-sampling soft --neg-lam 0.2 --neg-km 4.0
python baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv \\
    --method spatial_kmeans --context soft --neg-lam 0.2 --max-train 12000 --seeds 10

# ② matched 人群（病例-对照 k=5，TabPFN 甜区：上下文约 3.2k，不触上限）
python baseline_xgb.py   --features-csv features/v2/features_v2_train_38.csv \\
    --method spatial_kmeans --neg-sampling matched --neg-k 5
python baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv \\
    --method spatial_kmeans --context matched --match-k 5 --seeds 10

# ③ admin 分折复核（两套分折同号才算通过）
#    把 --method spatial_kmeans 换成 --method admin 重跑上面四条
```

**判定**（同折同种子配对）：

```bash
python tools/paired_compare.py \\
  xgb_spatial=results/runs/r10/<xgb-soft-spatial>.json \\
  tabpfn_spatial=results/runs/tabpfn/<tabpfn-soft-spatial>.json
```

Δ 必须**超过该口径 MDE**（matched5-spatial 0.0081；soft02-spatial 0.0032）且配对显著。

## 5. M1（库水位-高程算子，先做 (a) 特征化路线）

```bash
python tills/v2_extract_water_operator.py          # 生成 features/v2/groups/water_operator.csv
python baseline_xgb.py --features-csv features/v2/features_v2_train_38.csv \\
    --features <38 维 + 算子特征列> --method spatial_kmeans --neg-sampling soft --neg-lam 0.2
```

（脚本由本机侧编写，随下一版包更新；M1 也可完全在本地 CPU 跑，不需要 GPU。）

## 6. 结果回传

```bash
tar czf runs_$(date +%m%d).tgz results/runs results/*.txt
```

都很小（几 MB），直接拉回本地；**不要回传 GeoTIFF/权重**。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-terrain', action='store_true')
    ap.add_argument('--with-gee', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    os.makedirs(DIST, exist_ok=True)
    out = args.out or os.path.join(DIST, 'server_bundle_%s.zip' % datetime.date.today().strftime('%m%d'))
    print('打包 -> %s' % out)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in CODE:
            add_path(zf, rel)
        for rel in CODE_FILES:
            add_path(zf, rel, optional=True)
        add_path(zf, 'tools/wheels')
        for rel, _desc, opt in DATA:
            if rel.endswith('Terrain_MultiBand.tif') and not args.with_terrain:
                continue
            if '_unit_matrix.csv' in rel and not args.with_gee:
                continue
            add_path(zf, rel, optional=opt)
        zf.writestr('SETUP.md', setup_md(args.with_terrain, args.with_gee))
    print('\n完成：%s（%.1f MB）' % (out, os.path.getsize(out) / 1e6))
    print('服务器上：unzip + 按 SETUP.md 执行')


if __name__ == '__main__':
    main()
