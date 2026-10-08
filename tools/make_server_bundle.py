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

## 1. 环境

**本实验已确认可用的镜像组合：PyTorch 2.9 / CUDA 12.8.1 / Python 3.11（恒源云）**
→ torch ≥2.5 ⇒ **直接用 tabpfn 8.5.0，无需降级**。

```bash
# ① 先确认版本（期望：2.9.x / True / 你的卡名 / Python 3.11）
nvidia-smi
python -V
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ② 装依赖（固定 tabpfn 版本，别让它漂移）
pip install xgboost scikit-learn pandas numpy scipy shapely pyproj rasterio pyshp
pip install ./tools/wheels/tabpfn.whl          # 即 8.5.0；或 pip install "tabpfn==8.5.0"
python -c "import tabpfn;print('tabpfn', tabpfn.__version__)"
```

已验证兼容性：本机在 **torch 2.12.1** 上跑通 tabpfn 8.5.0 + 包内 ckpt（比 2.9 更新，故 2.9 无风险）。
若某台机器 torch <2.5，见 §1.1。

**离线安装（服务器不能联网时）**

⚠ 包内 `tools/wheels/` 的**编译型 wheel 是 Windows 版**，Linux 装不上；在服务器上重新抓：

```bash
python tools/fetch_python_pack.py              # 自动按平台+Python 版本选标签（cp311/manylinux）
export PYTHONPATH=$PWD/.deps
```

### 1.1 后备：如果某台机器的 torch 只到 2.4

TabPFN 版本与 torch 下限：**2.x 要求 `torch>=2.1`；6.x/8.x 要求 `torch>=2.5`**。
`tabpfn 2.1.4` 用**同一个 ckpt** 已实测跑通本项目 runner（冒烟 2 折 AUC 0.8194，诊断一致）：

```bash
pip install "tabpfn==2.1.4"                                                    # 联网
python tools/fetch_python_pack.py "tabpfn==2.1.4" && export PYTHONPATH=$PWD/.deps   # 离线
# 或升级 torch（与镜像无关，只看驱动 ≥525；50 系卡必须 cu128 且驱动 ≥570）
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
```

⚠ **tabpfn 版本必须全程固定**：2.x 与 8.x 的默认集成数不同，AUC 有 ≈0.005 的差异。
脚本会把 `tabpfn_version` / `torch_version` 写进 JSON，配对比较前核对一致。

## 2. 权重（已随包携带，无需外网）

`models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`（29 MB，来自 ModelScope 的
`Prior-Labs/TabPFN-v2-clf`；本机 huggingface.co / hf-mirror 均不可达，故走魔搭）。
`baseline_tabpfn.py` 默认就会用它；请务必带上：

```bash
export HF_HUB_OFFLINE=1        # 禁止联网找权重，强制用本地 ckpt
```

## 3. 显存与上下文（重要）

| 口径 | 上下文规模 | 显存建议 |
|---|---|---|
| `--context matched`（病例-对照 k=5） | 约 2.2~3.2k | **8 GB 够用**（推荐先跑这个） |
| `--context soft --max-train 12000` | 约 12k（近区负样本全留） | **建议 ≥16 GB**；8 GB 请降到 `--max-train 6000` |

CPU 实测参考（本机 16 线程、38 维）：上下文 1000 → 预测 400 行约 **32 s**；上下文 2000 → **89 s**。
GPU 会快 1~2 个数量级，但**必须实测**：先用 `--seeds 1` 跑一格计时，再决定 `--seeds 10` 是否划算。

## 4. 冒烟（1 分钟，确认链路）

```bash
export HF_HUB_OFFLINE=1
python baseline_tabpfn.py --smoke --context matched --method spatial_kmeans --n-estimators 2
python baseline_xgb.py --features-csv features/v2/features_v2_train_38.csv \\
    --method spatial_kmeans --seed 42 --neg-sampling matched --neg-k 5
```

## 5. M3 入场实验（2×2：模型 × 人群）—— 务必先做计时探针

**阶段 0（1 格计时，几分钟）**：先只跑一格 matched 口径，看 GPU 上到底多快，
再决定 `--seeds` 取 3 还是 10：

```bash
time python -u baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv \
    --method spatial_kmeans --context matched --match-k 5 --seeds 1
# 看输出的"平均 AUC/采样池 AUC"与 shell 的 real 时间；记录单格耗时 T
```

**阶段 1（四格 × 两分折，估计 1.5–2 小时机时）**

```bash
for M in spatial_kmeans admin; do
  # ① soft 等效人群
  python baseline_xgb.py    --features-csv features/v2/features_v2_train_38.csv \\
      --method $M --neg-sampling soft --neg-lam 0.2 --neg-km 4.0 --seed 42
  python baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv \\
      --method $M --context soft --neg-lam 0.2 --max-train 12000 --seeds 3
  # ② matched 人群（TabPFN 甜区）
  python baseline_xgb.py    --features-csv features/v2/features_v2_train_38.csv \\
      --method $M --neg-sampling matched --neg-k 5 --seed 42
  python baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv \\
      --method $M --context matched --match-k 5 --seeds 3
done
```

**阶段 2（若阶段 1 显示方向为正或接近 MDE，再把重复次数提到 10 以压小零分布 SD）**
把上面的 `--seeds 3` 改为 `--seeds 10`（TabPFN 侧；XGB 侧用 `--seeds` 不支持，改 `--seed` 逐次跑或直接复用阶段 1）。

**后台运行 + 日志**（GPU 实例断连不掉任务）

```bash
tmux new -s m3            # 或 screen -S m3
python -u baseline_tabpfn.py ... 2>&1 | tee logs/tabpfn_matched_spatial.log
# Ctrl-B D 脱离；tmux attach -t m3 回来
```

## 6. 判定（同折同种子配对，自动对照 MDE）

```bash
python tools/paired_compare.py \\
  xgb_matched_spatial=results/runs/r10/...json \\
  tabpfn_matched_spatial=results/runs/tabpfn/baseline_tabpfn_mt5f0_38d_s3.json
```

（`results/runs/` 下的确切文件名由脚本自动生成，含口径/维数/种子数标签；直接 `ls` 选即可。）

**门槛**：Δ 超过该口径 MDE（soft02-spatial **0.0032**、matched5-spatial **0.0081**）、配对显著、
**spatial 与 admin 两套分折同号**，才算"取代 XGBoost"。

## 7. 结果回传

```bash
tar czf runs_$(date +%m%d).tgz results/runs results/*.txt results/*.json docs
```

都很小（几 MB），拉回本地后可直接用 `tools/paired_compare.py` 复核；
**不要回传 GeoTIFF/权重**（本地可用脚本重建）。
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
