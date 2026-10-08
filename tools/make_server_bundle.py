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
DATA = [  # (相对路径, 说明, 是否可选, 是否进瘦身包)
    ('features/v2/features_v2_train_38.csv', '38 维主线（训练 25,509）', False, True),
    ('features/v2/features_v2_38.csv', '38 维主线（全量 25,939，出图用）', False, True),
    ('features/v2/features_v2_train.csv', '34 维对照', False, False),
    ('features/v2/features_v2.csv', '34 维对照（全量）', False, False),
    ('features/v2/unit_id_map_v2.csv', 'ID 对照表', False, True),
    ('features/v2/county_units_v2_train.csv', 'admin 分折用', False, True),
    ('features/v2/county_units_v2.csv', 'admin 分折用（全量）', False, True),
    ('features/v2/groups', '分组特征（地形/水系/道路/土壤/岩性/产状/力学等）', False, 'slim_groups'),
    ('features/v2/sources/daily_station_levels.csv', 'M1 逐日库水位', False, True),
    ('features/v2/sources/station_assign_v2.csv', '单元→就近水位站映射', False, True),
    ('data/slope_units', '斜坡单元 shp + 计数表（质心/几何/ID）', False, 'slim_units'),
    ('models/tabpfn', 'TabPFN v2 权重（29 MB，来自 ModelScope，服务器免 HF）', False, False),
    ('data/terrain/Terrain_MultiBand.tif', '30 m 地形栈（M2 用）', True, False),
    ('features/v2/sources/ndvi_unit_matrix.csv', 'GEE NDVI 矩阵（可选）', True, False),
    ('features/v2/sources/rain_unit_matrix.csv', 'GEE 降雨矩阵（可选）', True, False),
]
# 瘦身包：只带 entry 实验必需的东西（分组表只留匹配诊断用的 2 个 + M1 用的高程分位）
SLIM_GROUPS = ['terrain.csv', 'water.csv', 'elevation_quantiles.csv', 'factors_selected.csv']
SLIM_UNITS = ['slope_units_final.shp', 'slope_units_final.shx', 'slope_units_final.dbf',
              'slope_units_final.prj', 'slope_units_final.cpg',
              'slope_units_final_train.shp', 'slope_units_final_train.shx',
              'slope_units_final_train.dbf', 'slope_units_final_train.prj',
              'slope_units_final_train.cpg',
              'slope_units_final_count.csv', 'slope_units_final_train_count.csv']
SLIM_WHEELS = ['tabpfn-8.5.0-py3-none-any.whl', 'tabpfn-2.1.4-py3-none-any.whl']
SKIP_DIRS = {'__pycache__', '.git', 'archive', 'dist', '.deps', '.deps2', '_archive'}


def add_path(zf, rel, optional=False, pick=None):
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
                if pick is not None and f not in pick:
                    continue
                full = os.path.join(dirpath, f)
                zf.write(full, os.path.relpath(full, ROOT).replace(os.sep, '/'))
                n += 1
    sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(p)
             for f in fs if (pick is None or f in pick)) \
        if os.path.isdir(p) else os.path.getsize(p)
    print('  %-56s %5d 个文件 %8.1f MB' % (rel, n, sz / 1e6))
    return n


def setup_md(with_terrain, with_gee, slim=False):
    fetch_ckpt = '''
## 2. 权重：在服务器上现取（瘦身包不含权重）

国内直连 ModelScope 很快（本机实测 29 MB / 2.3 秒），比上传快得多：

```bash
python - <<'PY'
import os, shutil, urllib.request
url = ('https://www.modelscope.cn/api/v1/models/Prior-Labs/TabPFN-v2-clf/repo'
       '?Revision=master&FilePath=tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')
os.makedirs('models/tabpfn', exist_ok=True)
dst = 'models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=180) as r, open(dst, 'wb') as f:
    shutil.copyfileobj(r, f, 1 << 18)
print('downloaded %.1f MB -> %s' % (os.path.getsize(dst) / 1e6, dst))
PY
bash tools/verify_ckpt.sh 2>/dev/null || python -c "import os;print('ckpt', os.path.getsize('models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt')/1e6, 'MB')"
```

> 若 ModelScope 也慢，回退方案：本地 `dist/` 里的完整包内含权重，或用恒源云**公共网盘**中转。
''' if slim else '''
## 2. 权重（已随包携带，无需外网）

`models/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`（29 MB，来自 ModelScope 的
`Prior-Labs/TabPFN-v2-clf`；本机 huggingface.co / hf-mirror 均不可达，故走魔搭）。
`baseline_tabpfn.py` 默认就会用它；请务必带上：

```bash
export HF_HUB_OFFLINE=1        # 禁止联网找权重，强制用本地 ckpt
```
'''
    return f"""# 服务器运行包（模型线：M3 入场实验｜{'瘦身版' if slim else '完整版'}）

> 生成于 {datetime.date.today()}　｜　由 `tools/make_server_bundle.py` 打包
> 详细方案见 `docs/当前模型方案.md`；本文件只讲**怎么在服务器上跑起来**。

## 0. 目录约定

解压到 GPU 实例的数据盘（AutoDL: `/root/autodl-tmp/`；恒源云: `/hy-tmp/` 或你的数据盘）：

```bash
cd /root/autodl-tmp && unzip server_bundle_*.zip -d subjects && cd subjects
export PYTHONPATH=$PWD/.deps            # 若用离线 wheel 安装，见下
```

### 0.1 先激活环境（否则 `import torch` 报 ModuleNotFoundError）

恒源云官方文档明确：**登录终端默认不自动进入 base 环境**，此时 `python` 是系统解释器，
而 torch 装在 miniconda 的环境里 → 于是 `ModuleNotFoundError: No module named 'torch'`。

```bash
conda activate base        # 或 conda env list 看 torch 在哪个环境，再激活它
conda config --set auto_activate_base true   # 可选：以后登录自动进 base
python -V && python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

装包统一用 `python -m pip install ...`（而不是裸 `pip`），避免装到别的解释器里。

环境自检一把过（缺哪个包会直接列出来）：

```bash
python - <<'PY'
import sys, importlib
print('python', sys.version.split()[0], '|', sys.executable)
try:
    import torch
    print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
          '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
except Exception as e:
    print('torch 缺失 ->', e)
for m in ['tabpfn','xgboost','sklearn','pandas','numpy','scipy','shapely','pyproj','rasterio','shapefile']:
    try:
        mod = importlib.import_module(m); print('  OK  %-10s %s' % (m, getattr(mod,'__version__','?')))
    except Exception:
        print('  MISS %-9s <-- 需要装' % m)
PY
```

若 base 里确实没有 torch（镜像只带 CUDA）：`python -m pip install torch --index-url https://download.pytorch.org/whl/cu128`。
磁盘紧张时按官方建议把虚拟环境放到数据盘：`conda create -p /hy-tmp/myenv python=3.11`。

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
pip install ./tools/wheels/tabpfn-8.5.0-py3-none-any.whl          # 即 8.5.0；或 pip install "tabpfn==8.5.0"
python -c "import tabpfn;print('tabpfn', tabpfn.__version__)"
```

已验证兼容性：本机在 **torch 2.12.1** 上跑通 tabpfn 8.5.0 + 包内 ckpt（比 2.9 更新，故 2.9 无风险）。
若某台机器 torch <2.5，见 §1.1。

### 1.2 pip 提速：换国内镜像源（服务器下载慢时必做）

```bash
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.timeout 60
python -m pip config list          # 确认生效
```

常用 PyPI 镜像（任选，就近选速度最快的）：

| 镜像 | 地址 |
|---|---|
| 清华 TUNA | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple/` |
| 腾讯云 | `https://mirrors.cloud.tencent.com/pypi/simple` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple` |
| 华为云 | `https://repo.huaweicloud.com/repository/pypi/simple` |

临时用（不改配置）：`python -m pip install -i <镜像> <包>`

**两条"不要装"的省钱要点**（能省 2~3 GB 下载）：

1. **不要装 torch** —— 镜像已自带（2.9+cu128）。装了只会覆盖且白下 2.5 GB。
2. **不要装 geopandas** —— 本项目已把 `src/dataset.py` 的质心读取改成"geopandas 不可用时
   自动回退到 `tills/shp_io.py`"，因此**不需要 GDAL/pyogrio/fiona**（省数百 MB 且避免 DLL 问题）。
   只需 `shapely pyproj rasterio pyshp`。

<details><summary>sqlite/编码等偶发报错</summary>

`python -m pip install --upgrade pip setuptools wheel` 后再试；或加 `--no-cache-dir`。
</details>

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

## 7. 结果回传（**重要：/hy-tmp 关机 24 小时后会被清空**）

先落一份"清单+环境指纹"，再打包（JSON 自带口径/分折/维数/种子字段，可回溯命令行）：

```bash
cd /hy-tmp/subjects
mkdir -p logs
(
  echo "== 导出时间 =="; date
  echo "== 环境 =="; python -V
  python - <<'PY'
import torch, tabpfn
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
print('tabpfn', tabpfn.__version__)
PY
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -m pip freeze | grep -Ei "tabpfn|torch|xgboost|scikit|pandas|numpy|scipy|shapely|pyproj|rasterio|pyshp"
  echo "== 结果文件清单 =="
  ls -l results/*.json results/*.txt results/runs/tabpfn/*.json 2>/dev/null
) > EXPORT_MANIFEST.txt

tar czf export_$(date +%m%d_%H%M).tgz EXPORT_MANIFEST.txt results/*.json results/*.txt results/runs logs 2>/dev/null
ls -lh export_*.tgz
```

下载方式：JupyterLab 直接下 / 平台"数据下载" / 或先传到**个人数据（云盘）**长期保存。
**不要回传** `features/`、权重、GeoTIFF、`predictions/`（本地已有或可重建）。
拉回本地后：`python tools/paired_compare.py xgb=...json tabpfn=...json`

### 7.1 保存环境（下次少装 10 分钟）

平台给的"备份镜像"只保存**系统盘环境**（conda 环境/依赖），**不含 `/hy-tmp`**；
而 `/hy-tmp` 关机 24 小时即清空。所以：

1. **环境** → 控制台 `我的实例 → 实例管理 → 创建备份镜像`；下次创建实例时选"自定义镜像"，开机即可用。
2. **数据与结果** → 上传到 **个人数据（云盘）**（可随实例迁移、长期保存），或下载到本地。
3. 另一条轻量路线：把依赖清单存成文件，下次重装即可（我们依赖很少，约 1 分钟）
   —— 见 `EXPORT_MANIFEST.txt` 里的 `pip freeze` 片段。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-terrain', action='store_true')
    ap.add_argument('--with-gee', action='store_true')
    ap.add_argument('--slim', action='store_true',
                    help='瘦身包（约 25 MB）：只带 M3 入场实验必需的文件，'
                         '权重改在服务器上从 ModelScope 现取')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    os.makedirs(DIST, exist_ok=True)
    out = args.out or os.path.join(
        DIST, 'server_bundle_%s%s.zip' % (datetime.date.today().strftime('%m%d'),
                                          '_slim' if args.slim else ''))
    print('打包 -> %s%s' % (out, '（瘦身模式）' if args.slim else ''))
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in CODE:
            add_path(zf, rel, pick=(SLIM_WHEELS if (args.slim and rel == 'tools') else None))
        for rel in CODE_FILES:
            add_path(zf, rel, optional=True)
        for rel, _desc, opt, slim in DATA:
            if args.slim and slim is False:
                continue
            if rel.endswith('Terrain_MultiBand.tif') and not args.with_terrain:
                continue
            if '_unit_matrix.csv' in rel and not args.with_gee:
                continue
            pick = None
            if args.slim and slim == 'slim_groups':
                pick = set(SLIM_GROUPS)
            elif args.slim and slim == 'slim_units':
                pick = set(SLIM_UNITS)
            add_path(zf, rel, optional=opt, pick=pick)
        zf.writestr('SETUP.md', setup_md(args.with_terrain, args.with_gee, args.slim))
    print('\n完成：%s（%.1f MB）' % (out, os.path.getsize(out) / 1e6))
    print('服务器上：unzip + 按 SETUP.md 执行')


if __name__ == '__main__':
    main()
