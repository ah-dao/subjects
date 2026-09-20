# 服务器 / GPU 训练手册（v2 口径）

> **适用范围**：在云 GPU（AutoDL 等）或本机 GPU 上重跑 **v2 口径**的 GNN / TabPFN 主实验，以及本地 CPU 侧的 v2 训练与评估命令。
> 唯一权威数据口径见 [DATA_SPEC_V2.md](DATA_SPEC_V2.md)。本文的人群 / 路径 / 数字全部按 v2：训练人口 **25,509**、出图人口 **25,939**、唯一 ID `unit_id = 1..25939`、主线表 **`features/v2/features_v2_train.csv`**（34 维，正样本 661）。
>
> **v1 结果全部作废**：v1 的旧 shp / 特征表 / 脚本 / 结果都在 `archive/v1/`。旧 headline —— XGB 0.8131/0.8233、TabPFN 0.8286、GNN 方案 A/B/C（0.7961/0.7365/0.7553）——以及全部旧消融（含产状 / 土壤两轮）**不得再引用**，需在 v2 上重做。
> **当前进度**：v2 XGB 基线已完成（全单元 AUC **0.9093 ± 0.0008** / 采样池 0.8740 / recall@10% 0.6192）；**GNN 与 TabPFN 尚未在 v2 上跑过，本文相关条目均标注「待重跑」**。

---

## 1. 本机环境要点（结合本项目实际踩坑）

### 1.1 GPU 与 torch：本机是 CPU 版，要跑 GPU 必须装 cu128

| 项 | 现状 |
|---|---|
| GPU | **NVIDIA GeForce RTX 5060 Laptop（8,151 MiB ≈ 8 GB，Blackwell，sm_120）**，驱动 577.05 |
| Python | `C:\Users\dollars\.conda\envs\landslide\python.exe`（Python 3.10.20） |
| torch | **`2.12.1+cpu`** —— `torch.version.cuda = None`，`torch.cuda.is_available() = False` |
| 后果 | 目前一切训练只能走 CPU；GPU 加速需先换装 CUDA 12.8 构建 |

装 GPU 版 torch（**≥ 2.7，cu128 构建**）：

```powershell
pip install --index-url https://download.pytorch.org/whl/cu128 torch

# 自检：应输出含 cu128 的版本号与 True
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

> ⚠️ **不要用 torch 2.5.1**：它没有 sm_120 kernel，在 Blackwell（RTX 50 系）上会直接报 `no kernel image is available for execution on the device`；2.7+ 的 cu128 构建才带 sm_120。

### 1.2 TabPFN：离线 wheel + HuggingFace 权重（国内走镜像）

```powershell
pip install _tmppip\tabpfn.whl     # 离线依赖已随包下载（_tmppip\ 内含 huggingface_hub / pydantic / einops / safetensors 等 wheel）
```

- 权重需从 HuggingFace 取（`baseline_tabpfn.py` 默认 `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`）；国内先设镜像端点：

```powershell
$env:HF_ENDPOINT = 'https://hf-mirror.com'
# cmd 下： set HF_ENDPOINT=https://hf-mirror.com
```

- **8 GB 显存下 12k 上下文偏紧**（`--max-train` 默认 12000）：必要时下调到 8k（`--max-train 8000`）或直接 `--device cpu` 回退。
  注意：近区软采样负样本实测可达 ~9k，上下文压得太低会丢掉近区负样本（脚本注释建议不低于 11k）——显存不足时**优先 CPU 回退**，其次才是降上下文。

### 1.3 矢量 IO：绕开 geopandas 的 GDAL 栈

- geopandas 的 GDAL 矢量栈（pyogrio）在本机与 base anaconda 的 GDAL DLL **冲突，会间歇失败**（同一命令时好时坏）。
- 替代：项目自带 **`tills/shp_io.py`**（纯 Python 实现，含 `read_polygons` / `read_lines` / `read_dbf`）。v2 的基线 / 消融脚本已内置该替代（见 `tills/v2_run_baseline.py`、`tills/v2_ablation_inundation.py` 顶部对 `src.dataset.load_centroids*` 的覆盖）。

### 1.4 中文控制台乱码

- Windows 控制台编码会毁中文输出 → **报告一律写 UTF-8 文件再读**。本项目 `results/*.txt` 均以 `encoding='utf-8'` 写出，请用文本读取工具查看，不要靠 stdout 判断数值。
- 同理：**不要用 PowerShell 重写源码文件**（`Set-Content -Encoding UTF8` 会重编码中文串，导致脚本语法错）；改文件用编辑工具。

---

## 2. v2 训练 / 评估命令（本地）

### 2.1 XGBoost 基线（已完成，可复现）

```powershell
python tills\v2_run_baseline.py --tag v2final
# 口径：admin × soft（4 km 内权重 1.0，远区 λ=0.2），5 折 × 3 seeds（42/43/44）
# 输入：features/v2/features_v2_train.csv（25,509 × 34 维，正样本 661）
# 输出：results/v2_baseline_v2aw.txt
# 期望：全单元 AUC 0.9093 ± 0.0008 | 采样池 AUC 0.8740 | recall@10% 0.6192
```

单条 XGB（单 seed，调试 / 消融用）：

```powershell
python baseline_xgb.py --features-csv features/v2/features_v2_train.csv `
    --method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2 --folds 5 --seed 42
```

> ⚠️ `baseline_xgb.py` 的 `--features-csv` **默认仍是 `features/features.csv`**（v1 静态全窗口 21 维对照口径），**必须显式传 v2 表**；`tills/v2_run_baseline.py` 已经替你传好（并跑满 3 个 seed）。特征表第 1 列必须是 `unit_id`（`src/dataset.py::load_features` 按该列取名），不要改成行索引。

### 2.2 消融与数据体检

```powershell
python tills\v2_ablation_inundation.py     # 淹没判定阈值消融 → results/v2_ablation_inundation.txt / .json
python tills\v2_univariate_scan.py         # 单变量判别力扫描 → results/v2_univariate_auc.txt
```

### 2.3 TabPFN（**待重跑**）

```powershell
python baseline_tabpfn.py --seeds 3
# 读 features/v2/features_v2_train.csv + features/v2/county_units_v2_train.csv（config 已切 v2）
# 8 GB 显存吃紧时： --device cpu  或  --max-train 8000
```

- 尚未在 v2 上运行过；旧值 0.8286 作废。仍需先按 §1.1 换 cu128 torch 才能用 GPU。

### 2.4 GNN（**待重跑**，前置条件未满足）

先解决三处 v1 残留，再谈重跑：

1. **v2 图尚未重建**：`src/config.py` 的 `GRAPH_NPZ` 已指向 `features/v2/graph_v2_train.npz`（待生成），旧的 `tills/build_graph.py` 已归档到 `archive/v1/tills/` → 需按 v2 的 25,509 人群与 `unit_id` 重建图。
2. `STUDY_UNITS_COUNT_CSV` / `STUDY_SLOPE_UNITS_SHP` 仍指向 v1 文件（`data/slope_units/study_units_count.csv`、`study_units_fixed.shp`，已随 v1 归档）。
3. `predict_gnn.py` 的 `FULL_SHP` / `SUBMERGED_CSV` / `BACKFILL_MAP` 仍是 v1 路径（`slope_units_fixed.shp`（26,068）/ `submerged_units_combined.csv` / `train_backfill_map.csv`）——v2 出图应改为 `slope_units_final.shp`（25,939）+ 430 个河道内单元回填 prob=0。

接线完成后的命令（命令形态与旧版一致，数据侧已由 config 指向 v2）：

```powershell
# 冒烟（本地 CPU）
python train_gnn.py --plan A --folds 3 --epochs 50

# 正式主实验（服务器 GPU）
python train_gnn.py --plan B --folds 5 --epochs 200 --patience 20 `
    --fold-method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2

# 出图（回本地 CPU）
python predict_gnn.py --plan B --method fixed
```

产出路径（已按 v2 切好，见 `src/config.py`）：

| 文件 | 说明 |
|---|---|
| `results/train_gnn_<plan>.json` | 各折 AUC + 均值 ± std |
| `features/v2/oof_predictions_v2_train.csv` | v2 OOF 外推预测 |
| `models/best_<plan>.pth` / `models/scaler_<plan>.npz` | 最终权重 / MinMax 归一化参数 |

---

## 3. 租什么服务器（结论速览）

| 项目 | 需求 | 说明 |
|---|---|---|
| GPU | **≥8GB 显存即可**（推荐 12–24GB） | 图 **25,509** 节点 × 34 维输入，hidden=64；Transformer 注意力按 512 节点分块（`CHUNK_SIZE`），实测峰值显存预计 <2GB |
| 推荐机型 | **RTX 4090 24GB**（¥2.0–2.6/时）或 **RTX 3090 24GB**（¥1.2–1.6/时） | T4/L4 慢 2–4 倍不建议 |
| 平台 | **AutoDL 首选**（备选：恒源云/智星云/矩池云；免费路线可试百度 AI Studio 算力点） | 新人注册通常送算力券；**无卡模式约 ¥0.1/时**——先无卡开机传数据+装环境，关机后换 GPU 卡开机只计时训练段，本任务总花费 ≤¥3 |
| CPU / 内存 | ≥4 核 / ≥16GB | 特征矩阵仅 3.5MB，需求很低 |
| 磁盘 | 系统盘默认即可（≥20GB 空闲） | 上传物 ~80MB，产出 <30MB |
| 镜像 | 选预装 **CUDA 12.x + PyTorch 2.x** 的镜像 | 本项目**不需要 torch-geometric**（`src/model.py` 自带 SAGEConv）；若租到 **RTX 50 系（Blackwell/sm_120）**机型，需 cu128 构建的 torch ≥ 2.7，见 §1.1 |

**费用估算（4090，单任务）**：5 折 CV 约 60–120 epoch/折（patience=20 早停）×1–3s ≈ 10–20 分钟 + 最终模型 200 epoch ≈ 5–10 分钟 → **合计约 30–45 分钟 ≈ ¥2–4**。
建议顺序：**无卡模式开机**（¥0.1/时）→ 上传 zip + 装环境 → 关机换 4090 开卡 → 冒烟验证 → 正式训练 → 下载结果即关机。

### 3.1 AutoDL 全流程速览（无卡模式 → 换卡训练）

1. **注册充值**：autodl.com 注册 → 支付宝实名 → 充值 ¥10（新人算力券自动抵扣）。
2. **创建实例（先选 GPU）**：容器实例 → 创建实例 → 按量计费 + 选 RTX 4090 + 镜像选预装 CUDA 12.x / PyTorch 2.x 的版本。
   ⚠️ 创建对话框里没有无卡选项——创建必须先选卡（自动有卡开机，按秒计费）。
3. **切无卡模式**：实例**关机**后，实例卡片「**更多**」下拉 →「**无卡模式开机**」（¥0.1/时；新版 UI 也可能出现在点「开机」后的弹窗里）。之后：JupyterLab 进 `/root/autodl-tmp` → 拖拽上传 `server_train.zip` → Terminal 解压（见 §4.2）→ pip 装依赖（见 §5）→ **关机**。
   注意：无卡模式下 `torch.cuda.is_available()` 恒为 False，正常，有卡后再验。
4. **换卡开机**：关机状态同一实例点「开机」→ 弹窗选 **RTX 4090（24GB）+ 按量计费**（无货选 3090，命令不变）。环境与数据自动保留（关机 15 天内免费保留实例数据）。
5. **训练段**：`nvidia-smi` + CUDA 自检（§5）→ 冒烟（§6）→ 正式主实验（§6）→ 跑完 **立即关机**。
6. **收尾**：JupyterLab 勾选下载产出（§7 清单）→ 确认本地齐全后再「释放实例」（释放=删数据，勿急）。

---

## 4. 本地打包与上传（v2 清单）

### 4.1 打包（PowerShell，在 `C:\Users\dollars\code\subjects` 下执行）

> ⚠️ **不要**把 `features/v2/xxx.csv` 这类带子目录的文件直接塞进 `Compress-Archive -Path`——
> 它会把文件**拍平到压缩包根目录**（只有整个目录条目才保留层级）。先在临时目录搭好结构再压缩：

```powershell
$stage = "$env:TEMP\server_stage"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory "$stage\features\v2", "$stage\data\slope_units" -Force | Out-Null

Copy-Item src, train_gnn.py, baseline_xgb.py, baseline_tabpfn.py, predict_gnn.py, requirements.txt $stage -Recurse
Copy-Item features/v2/features_v2_train.csv, features/v2/county_units_v2_train.csv, `
  features/v2/graph_v2_train.npz "$stage\features\v2\"
Copy-Item data/slope_units/slope_units_final_train.shp, data/slope_units/slope_units_final_train.shx, `
  data/slope_units/slope_units_final_train.dbf, data/slope_units/slope_units_final_train.prj, `
  data/slope_units/slope_units_final_train.cpg, data/slope_units/slope_units_final_train_count.csv `
  "$stage\data\slope_units\"

Compress-Archive -Force -Path "$stage\*" -DestinationPath server_train.zip
Remove-Item $stage -Recurse -Force
# 本地抽查压缩包结构（应看到 src/、features/v2/、data/slope_units/ 及各 py 文件）
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::OpenRead("$PWD\server_train.zip").Entries.FullName |
    Select-Object -First 15
```

清单说明（每个文件都被训练链路引用，缺一报错）：

| 文件 | 用途 |
|---|---|
| `src/`、`train_gnn.py`、`requirements.txt` | 训练代码（baseline_xgb / baseline_tabpfn / predict_gnn 顺手带上备用） |
| `features/v2/features_v2_train.csv` | v2 主线 34 维特征表（训练人群 25,509，661 正样本） |
| `features/v2/graph_v2_train.npz` | v2 图邻接 edge_index（25,509 节点）——**待重建，打包前先确认存在** |
| `features/v2/county_units_v2_train.csv` | 单元→县归属（`--fold-method admin` 必需） |
| `data/slope_units/slope_units_final_train.{shp,shx,dbf,prj,cpg}` | 训练单元几何（质心 / 邻域采样用） |
| `data/slope_units/slope_units_final_train_count.csv` | 训练人群计数表（`--weight-scheme none` 也读取） |

> **不传**：`slope_units_final.shp`（出图 25,939，回本地用）、栅格 tif、`features_v2.csv` 全量表——训练用不到。
> 上传后**目录结构必须保持**（`src/`、`features/v2/`、`data/slope_units/` 相对位置不变，`src/config.py` 用 `ROOT` 推导）。

### 4.2 上传

- AutoDL 等国内平台：网页 **JupyterLab → 上传**最省事（~80MB，几分钟）——**先在左侧文件树进入 `/root/autodl-tmp` 再上传**；或本地 `scp -P 端口 server_train.zip root@主机:/root/autodl-tmp/`（-P 大写）。
- 服务器上解压 + 结构自检：

```bash
cd /root/autodl-tmp && unzip server_train.zip -d landslide && cd landslide
ls src features/v2 data/slope_units          # 三处就位即正确

# 路径自检：应输出 True True True True（第二个是 v2 图，未重建时会是 False）
python -c "from src.config import EVENT_WINDOW_FEATURES_CSV, GRAPH_NPZ, COUNTY_UNITS_CSV, study_shp_path; \
print(EVENT_WINDOW_FEATURES_CSV.exists(), GRAPH_NPZ.exists(), COUNTY_UNITS_CSV.exists(), study_shp_path().exists())"
```

---

## 5. 环境安装（用镜像自带环境，无需 conda 新建）

> **恒源云（GpuShare）适配**：镜像 CUDA 12.1.1 + Python 3.11 同样可行（依赖均有 cp311 wheel）。
> 差异：① 数据盘路径是 **`/hy-tmp`**（非 `/root/autodl-tmp`），上传 / 解压 / 训练都在 `/hy-tmp/landslide`；
> ② 框架镜像自带 torch 版本不确定——**先自检**；torch < 2.4 需 `numpy<2` 钉版，≥ 2.4 可不钉（统一钉 `<2` 也零风险）。
> ③ 若实例是 **RTX 50 系（Blackwell / sm_120）**，镜像自带 torch 多半没有 sm_120 kernel，必须换 cu128 构建的 torch ≥ 2.7（见 §1.1）。
> 恒源云数据保留期较短，**产出及时下载**。

```bash
python --version                                              # 3.10（本地）/ 3.11 / 3.12 均可
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 例：cu121 镜像输出 2.x+cu121；cu124 镜像输出 2.5.1+cu124；Blackwell 机型需 2.7+cu128

# 其余依赖（torch 镜像已自带，跳过安装）
pip install "numpy<2" pandas scipy scikit-learn matplotlib tqdm \
            geopandas shapely rasterio rasterstats xlrd xgboost

# 依赖自检（无卡模式下也会通过；cuda:False 属正常，有卡开机后再验 True）
python -c "import torch, numpy, pandas, sklearn, geopandas, rasterio; \
print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

> 本仓库无需 torch-geometric / jenkspy；方案 C 才需要 performer-pytorch。
> TabPFN（若要在服务器上跑）：`pip install _tmppip/tabpfn.whl`（或 `pip install tabpfn`），国内先 `export HF_ENDPOINT=https://hf-mirror.com`。

---

## 6. 冒烟验证与正式主实验（**GNN 前置未满足，属待重跑**）

> 先决条件：v2 图 `features/v2/graph_v2_train.npz` 已重建，且 §2.4 列出的三处 v1 残留已接线。否则 `train_gnn.py` 会因找不到图 / v1 计数表而报错。

冒烟（3 分钟，确认 CUDA + pipeline + 摸清单 epoch 耗时）：

```bash
nohup python -u train_gnn.py --plan A --folds 3 --epochs 50 > log_smoke.txt 2>&1 &
tail -f log_smoke.txt
```

- 日志开头应打印 `设备: cuda`；随后每 10 epoch 一行 `epoch … | loss … | val_auc … | Xs`。
- 记下 **A 方案单 epoch 秒数 ×3 ≈ B 方案单 epoch 秒数**，据此估算总时长：
  总时长 ≈ 5 折 × 早停折数(60–120) × t_B + 最终模型 200 × t_B。例：t_B = 2s → 5×90×2 + 200×2 = 1300s ≈ 22 分钟。
- 冒烟正常后 Ctrl-C 停掉 tail（后台任务可继续，或 `pkill -f "plan A"`）。

正式主实验（单条命令）：

```bash
nohup python -u train_gnn.py --plan B --folds 5 --epochs 200 --patience 20 \
    --fold-method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2 \
    > log_train_B.txt 2>&1 &
tail -f log_train_B.txt
```

预期执行序列与产出：

1. **5 折交叉验证**：每折开头打印 `[软负采样] 4km, λ=0.2 | 邻近负 … / 远区负 … | ESS …` 与 `=== Fold i/5 | … pos_weight … ===`；每 10 epoch 打印 loss/val_auc；早停 20 epoch 后进入下一折。结束打印各折 AUC + `平均 AUC: x.xxxx ± x.xxxx` 与采样池 AUC。
2. **最终模型**（全数据，200 epochs）→ 保存 `models/best_B.pth`、`models/scaler_B.npz`。
3. 结果汇总 → `results/train_gnn_B.json`；OOF 外推 → `features/v2/oof_predictions_v2_train.csv`（**会覆盖本地同名文件**，属预期；先备份旧的）。

> 可选对照（本次主实验不跑）：同命令改 `--neg-sampling none` 得 admin×全域对照；`--plan A` 得纯 GraphSAGE 基线。多跑项各加 ~10–15 分钟。
> 若某折 epoch 耗时异常（>10s），检查是否真在用 GPU（`nvidia-smi`）。

---

## 7. 带回与本地出图

下载到本地 `C:\Users\dollars\code\subjects\` 对应目录：

```
models/best_B.pth                     models/scaler_B.npz
results/train_gnn_B.json              features/v2/oof_predictions_v2_train.csv（覆盖前先备份）
log_train_B.txt                       （日志，存档用）
```

本地（CPU，几分钟）出全量图：

```powershell
python predict_gnn.py --plan B
# 产出：predictions/ 下的 5 级易发性 shp + 概率 CSV + statistics.txt + 示意图
```

> ⚠️ `predict_gnn.py` 目前仍指向 v1 文件（`data/slope_units/slope_units_fixed.shp` 全量 26,068、`features/submerged_units_combined.csv`、`features/train_backfill_map.csv`，均已归档）——**v2 出图前必须先改这三处**：全量 shp 用 `data/slope_units/slope_units_final.shp`（25,939），河道内单元（430 个）按 `features/v2/unit_id_map_v2.csv` 的 `is_submerged` 回填 prob=0。

跑完后把 `results/train_gnn_B.json` 的 AUC 记入 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)，与 **v2 XGB 基线 0.9093 ± 0.0008** 对照（v1 的 0.8233 不再是参照物）。

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 本机即如此（装的是 `2.12.1+cpu`）：装 **cu128 构建的 torch ≥ 2.7** —— `pip install --index-url https://download.pytorch.org/whl/cu128 torch`；服务器上则确认镜像自带 torch 是否匹配机型（Blackwell 必须 cu128） |
| 报 `no kernel image is available` / sm_120 | torch 版本没有 Blackwell kernel（如 2.5.1）→ 换 cu128 构建 ≥ 2.7 |
| geopandas 读矢量间歇失败（GDAL/pyogrio DLL 冲突） | 别重试 geopandas：改 `tills/shp_io.py`（`read_polygons` / `read_lines` / `read_dbf`）；v2 基线 / 消融脚本已内置替代 |
| 控制台中文乱码、数值看不准 | 报告一律写 UTF-8 文件（`results/*.txt`）再读；不要用 PowerShell 重写源码（会损坏中文串） |
| `train_gnn.py` 报找不到图 / 找不到计数表 | v2 图 `features/v2/graph_v2_train.npz` 尚未重建；`STUDY_UNITS_COUNT_CSV` 等仍指 v1（已归档）→ 见 §2.4 |
| unzip 后找不到 `features/`、`data/`（文件散在根目录） | 旧打包命令把文件拍平了：服务器上 `mkdir -p features/v2 data/slope_units` 后把散文件 `mv` 归位，再用 §4.2 末尾的 config exists 自检验证；本地重新打包请用 §4.1 修正版命令 |
| unzip: cannot find server_train.zip | zip 没传到位：`find /root -name "*.zip"` 定位；没传就用 JupyterLab（先进 `/root/autodl-tmp` 再上传）或 `scp -P 端口` 重传 |
| TabPFN 下载权重失败 | 设 `HF_ENDPOINT=https://hf-mirror.com`（PowerShell：`$env:HF_ENDPOINT='https://hf-mirror.com'`）；确认用的是 `--model-path` 默认的非 gated 权重名 |
| TabPFN 显存不足 / OOM | 8 GB 下 12k 上下文偏紧：先 `--device cpu` 回退，再考虑 `--max-train 8000`（注意近区负样本可达 ~9k） |
| 日志报 GDAL/GDAL_DATA 警告、shp 属性编码警告 | 无害，忽略 |
| SSH 断开训练中断 | 已用 nohup 防断；重连后 `tail -f log_train_B.txt` 继续观察，模型照常落盘 |
| 磁盘/内存不够 | 上传物 <100MB、特征矩阵 3.5MB，系统盘 20GB+ 即可 |
| 想提前结束 | `pkill -f train_gnn.py`；已完成的折结果不落盘（JSON 只在全部结束后写）——重跑会从头开始，请留足时长再开机 |
| 时间预算紧张 | 先把 `--epochs 200 --patience 20` 收紧为 `--epochs 100 --patience 15`（早停一般 60–120 epoch 内触发，影响小） |
