# 服务器训练手册：GNN 方案 B 主实验（GraphSAGE×2 + Transformer）

> 目的：明天在云 GPU 服务器上跑 **34 维主线 × plan B × 5 折 admin × 软采样 λ=0.2** 的正式 GNN 训练
> （XGBoost 同口径基线 AUC 0.8233 ± 0.0234，见 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)）。
> 训练全部在服务器完成，出图（predict_gnn.py，方案 C 水下单元回填）回本地 CPU 跑。
> 训练命令与文档 §5 一致：`python train_gnn.py --plan B --folds 5 --fold-method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2`。

---

## 1. 租什么服务器（结论速览）

| 项目 | 需求 | 说明 |
|---|---|---|
| GPU | **≥8GB 显存即可**（推荐 12–24GB） | 图 25,636 节点 × 34 维输入，hidden=64；Transformer 注意力按 512 节点分块（`CHUNK_SIZE`），实测峰值显存预计 <2GB，8GB 卡有余量做 hidden=128 等扩展 |
| 推荐机型 | **RTX 4090 24GB**（¥2.0–2.6/时）或 **RTX 3090 24GB**（¥1.2–1.6/时） | T4/L4 慢 2–4 倍不建议 |
| 平台 | **AutoDL 首选**（备选：恒源云/智星云/矩池云；免费路线可试百度 AI Studio 算力点） | 新人注册通常送算力券；**无卡模式约 ¥0.1/时**——先无卡开机传数据+装环境，关机后换 GPU 卡开机只计时训练段，本任务总花费 ≤¥3 |
| CPU / 内存 | ≥4 核 / ≥16GB | 特征矩阵仅 3.5MB，需求很低 |
| 磁盘 | 系统盘默认即可（≥20GB 空闲） | 全部上传物 **~80MB**，产出 <30MB |
| 镜像 | 选预装 **CUDA 12.x + PyTorch 2.x** 的镜像（或 python 3.10 基础镜像后自装） | 本项目**不需要 torch-geometric**（src/model.py 自带 SAGEConv），环境极简 |

**费用估算（4090 上，单任务）**：5 折 CV 约 60–120 epoch/折（patience=20 早停）×1–3s ≈ 10–20 分钟
+ 最终模型 200 epoch ≈ 5–10 分钟 → **合计约 30–45 分钟 ≈ ¥2–4**。
建议顺序：**无卡模式开机**（¥0.1/时）→ 上传 zip + 装环境 → 关机换 4090 开卡 → 冒烟验证 → 正式训练 → 下载结果即关机。

### 1a. AutoDL 全流程速览（无卡模式 → 换卡训练）

1. **注册充值**：autodl.com 注册 → 支付宝实名 → 充值 ¥10（新人算力券自动抵扣）。
2. **创建实例（先选 GPU）**：容器实例 → 创建实例 → 按量计费 + 选 RTX 4090 +
   镜像选 **PyTorch 2.5.1 / Python 3.12 / CUDA 12.4**（numpy 2 无需钉版；
   若只有 torch 2.3 的镜像可选，需按 §3 旧注钉 numpy<2）。
   ⚠️ 创建对话框里没有无卡选项——创建必须先选卡（自动有卡开机，按秒计费）。
3. **切无卡模式**：实例**关机**后，实例卡片「**更多**」下拉 →「**无卡模式开机**」（¥0.1/时；
   新版 UI 也可能出现在点「开机」后的弹窗里）。之后：JupyterLab 进 `/root/autodl-tmp` →
   拖拽上传 `server_train.zip` → Terminal 解压（见 §2.2）→ pip 装依赖（见 §3，torch 优先用镜像自带的）→ **关机**。
   注意：无卡模式下 `torch.cuda.is_available()` 恒为 False，正常，有卡后再验。
4. **换卡开机**：关机状态同一实例点「开机」→ 弹窗选 **RTX 4090（24GB）+ 按量计费**（无货选 3090，命令不变）。
   环境与数据自动保留（关机 15 天内免费保留实例数据）。
5. **训练段**：`nvidia-smi` + CUDA 自检（§3）→ 冒烟（§4）→ 正式主实验（§5）→ 跑完 **立即关机**。
6. **收尾**：JupyterLab 勾选下载产出（§6 清单）→ 确认本地齐全后再「释放实例」（释放=删数据，勿急）。

---

## 2. 本地打包与上传

### 2.1 打包（PowerShell，在 `C:\Users\dollars\code\subjects` 下执行）

> ⚠️ **不要**把 `features/xxx.csv` 这类带子目录的文件直接塞进 `Compress-Archive -Path`——
> 它会把文件**拍平到压缩包根目录**（只有整个目录条目才保留层级）。先在临时目录搭好结构再压缩：

```powershell
$stage = "$env:TEMP\server_stage"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory "$stage\features", "$stage\data\slope_units" -Force | Out-Null

Copy-Item src, train_gnn.py, baseline_xgb.py, predict_gnn.py, requirements.txt $stage -Recurse
Copy-Item features/event_window_features_k2_v34_train.csv, features/graph_train.npz, `
  features/county_units_train.csv, features/submerged_units_combined.csv, `
  features/train_backfill_map.csv "$stage\features\"
Copy-Item data/slope_units/slope_units_train.shp, data/slope_units/slope_units_train.shx, `
  data/slope_units/slope_units_train.dbf, data/slope_units/slope_units_train.prj, `
  data/slope_units/slope_units_train.cpg, data/slope_units/slope_units_train_count.csv, `
  data/slope_units/study_units_count.csv "$stage\data\slope_units\"

Compress-Archive -Force -Path "$stage\*" -DestinationPath server_train.zip
Remove-Item $stage -Recurse -Force
# 本地抽查压缩包结构（应看到 src/、features/、data/slope_units/ 及各 py 文件）
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::OpenRead("$PWD\server_train.zip").Entries.FullName |
    Select-Object -First 15
```

清单说明（每个文件都被训练链路引用，缺一报错）：

| 文件 | 用途 |
|---|---|
| `src/`、`train_gnn.py`、`requirements.txt` | 训练代码（baseline_xgb/predict_gnn 顺手带上备用） |
| `features/event_window_features_k2_v34_train.csv` | 34 维特征表（训练人群 25,636，661 正） |
| `features/graph_train.npz` | 图邻接 edge_index（25,636 节点） |
| `features/county_units_train.csv` | 单元→县归属（`--fold-method admin` 必需） |
| `features/submerged_units_combined.csv` + `train_backfill_map.csv` | 仅 predict 用（可先不传，本地已有） |
| `data/slope_units/slope_units_train.{shp,shx,dbf,prj,cpg}` | 训练单元几何（质心/邻域采样用） |
| `data/slope_units/slope_units_train_count.csv` | 训练人群计数表（weight-scheme none 也读取，见 src/dataset.py load_sample_weights） |
| `data/slope_units/study_units_count.csv` | `STUDY_UNITS_COUNT_CSV` 指向的历史计数表（train_gnn 读取） |

> **不传**：`slope_units_fixed.shp`（全量 26068，predict 出图回本地用）、栅格 tif、CSV 全量表——训练用不到。
> 上传后**目录结构必须保持**（src/、features/、data/slope_units/ 相对位置不变，config 用 `ROOT` 推导）。

### 2.2 上传

- AutoDL 等国内平台：网页 **JupyterLab → 上传** 最省事（~80MB，几分钟）——**先在左侧文件树进入
  `/root/autodl-tmp` 再上传**；或本地 `scp -P 端口 server_train.zip root@主机:/root/autodl-tmp/`（-P 大写）。
- 服务器上解压 + 结构自检：

```bash
cd /root/autodl-tmp && unzip server_train.zip -d landslide && cd landslide
ls src features data/slope_units               # 三目录就位即正确

# 路径自检：应输出 True True True True
python -c "from src.config import EVENT_WINDOW_FEATURES_CSV, GRAPH_NPZ, COUNTY_UNITS_CSV, study_shp_path; \
print(EVENT_WINDOW_FEATURES_CSV.exists(), GRAPH_NPZ.exists(), COUNTY_UNITS_CSV.exists(), study_shp_path().exists())"
```

---

## 3. 环境安装（用镜像自带环境，无需 conda 新建）

> **恒源云（GpuShare）适配**：镜像 CUDA 12.1.1 + Python 3.11 同样可行（依赖均有 cp311 wheel）。
> 两处差异：① 数据盘路径是 **`/hy-tmp`**（非 autodl-tmp），上传/解压/训练都在 `/hy-tmp/landslide`；
> ② cu121 框架镜像自带 torch 版本不确定（2.1~2.5 都可能）——**先自检 torch 版本**：
> torch <2.4 必须 `numpy<2` 钉版；≥2.4 可不钉（统一钉 `<2` 也零风险，下面命令已统一钉）。
> 上传用其 JupyterLab 或 oss 工具/scp；恒源云数据保留期较短，**产出及时下载**。
> AutoDL 镜像 **PyTorch 2.5.1 / Python 3.12 / CUDA 12.4**：自带 GPU 版 torch 2.5.1（≥2.4，
> 官方兼容 numpy 2.x，钉不钉均可）；本仓库代码与全部依赖在 Python 3.10/3.11/3.12 下都有预编译 wheel。

```bash
python --version                                              # 3.11（恒源云）/ 3.12（AutoDL）
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 恒源云 cu121 镜像应输出 2.x+cu121；AutoDL 12.4 镜像输出 2.5.1+cu124

# 其余依赖（torch 镜像已自带，跳过安装；numpy<2 在任何 torch 2.x 下都安全）
pip install "numpy<2" pandas scipy scikit-learn matplotlib tqdm \
            geopandas shapely rasterio rasterstats xlrd xgboost

# 依赖自检（无卡模式下也会通过；cuda:False 属正常，有卡开机后再验 True）
python -c "import torch, numpy, pandas, sklearn, geopandas, rasterio; \
print('torch', torch.__version__, '| numpy', numpy.__version__)"
```

> 本仓库无需 torch-geometric / jenkspy / performer-pytorch（方案 C 才需要 performer）。

---

## 4. 冒烟验证（3 分钟，先确认 CUDA + pipeline + 摸清单 epoch 耗时）

```bash
nohup python -u train_gnn.py --plan A --folds 3 --epochs 50 > log_smoke.txt 2>&1 &
tail -f log_smoke.txt
```

- 日志开头应打印 `设备: cuda`；随后每 10 epoch 一行 `epoch … | loss … | val_auc … | Xs`。
- 记下 **A 方案单 epoch 秒数 ×3 ≈ B 方案单 epoch 秒数**，据此估算正式任务总时长：
  总时长 ≈ 5 折 × 早停折数(60–120) × t_B + 最终模型 200 × t_B。
  例：t_B = 2s → 5×90×2 + 200×2 = 1300s ≈ 22 分钟。
- 冒烟正常后 Ctrl-C 停掉 tail（后台任务继续跑完无所谓，可直接 kill：`pkill -f "plan A"` 或等它结束）。

---

## 5. 正式主实验（单条命令）

```bash
nohup python -u train_gnn.py --plan B --folds 5 --epochs 200 --patience 20 \
    --fold-method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2 \
    > log_train_B.txt 2>&1 &
tail -f log_train_B.txt
```

预期执行序列与产出：

1. **5 折交叉验证**：每折开头打印 `[软负采样] 4km, λ=0.2 | 邻近负 … / 远区负 … | ESS …` 与
   `=== Fold i/5 | … pos_weight … ===`；每 10 epoch 打印 loss/val_auc；早停 20 epoch 后进入下一折。
   结束打印各折 AUC + `平均 AUC: x.xxxx ± x.xxxx` 与采样池 AUC。
2. **最终模型**（全数据，200 epochs）→ 保存：
   - `models/best_B.pth`（权重 + 超参）
   - `models/scaler_B.npz`（minmax 归一化参数，predict 必需）
3. 结果汇总 → `results/train_gnn_B.json`；OOF 外推 → `features/oof_predictions_train.csv`
   （**会覆盖本地同名文件**，属预期；先备份本地旧的再下载）。

> 可选对照（本次主实验不跑）：同命令改 `--neg-sampling none` 得 admin×全域对照；
> `--plan A` 得纯 GraphSAGE 基线。多跑项各加 ~10–15 分钟。
> 全程约 30–45 分钟。若某折 epoch 耗时异常（>10s），检查是否真在用 GPU（`nvidia-smi`）。

---

## 6. 带回与本地出图

下载到本地 `C:\Users\dollars\code\subjects\` 对应目录：

```
models/best_B.pth                     models/scaler_B.npz
results/train_gnn_B.json              features/oof_predictions_train.csv（覆盖前先备份）
log_train_B.txt                       （日志，存档用）
```

本地（CPU，几分钟）出全量图：

```powershell
python predict_gnn.py --plan B
# 产出：predictions/susceptibility_units_B_full.shp（26068 全量，432 水下单元回填 prob=0/1级）
#       predictions/gnn_B_probabilities.csv / statistics_B.txt / susceptibility_map_B.png
```

然后把 `results/train_gnn_B.json` 的 AUC 记入 `docs/EXPERIMENT_RESULTS.md`
（对照 XGB 0.8233，验证 Transformer 表达力增益，对应 §4 结论第 5 条"GNN-B 待服务器训练后补充"）。

---

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | 驱动过旧或装了 CPU 版：`pip install torch --index-url https://download.pytorch.org/whl/cu121`（cu 号按 `nvidia-smi` 驱动支持的最高版本选）；换预装 CUDA 的镜像最省事 |
| unzip 后找不到 `features/`、`data/`（文件散在根目录） | 旧打包命令把文件拍平了：服务器上 `mkdir -p features data/slope_units` 后把散文件 `mv` 归位（5 个特征文件 → features/；`slope_units_train.*` + `study_units_count.csv` → data/slope_units/），再用 §2.2 末尾的 config exists 自检验证；本地重新打包请用 §2.1 修正版命令 |
| unzip: cannot find server_train.zip | zip 没传到位：`find /root -name "*.zip"` 定位；没传就用 JupyterLab（先进 `/root/autodl-tmp` 再上传）或 `scp -P 端口` 重传 |
| train_gnn 报 `NameError: proximity_weights / NEG_LAM` 或 cent_utm 为 None | 旧版 zip 的 src/train.py 软采样路径未接线（2024 修复前打包）：从本地仓库重传**已修复的 src/train.py** 覆盖 `src/` 后重跑，无需重新解压 |
| 日志报 GDAL/GDAL_DATA 警告 | 无害，忽略 |
| shp 读取出属性编码警告 | 无害（本链路只用几何质心，不读中文字段） |
| SSH 断开训练中断 | 已用 nohup 防断；重连后 `tail -f log_train_B.txt` 继续观察，模型照常落盘 |
| 磁盘/内存不够 | 上传物 <100MB、特征矩阵 3.5MB，系统盘 20GB+ 即可，无需扩容 |
| 想提前结束 | `pkill -f train_gnn.py`；已完成的折结果不落盘（JSON 只在全部结束后写）——重跑会从头开始，请留足时长再开机 |
| 时间预算紧张 | 先把 `--epochs 200 --patience 20` 收紧为 `--epochs 100 --patience 15`（早停一般 60–120 epoch 内触发，影响小）；或 4090 卡直接跑完最省心 |
