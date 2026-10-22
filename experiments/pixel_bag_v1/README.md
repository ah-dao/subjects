# 像元袋独立实验 v1

> 首轮完整空间验证已经完成。后续补种子、行政验证与表格/均值池化对照，请使用 [FOLLOWUP_V11.md](FOLLOWUP_V11.md) 和 `pixel_bag_v11_patch.zip`；不要重做原始袋或旧缓存。以下首次准备流程保留供追溯。

本目录是新实验，原有 `src/`、`tills/`、主线特征表和制图结果均不修改。

## 已有代码与本次复用

- `tills/v2_build_unit_grid.py`、`v2_build_unit_bags.py`：服务器已经运行的数据准备步骤；原始输出继续使用。
- `tills/v2_model_mil.py`：保留作为旧探索版本。本实验不导入其全人口标准化、全局 OOF 残差底座或训练循环。
- `src/dataset.py`：仅复用行政整县分折、训练折内软负样本权重函数；不使用标签传播。
- `tills/shp_io.py`：复用轻量几何读取；训练表与几何通过 `unit_id` 显式对齐。
- `run.py`：独立完成 prepare/train/export；默认读项目现有数据，仅向新目录输出。输出目录已经存在时直接报错，避免覆盖。

## 当前数据结论

服务器实际得到 8,366,561 个原始像元、25,929 个有像元单元。训练正样本均有至少 8 个有效像元。10 个原本无像元的负单元继续保留。

极端 TRI 与负曲率几乎逐点相等，来源尚未确定。本实验只使用高程、坡度、坡向 sin/cos；曲率和 TRI 暂时全部排除。高程/坡度/坡向为非有限值的像元不进入新缓存。坡度限定 0–90、坡向限定 0–360（不含 360）；平地坡向 sin/cos 置零。

原始 `unit_bags.npz` **不重写**。新的 `.npy` 缓存保留过滤后的完整像元，训练时再抽样。小袋使用实际像元并掩码填充；空袋的像元贡献严格为零，由单元表格分支预测。

## 实验臂及贡献边界

| 名称 | 输入/结构 | 用途 |
|---|---|---|
| table_xgb | 原 38 维 + 有效像元数/可用性 | 本实验统一协议下的表格基线 |
| stats_xgb | 同上 + 全部有效像元的面积加权均值、标准差、分位数及水位—坡度联合统计 | 丰富统计量强对照 |
| mil | 同一单元表格 + 地形局部编码 + 面积先验注意力 | 普通 MIL 基线 |
| concat_mil | 同一表格 + 地形/水位条件直接拼接 + 普通注意力 | 同信息集对照，区分水位信息与 FiLM 结构的作用 |
| water_mil | 同一单元表格 + 水位条件 FiLM 局部调制 + 同样注意力 | 局部交互候选模型 |

五臂均使用相同人群、固定折、训练折内 soft 权重，或显式 `--neg-sampling none`。不加额外类别正样本权重，不使用旧 OOF 概率。water_mil 仅胜过 mil 而没有胜过 concat_mil，不足以支持条件结构的贡献。

水位条件为历史站点水位与像元高程构成的被淹比例及相对 p10/p50/p90 水位高程；它们通过无标签的固定公式计算。该版本不是时序记忆模型，没有空间邻接模块，也不能宣称首次提出 MIL/FiLM。水位条件只是待检验归纳偏置，不能称真实孔压。

**仅适用于回顾性空间易发性研究。** 水位摘要覆盖源表历史期间，原 38 维还包含事件/伪事件窗口变量；不能把结果写作固定预测时点的未来风险或提前预警。

不能直接把本次 table_xgb 分数与历史报告数字作逐点比较：本次增加相同可用性变量、显式按 ID 对齐投影几何质心、使用严格面积预算前缀，XGB 使用 hist，全部需在新折表上配对比较。

## 服务器操作（从已经构建好像元袋的状态开始）

本例项目目录为 `/3250707052/landslide/mil_prepare`。上传 `dist/pixel_bag_v1_patch.zip` 到该目录。包只包含本实验代码、说明和依赖代码指纹，解压只增加 `experiments/pixel_bag_v1/`；复用的主线代码已经在之前的服务器包中。

```bash
source /3250707052/.venv-mil/bin/activate
cd /3250707052/landslide/mil_prepare
unzip pixel_bag_v1_patch.zip
python experiments/pixel_bag_v1/verify_sources.py
```

`verify_sources.py` 会核对服务器现有 `src/config.py`、`src/dataset.py`、`tills/shp_io.py` 与打包时指纹是否一致。若不同，先停止，不覆盖主线文件。它也校验新实验代码，检查上传完整性。

### 1. 测试（无需真实像元袋）

```bash
python experiments/pixel_bag_v1/test_pipeline.py
```

这是合成数据的训练/预处理检查，不是论文性能验证。

### 2. 准备新缓存（一次）

```bash
mkdir -p logs
python -u experiments/pixel_bag_v1/run.py prepare \
  --bags features/v2/unit_bags.npz \
  --out results/pixel_bag_v1/prepared \
  > logs/pixel_bag_v1_prepare.log 2>&1
tail -n 40 logs/pixel_bag_v1_prepare.log
```

成功标志是出现 `results/pixel_bag_v1/prepared/manifest.json`，这是准备完成标志，最后写入。检查其中人口 25,509、正样本 661、空袋正样本 0。像元数量是**训练人口**过滤后的数量，预期不会等于原始全量袋的 8,366,561；这是有意排除非训练人口与无效像元。

输出：`pixels.npy`、`pixel_area.npy`（内存映射读取）；`units.npz`（表格、统计量、坐标、面积、袋索引）；`unit_audit.csv`（每单元原始/有效像元数）；`manifest.json`（源文件指纹、波段、历史水位时间范围等）。

像元面积通过 EPSG:4326 每行像元的椭球面积计算，不能把每个像元均设为 900 m²。严格网格对齐抽查在 prepare 中进行。

### 3. 技术冒烟：一个完整空间留出折，2 轮训练

```bash
python -u experiments/pixel_bag_v1/run.py train \
  --prepared results/pixel_bag_v1/prepared \
  --out results/pixel_bag_v1/smoke \
  --only-fold 0 --seeds 42 --epochs 2 --trees 30 \
  --k 32 --hidden 32 --batch 128 --eval-reps 2 --threads 4 --device cuda \
  > logs/pixel_bag_v1_smoke.log 2>&1
tail -n 30 logs/pixel_bag_v1_smoke.log
```

这里五臂均使用同一留出折，但树数/轮数为冒烟设置，不能作为最终模型优劣结论。确认没有非有限损失、预测缺失或显存报错；`summary.json` 的 `complete_oof=false` 是单折实验的正常标记。先回传本轮小结果包再决定正式配置，避免选完参数后仍把同一筛选折视作未见过的数据。

### 4. 导出小结果包（首轮应做）

```bash
python experiments/pixel_bag_v1/run.py export \
  --prepared results/pixel_bag_v1/prepared \
  --run results/pixel_bag_v1/smoke \
  --out pixel_bag_v1_smoke_return.zip
```

下载 `pixel_bag_v1_smoke_return.zip` 后发到聊天即可。本地可据此导入真实指纹、逐单元预测、人口审计、折和指标。默认不含大缓存、原始袋或模型权重，权重仍留服务器。

### 5. 完整验证（配置确定后）

```bash
python -u experiments/pixel_bag_v1/run.py train \
  --prepared results/pixel_bag_v1/prepared \
  --out results/pixel_bag_v1/spatial_full \
  --method spatial_kmeans --seeds 42,43,44 --epochs 12 \
  --k 64 --hidden 32 --batch 128 --threads 4 --device cuda \
  > logs/pixel_bag_v1_spatial_full.log 2>&1
```

行政验证将 `--method` 改为 `admin`，`--out` 与日志名称也换新。12 轮只是固定起始配置，不代表已优化。若调参需训练部分内层划分；程序不利用外层验证指标早停或选轮数。

固定 fold-seed=42，不随模型 seed 变化；每个模型 seed 重新训练。验证像元抽样由单元 ID 和固定 sampling-seed 决定，各模型相同；默认重复 3 次取平均。大袋按像元面积有放回抽样，小袋完整使用并按面积加权；这不是把重复像元当成额外观测。

最终指标：全域 AUC、PR-AUC、Brier、严格前 10/20% 面积覆盖；另报消落带**内部排序/内部面积预算**下同类指标。全域和带内指标不可混称。soft 加权输出是分数，Brier 只是诊断，不代表已完成绝对概率校准。

本入口暂不自动给显著性或风险保证。论文前还需完成训练内调参、空间块配对置信区间、水位交互消融、袋容量/采样稳健性。此包是可运行的首轮实验，不是已证明有效的新模型。

## 服务器数据怎样保存/回传

1. 原始 `features/v2/unit_bags.npz`、`unit_id_grid.tif` 和原报告继续留服务器；训练不覆盖它们。
2. 建议现在就下载一次原始袋（约 55.8 MB）及编号栅格，保留服务器释放前的独立备份。没有原始地形栅格时，这些文件也能保存既有提取结果。
3. 小结果回传使用 export 命令；若希望同一返回包包含原始袋，加 `--raw-bags features/v2/unit_bags.npz` 并换一个新输出文件名。
4. 缓存 `.npy` 可从原始袋重建，首轮不用下载；正式模型权重 `.pt`、XGB 模型 JSON 和 scaler 文件在服务器运行目录中，结束租用前也应单独备份。

原始数据备份命令：

```bash
tar -czf pixel_bag_raw_backup.tgz \
  features/v2/unit_bags.npz \
  features/v2/unit_id_grid.tif \
  results/unit_grid_report.txt \
  results/unit_bags_report.txt
sha256sum pixel_bag_raw_backup.tgz
```

日志可额外下载 `logs/pixel_bag_v1_*.log`。不要只留下聊天里的统计数字；这些数字无法重建像元袋。

## 本地已记录/尚未取得的内容

`server_observations.json` 记录用户提供的服务器环境与检查结果，`evidence/` 保存收到的文本附件。来源标记为 user_reported，不能当作本地复跑或自动读取服务器的证据。

尚未取得服务器真实 `unit_bags.npz`，也未训练真实模型。没有服务器连接授权与可用连接，本地不会自动同步。拿到返回包后再核对指纹和导入结果。
