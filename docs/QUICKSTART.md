# 快速使用指南（斜坡单元方案 · v2 口径）

> **口径迁移提示（本文按 v2 数据规范刷新，2026-09-20 起生效）**
> 唯一权威口径见 [DATA_SPEC_V2.md](DATA_SPEC_V2.md)。本文所有路径 / 命令 / 数字均以 v2 为准。
>
> | 项目 | v2 取值 |
> |---|---|
> | 出图人口 | **25,939** —— `data/slope_units/slope_units_final.shp` |
> | 训练人口 | **25,509** —— `data/slope_units/slope_units_final_train.shp`（再剔除 430 个河道内单元） |
> | 正样本 | 全量 662 / 训练 **661** |
> | 唯一 ID | **`unit_id = 1..25939`**，出图与训练共用同一套取值；权威对照表 `features/v2/unit_id_map_v2.csv` |
> | 主线特征表 | **`features/v2/features_v2_train.csv`**（25,509 × 34 维）；全量 `features/v2/features_v2.csv`（25,939） |
> | XGB 基线 | 全单元 AUC **0.9093 ± 0.0008**｜采样池 0.8740｜recall@10% 0.6192（v1 对照 0.8134±0.0039 / 0.7699 / 0.4394） |
>
> **两条禁令**
> 1. **禁止跨表按 ID 数值直连**：新 `unit_id` 与旧 `src_fixed_Id` 数值相同的仅 **136/25,939**，按 ID 直连会让 25,803 行落到错误几何（这正是 v1 出问题的机制）。
> 2. **禁止混用 v1 表**：v1 的旧 shp / 特征表 / 脚本 / 结果全部在 `archive/v1/`，**v1 全部 headline 与消融已作废**（数据规范 §7），且 v1 与 v2 不可做逐单元对比。

流程总览：

```
环境准备 → v2 数据准备（11 条命令） → XGB 基线 → 数据健康检查 → （GNN / TabPFN：待重跑）
```

***

## 0. 环境准备

```powershell
# 推荐 conda 环境（本机现成环境：C:\Users\dollars\.conda\envs\landslide\python.exe，Python 3.10）
conda create -n landslide python=3.10 -y
conda activate landslide
pip install -r requirements.txt
```

> - 本机 GPU 是 RTX 5060 Laptop（8 GB，Blackwell sm_120），当前装的是 **CPU 版 torch**（`torch 2.12.1+cpu`，`torch.version.cuda = None`）：**现在只能 CPU 跑**，要上 GPU 见 [SERVER_TRAIN.md](SERVER_TRAIN.md) §1。
> - geopandas 的 GDAL 矢量栈（pyogrio）在本机间歇失败 → 用项目自带 `tills/shp_io.py`（纯 Python，含 `read_polygons` / `read_lines` / `read_dbf`）；v2 脚本已内置该替代。
> - Windows 无需安装 torch-geometric（模型自带 SAGEConv 实现）；方案 C 可选 `pip install performer-pytorch`。

***

## 1. 数据准备：v2 复现顺序

前置数据（**跨版本共用源**，v2 未重跑 GEE）：`features/daily_water_levels.csv`（逐日水位，3 站）、`features/station_assign.csv`、`features/ndvi_unit_matrix.csv`、`features/rain_unit_matrix.csv`。

下列清单照抄数据规范 §9.2（共 11 条命令，步骤编号同 §9.2：步骤 2 含 5 个提取器、步骤 4 含基线）。命令均在项目根目录执行：

| 步骤 | 完整命令 | 主要产出 |
|---|---|---|
| 步骤 1 人口与新 ID | `python tills\build_final_population.py` | `data/slope_units/slope_units_final.shp`（25,939）、`slope_units_final_train.shp`（25,509）、计数表、`features/v2/unit_id_map_v2.csv` |
| 步骤 2 几何+地形 | `python tills\v2_extract_geometry_terrain.py` | `features/v2/groups/geometry.csv`、`terrain.csv`、`elevation_quantiles.csv` |
| 步骤 2 高程像元对 | `python tills\v2_dump_elev_pairs.py` | `features/v2/groups/elev_pairs.npz`（面积加权淹没用） |
| 步骤 2 CLCD 矩阵 | `python tills\v2_extract_landuse.py` | `features/v2/groups/landuse.csv` |
| 步骤 2 水系+道路 | `python tills\v2_extract_network_features.py` | `features/v2/groups/water_network.csv`、`road.csv` |
| 步骤 2 水位/淹没 | `python tills\v2_build_inundation_variants.py` | `features/v2/groups/water_variants.csv`（7 口径）+ `water.csv`/`wetdry.csv`（**面积加权，主线用**；-0908 四站就近匹配） |
| 步骤 3 GEE 10 列 | `python tills\v2_build_gee_features.py` | `features/v2/groups/gee.csv`（按新 ID 重映射） |
| 步骤 4 组装主线 | `python tills\v2_assemble_features.py` | `features/v2/features_v2.csv`（25,939）、`features/v2/features_v2_train.csv`（25,509，34 维） |
| 步骤 4 基线 | `python tills\v2_run_baseline.py --tag v2final` | `results/v2_baseline_v2aw.txt`（见 §3） |
| 步骤 4b 淹没消融 | `python tills\v2_ablation_inundation.py` | `results/v2_ablation_inundation.txt` / `.json` |
| 单变量扫描 | `python tills\v2_univariate_scan.py` | `results/v2_univariate_auc.txt`（见 §4） |

要点：

- **顺序不可颠倒**：`v2_dump_elev_pairs.py` 必须在 `v2_build_inundation_variants.py` 之前（后者读 `groups/elev_pairs.npz` 与 `groups/elevation_quantiles.csv`）；`v2_assemble_features.py` 必须在全部 `groups/*.csv` 齐备后运行。
- **ID 只认 `unit_id`**：v2 shp 只保留这一列 ID；读单元主键一律用 `features/v2/unit_id_map_v2.csv`，不要用行索引（v1 的 bug 根因，见数据规范 §6）。
- **淹没口径**：v2 = 面积加权（`inundation_fraction = Σ_像元 天数(水位 ≥ 该像元高程) / (像元数 × 总天数)`）；单点参考高程（均值/p10/p20/p40/p50/p80）六种版本结果逐位相同且被模型完全忽略（淹没列重要性 0.00%），故不采用。
- 缺失处理：468 个无 DEM 覆盖的微小/边缘单元按列均值填充（占 1.8%）。

***

## 2. `main.py` 现状：**当前不可用**

`main.py` 仍编排 **v1 旧脚本**，而这些脚本已经归档到 `archive/v1/tills/`，所以现在直接运行会在第二步就失败：

```powershell
python main.py --stage data       # ✗ 报 FileNotFoundError（join_landslide_dates.py 等已归档）
python main.py --stage graph      # ✗ tills/build_graph.py 已归档
python main.py --stage baseline   # ✗ 仍指向 features/event_window_features_k2_v30.csv（v1 表，已移入 archive/v1/）
```

**`main.py` 需要改写成 v2 流程**（数据规范 §10 待办第 1 条）。建议的 v2 编排顺序就是 §1 的清单，即：

```
build_final_population.py
  → v2_extract_geometry_terrain.py → v2_dump_elev_pairs.py → v2_extract_landuse.py
  → v2_extract_network_features.py → v2_build_inundation_variants.py
  → v2_build_gee_features.py → v2_assemble_features.py
  → v2_run_baseline.py → v2_ablation_inundation.py → v2_univariate_scan.py
```

改写前，请按 §1 逐条手工执行（每条命令都是幂等的重算脚本）。

***

## 3. 跑基线（v2 主线）

```powershell
python tills\v2_run_baseline.py --tag v2final
```

- 口径：**admin × soft**（按县整县分折，21 县；软负采样 4 km 内权重 1.0、远区 λ=0.2），**5 折 × 3 seeds**（42/43/44），特征表 `features/v2/features_v2_train.csv`（25,509 × 34 维，正样本 661）。
- 输出：**`results/v2_baseline_v2aw.txt`**（同时逐 seed 打印到控制台）。
- **期望结果**：全单元 AUC **0.9093 ± 0.0008**、采样池 AUC **0.8740**、recall@10% **0.6192**。
- 单条 XGB 复现（单 seed，便于调试）：

```powershell
python baseline_xgb.py --features-csv features/v2/features_v2_train.csv `
    --method admin --neg-sampling soft --neg-km 4 --neg-lam 0.2 --folds 5 --seed 42
```

> 注意 `features/v2/features_v2_train.csv` 的第 1 列必须是 `unit_id`（不是行索引）；`baseline_xgb.py` 的 `--features-csv` **默认仍是 `features/features.csv`**（v1 静态对照口径），必须显式传 v2 表——`tills\v2_run_baseline.py` 已替你传好。GNN / TabPFN 的旧 headline（GNN A/B/C 0.7961/0.7365/0.7553、TabPFN 0.8286）均已作废，需在 v2 上重跑。

***

## 4. 数据健康检查（单变量判别力）

```powershell
python tills\v2_univariate_scan.py
# → results/v2_univariate_auc.txt
```

用途：对照**新旧两套表**的单变量 AUC，检查"关键因子是否回到正确归属"、排查泄漏。已知关键对照（新 / 旧）：

| 特征 | 新表 AUC | 旧表 AUC |
|---|---|---|
| `elevation_mean` | 0.1642（反向等效 0.836） | 0.4156（近噪声） |
| `inundation_fraction` | 0.8550 | 0.5220 |
| `wet_dry_cycles` | 0.8555 | 0.5143 |
| `ant_inund_days_3m` | 0.7431 | 0.5101 |
| `mainstream_dist_m`（本来就对齐） | 0.2679 | 0.2675 |
| `drainage_density`（本来就对齐） | 0.7004 | 0.7005 |

判读：v1 的 10 列（地形 5 + 淹没 1 + 干湿 4）因 `get_unit_id()` 候选列名漏 `Id` 而贴错单元；去掉全部 5 个水位/淹没/干湿列后 AUC 仍 0.8999，说明 AUC 跃升不是泄漏，而是"关键因子恢复正确归属"。

***

## 5. 新增因子（岩性 / 产状 / 力学参数 / 土壤）

已按 v2 口径重算完成（产物在 `features/v2/groups/`，结论见 [NEW_FACTORS_V2.md](NEW_FACTORS_V2.md)；若后续还要新增其它数据，一律**按 v2 口径重算**：

- 以 `data/slope_units/slope_units_final.shp` 的 **`unit_id` 为唯一主键**，输出到 **`features/v2/groups/<组名>.csv`**；
- 经 `python tills\v2_assemble_features.py` 并入主线表；
- **禁止**沿用 v1 人口下算出的旧表（已归档）——旧口径的人口、ID、负样本伪事件抽样都不同，直接拼接会重现错位问题。

***

## 6. GNN / TabPFN 现状（v2 上均**待重跑**）

| 项 | 现状 |
|---|---|
| GNN | 图尚未按 v2 重建（`src/config.py` 里 `GRAPH_NPZ = features/v2/graph_v2_train.npz` 待生成，`build_graph.py` 已归档到 `archive/v1/tills/`）；训练入口 `train_gnn.py` 与出图 `predict_gnn.py` 需按 v2 接线（`predict_gnn.py` 仍指向 v1 的 `slope_units_fixed.shp` / `submerged_units_combined.csv` / `train_backfill_map.csv`） |
| TabPFN | 未在 v2 上运行过（`baseline_tabpfn.py` 读的已是 `features/v2/features_v2_train.csv`，但尚未跑） |
| 命令与环境 | 见 [SERVER_TRAIN.md](SERVER_TRAIN.md)：v2 训练/评估命令清单、本机 torch/TabPFN/GDAL 踩坑、云 GPU 租用与打包 |

***

## 7. 易发性等级

| 等级 | 名称 | 颜色建议 | 说明 |
|------|------|----------|------|
| 0 | 极低易发性 | 深绿 | 概率 < 0.2 |
| 1 | 低易发性 | 浅绿 | 0.2-0.4 |
| 2 | 中易发性 | 黄 | 0.4-0.6 |
| 3 | 高易发性 | 橙 | 0.6-0.8 |
| 4 | 极高易发性 | 红 | ≥ 0.8 |

`--method quantile` 时按分位数每级 20% 单元，等级面积均衡；`--method fixed` 按上表固定阈值。
出图时 **430 个河道内（常年水下）单元回填 prob=0**，保证 25,939 个单元无空缺。

***

## 8. 常见问题

| 问题 | 处理 |
|------|------|
| `main.py` 报 FileNotFoundError | 属预期：`main.py` 仍编排 v1 旧脚本（已归档），见 §2，改用 §1 的 v2 清单 |
| geopandas 读矢量间歇失败（GDAL/pyogrio DLL 冲突） | 用 `tills/shp_io.py`（`read_polygons` / `read_lines` / `read_dbf`）；勿反复重试 geopandas |
| 特征表缺列 / 行数不对 | 检查 `features/v2/groups/` 下 10 个中间产物是否齐备（geometry / terrain / elevation_quantiles / elev_pairs.npz / water / wetdry / water_network / road / landuse / gee），再跑 `python tills\v2_assemble_features.py` |
| 表格行数应为 25,939 却不然 | 确认用的是 `slope_units_final.shp`（出图）或 `slope_units_final_train.shp`（训练），且主键取 `unit_id`，不要用 0 基行索引 |
| 基线 AUC 异常高（接近 1.0） | 存在特征泄漏：检查是否混入复发特征/截断类时序特征（防泄漏口径见数据规范 §3） |
| 训练 loss 为 NaN | 检查特征表是否有 Inf；组装脚本已用列均值填充 NaN |
| torch 是 CPU 版 / `torch.cuda.is_available()` 为 False | 本机现状即如此（`2.12.1+cpu`）；要跑 GPU 需装 CUDA 12.8 构建的 torch ≥ 2.7，见 [SERVER_TRAIN.md](SERVER_TRAIN.md) §1 |
| 控制台中文乱码 | 报告一律写 UTF-8 文件（`results/*.txt`）再读，不要依赖 stdout |
| 想换事件窗口 K | 时序特征的 K 在 v2 组装/GEE 环节确定；改动需重跑对应 v2 脚本并重做基线对比，不要沿用 v1 的中间表 |
