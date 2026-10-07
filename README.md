# 三峡库区消落带斜坡单元滑坡易发性（v2 主线）

> **一页速览**：25,939 个斜坡单元（训练 25,509，剔除 430 个河道内单元）→ **38 维特征**（34 维基础 + 4 维地质精选）
> → XGBoost 5 折空间/行政交叉验证（**OOF AUC 0.904**）→ 易发性图 + 因子归因 + 机理证据链。
> 代码、文档、交付表入库；`data/` 与全部生成产物不入库（见 `.gitignore`）。

---

## 现在处于什么状态（2026-10）

| | 内容 | 权威文件 |
|---|---|---|
| **模型** | 38 维主线；XGBoost（300 树、depth 5、lr 0.05） | `src/config.py`（`MAIN_FEATURES_38`）、`features/v2/features_v2_train_38.csv` |
| **出图口径 soft02** | 正式易发性图，5 级等频分级：极高易发占 23.5% 面积、含 86.8% 的滑坡 | `predictions/susc_v2_soft02.*` |
| **归因口径 matched5** | 病例-对照匹配 + 先验校正（混杂 AUC 0.8357→0.557） | `predictions/susc_v2_matched5.*`（仅 `support=1`） |
| **三层分解** | 位置/水位 15 维已达 38 维判别力的 99.4% | `predictions/contrib_decomp.shp` |
| **地质图层** | 岩性 / 硬度 / 坡面结构 / 土壤 / 年代 → 图层交付，**不进模型** | `layers/` |
| **地质因子结论** | 判别力层"检不出"（低于检出限），机理层"显著"（顺向坡 OR=1.33、T2 1.60、J2 0.73、T1 0.45） | `docs/地质机理与消融_v2.md` |
| **已弃用** | 黏聚力 c / 剪切力 fan（1 个自由度 + 可由地形复现 90% + 单位不明） | `deliverables/列状态_保留与弃用.csv` |

---

## 从哪开始

1. **想了解项目** → `docs/项目结构.md`（目录地图 + 流水线命令）→ `docs/README.md`（文档索引）
2. **想复现结果** → `docs/出图与归因_SOP.md` 的三条流水线（出图约 10 分钟 / 机理约 5 分钟 / 消融约 2 小时）
3. **想改特征** → 读 `docs/FEATURE_SELECTION_V2.md` 与 `docs/地质机理与消融_v2.md` 的口径与判据，
   改完必须重跑第 10 轮消融（`tills/v2_ablation_r10_geology.py`）才能下结论
4. **要答辩/评审** → `docs/出图与归因_SOP.md` §四「证据对应表」逐条对应到文件

---

## 环境（重要）

```powershell
$py = "$env:USERPROFILE\.conda\envs\landslide\python.exe"   # 用 landslide 环境
& $py tills\v2_predict_map.py --method spatial_kmeans --seeds 42,43,44
```

- 必须用 **`landslide`** 环境（含 shapely / pyproj / rasterio / pyshp / xgboost）；
  base 环境缺 shapely，会直接报 `ModuleNotFoundError: geopandas`。
- 本项目**不使用 geopandas**（GDAL DLL 冲突）：统一走 `tills/shp_io.py` 自带的 shapefile 读写；
  `baseline_xgb.py` 内部的几何读取由调用脚本用 `runpy` + 猴补丁替换。
- `rasterio` 必须在 `pyproj` 之前 import；沙箱内不要用 `n_jobs>1` 的 joblib 并行。

---

## 目录一览

```
data/          原始数据（只读，21 GB，不入库）
features/      特征层：features_v2_train_38.csv（38 维主线）、v2/groups/*、v2/ablation/（中间件，可删）
src/           核心库：config / dataset（含 matched_negatives）/ metrics（model、train 为 v1 GNN 遗留）
tills/         数据准备与实验脚本（46 个，索引见 docs/脚本索引.md）
tools/         一次性校验与制表工具
results/       结论层：顶层权威报告 + runs/（逐次运行 JSON）+ archive/（被取代的旧报告）
predictions/   出图产物：双口径 shp/tif + 三层分解 + OOF 数值 + 预览图
layers/        地质图层：单元级综合 shp + 5 张同网格栅格 + 图例
deliverables/  对外交付表：42/48 维表、中文表头、数据字典、列状态表
docs/          文档（12 份，索引见 docs/README.md）
archive/       v1 全量归档 + 2026-10 过程文件清理记录
```

---

## 已知限制（写论文/答辩必须声明）

1. 训练人口剔除的 430 个河道内单元中含 **1 个带标签单元**（`unit_id=16490`）：全库正样本 662、训练 661。
2. 地质无覆盖 **89 个单元**按训练人口列均值填补；坡面结构无数据 628 个单元（图层中记为空值，不填补）。
3. 滑坡编目存在**空间自相关**：所有"按单元独立"的 p 值偏乐观，因此机理层引入 3 km 空间块 bootstrap。
4. 地质 4 维的判别力增量（+0.0005~+0.0015）**低于各口径的检出限**（soft02 0.0019~0.0032，matched5 0.0065~0.0081）：
   结论应表述为"未检出"而非"无效"；要把检出限压到 0.001 约需 2,500 个正样本（现有 661）。
