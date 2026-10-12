# 文档索引

> 项目文档：**现行 13 份**（`docs/`）+ 历史归档（`archive/docs_history/`）。
> 带 ⭐ 的是答辩/评审最常引用的文档；**当前操作性方案只有一份：`N1_N2_N3_方案.md`**。

## 现行文档（按推荐阅读顺序）

| # | 文档 | 一句话 | 状态 |
|---|---|---|---|
| 1 | ⭐ **`项目结构.md`** | 项目总地图：目录、产物、流水线命令、口径规则 | 现行 |
| 2 | ⭐ **`N1_N2_N3_方案.md`** | **当前唯一操作性方案**：三条主张（校正充分性/口径盲报告/空间支撑）+ 任务流程 + 止损规则 | 现行 |
| 3 | ⭐ **`主张台账.md`** | 主张登记：被推翻条件、结构性空缺理由、证据、已作废方向 | 现行 |
| 4 | `论文骨架与图表.md` | 论文骨架、图表清单、评审备答、降级清单 | 现行 |
| 5 | ⭐ **`出图与归因_SOP.md`** | 双口径叙事 + 三条流水线 + 产物校验点 + 答辩证据对应表 | 现行 |
| 6 | ⭐ **`地质机理与消融_v2.md`** | 第 10 轮消融（732 次运行）与 M1–M9 机理证据链 | 现行 |
| 7 | ⭐ **`FEATURE_SELECTION_V2.md`** | 43 → 4 维的筛选过程；§8 解释"物理真实但重要性低" | 现行 |
| 8 | ⭐ **`当前模型方案.md`** | **模型线**：M1 库水位算子 / M2 MIL 单元袋 / M3 TabPFN；统一判定标准、已证伪路线、排期 | 现行（待执行） |
| 9 | `汇报_模型线阶段结果.md` | 汇报用一页结论 + 主结果 + 备答（汇报文档已收敛为这一份） | 现行 |
| 10 | `本地GPU.md` | 本机 RTX 5060 跑 TabPFN 的安装、显存口径与命令 | 现行 |
| 11 | `DATA_SPEC_V2.md` | 数据规格：源数据、字段、CRS、口径 | 现行 |
| 12 | `FEATURES_V2.md` | 34 维特征逐项定义与提取口径 | 现行 |
| 13 | `NEW_FACTORS_V2.md` | 四类地质因子的早期消融（§3.3 的 9 列推荐集**已作废**） | 历史+标注 |
| 14 | `负采样/依据清单.md` | 负采样/病例-对照的方法学依据（L1–L7） | 现行（部分待补） |
| 15 | `PROJECT_OVERVIEW.md` | v1 期项目总览；**`src/*.py` 的模块说明与 `FEATURES_V2.md` 仍引用它** | 历史+标注 |
| 16 | `脚本索引.md` | 全部脚本的一句话说明（自动生成） | 自动 |

## 已归档（`archive/docs_history/`）

| 文件 | 归档原因 |
|---|---|
| `文献查证档案_五轮.md` | 五轮文献查证的完整依据（邻近工作、判决表、已作废方向、找 niche 的方法）——**写作时引用，不必反复阅读** |
| `EXPERIMENT_RESULTS.md` · `MODEL_UPGRADE_PLAN.md` · `QUICKSTART.md` · `SERVER_TRAIN.md` | v1/早期文档，已被现行文档取代（v1 结果整体作废；服务器流程见上传包 `SETUP.md`） |
| `REPORT_V2_MIGRATION.md` · `GEE_UNIT_STATS_MONTH.md` | 迁移记录 / GEE 数据字典，属历史留档 |
| `汇报_模型现状总览.md` · `汇报_模型现状总览_详细版.md` | 汇报文档重复，已收敛为 `docs/汇报_模型线阶段结果.md` |

`archive/tools_history/`：一次性或已过时的工具（`patch_geopandas_fallback.py`、`m3_local_xgb_seeds.py`、`inspect_new_lithology.py`、`litho_digitization_priority.py`、`make_chinese_header_48.py`、`make_delivery_48d_dict.py`、`check_clip_extent.py`）。

## 相关的非 docs 文档

| 文件 | 内容 |
|---|---|
| `deliverables/README.md` | 交付表说明（§0 是 2026-10 的列状态更新，**先读 §0**） |
| `deliverables/列状态_保留与弃用.csv` | 48 列逐列状态与依据 |
| `layers/README.md` | 地质图层清单、字段与图例编码 |
| `results/README.md` | `results/` 目录导航：哪些是权威结论、哪些是逐次运行 |
| `archive/v2_process/README.md` | 2026-10 过程文件清理记录 + 各轮臂表的**重建方法** |
| `tools/inventory_cleanup.py` | 清理时的引用热度清点工具（可复用） |

## 文档维护约定

1. **结论必须写进 docs，不散落在聊天记录里**：新实验 → 新的 `results/*.txt` + 在对应 docs 增补一节。
2. **主张先登记再实验**：新想法先进 `主张台账.md`，判据（被推翻条件）先写定，再跑实验。
3. **改特征/口径必须同步四处**：`地质机理与消融_v2.md`、`出图与归因_SOP.md`、
   `deliverables/列状态_保留与弃用.csv`、`FEATURE_SELECTION_V2.md`。
4. **过时结论不要直接删**：归档到 `archive/docs_history/` 或加"⚠ 后续状态"标注，保留推理链。
5. **不确定性分级统一用**：✅ 已测量 / 📚 文献支撑 / ⚠️ 推断或未证实。
