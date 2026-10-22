# 三峡库区斜坡单元易发性研究 · V3

当前主线是在有限训练样本、区域环境存在差异的条件下，利用坡内信息改善易发性识别，以空间与行政五折评价研究区内部地域迁移。

V3 使用修正站点与事件前时间口径，输入为38维表格、2维像元可用性和44维像元统计，共84维。建模25,509个单元、661个正样本；制图25,939个单元，430个河道单元按约定赋最低等级。

代码通过Git同步，服务器直接拉取；本地Codex可经SSH连接远程项目。统一运行入口为`bash scripts/run_v3.sh`，操作方法见[Git与SSH流程](docs/V3_GIT_SSH_WORKFLOW.md)。

## 工作入口

- [V3文档索引](docs/README.md)
- [主线、版本边界与当前状态](docs/V3_MAINLINE.md)
- [数据准备与84维训练](docs/V3_EXPERIMENT_RUNBOOK.md)
- [40/48/84维对照与OOF口径](docs/V3_CONTROLS_AND_MAPS.md)
- [唯一保留的正式图集](docs/V3_BASEMAP_ATLAS.md)
- [84维解释方法](docs/V3_ALL84_INTERPRETATION.md)
- [当前解释模型运行入口](docs/V3_EXPLAIN84_CANONICAL_MODEL.md)
- [V2历史归档](archive/v2/README.md)

源码在`src/v3/`，配置在`config/v3/`，固定分折在`splits/v3/`，特征在`features/v3/`。权威水位源为`data/water/干流站点水位-0908.xlsx`，旧`水位.xlsx`停用。

正式出图仅使用`results/v3_county_borders_atlas_20261021/`。其他迭代图集与本地出图分析目录已经清理；实验评分、上下文、信息量表和重载诊断证据保留。当前解释入口为`src.v3.explain84_canonical`，本地文档不代表服务器已运行完成。

V2文档集中在`archive/v2/`；旧`src`、`tills`、`experiments`保留兼容与复现用途。仅放行训练必需的冻结特征、上下文和参考数据；其余原始数据、模型、结果和压缩包不入Git；本轮未提交Git或同步服务器。
