# 三峡库区斜坡单元易发性研究 · V3

当前屏障区主线为84维TabPFN、8成员、1:3子采样、seed42空间五折。统一入口为`bash scripts/run_v3.sh mainline`；已完成结果无需重跑。原2成员配置和行政五折结果保留为历史对照，不再新增行政实验。

V3 使用修正站点与事件前时间口径，输入为38维表格、2维像元可用性和44维像元统计，共84维。当前建模和制图均为25,509个单元、661个正样本；430个河道单元不纳入当前SHP及统计。

代码通过Git同步，服务器直接拉取；本地Codex可经SSH连接远程项目。统一运行入口为`bash scripts/run_v3.sh`，操作方法见[Git与SSH流程](docs/V3_GIT_SSH_WORKFLOW.md)。

## 工作入口

- [V3文档索引](docs/README.md)
- [主线、版本边界与当前状态](docs/V3_MAINLINE.md)
- [数据准备与84维训练](docs/V3_EXPERIMENT_RUNBOOK.md)
- [40/48/84维对照与OOF口径](docs/V3_CONTROLS_AND_MAPS.md)
- [历史底图图集](docs/V3_BASEMAP_ATLAS.md)
- [主线升级后的消融与报告证据](docs/V3_REPORT_EVIDENCE.md)
- [84维解释方法](docs/V3_ALL84_INTERPRETATION.md)
- [当前解释模型运行入口](docs/V3_EXPLAIN84_CANONICAL_MODEL.md)
- [V2历史归档](archive/v2/README.md)

源码在`src/v3/`，配置在`config/v3/`，固定分折在`splits/v3/`，特征在`features/v3/`。权威水位源为`data/water/干流站点水位-0908.xlsx`，旧`水位.xlsx`停用。

当前1:3分区位于`results/v3_sampling_jenks5_20261030/`，旧`results/v3_county_borders_atlas_20261021/`保留历史身份。旧canonical解释、上下文和信息量证据保留；新1:3主线的匹配消融与模型特定因子解释尚待补齐。

V2文档集中在`archive/v2/`；旧`src`、`tills`、`experiments`保留兼容与复现用途。仅放行训练必需的冻结特征、上下文和参考数据；其余原始数据、模型、结果和压缩包不入Git。
