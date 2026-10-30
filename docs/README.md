# V3 文档索引

现行文档只保留V3研究与运行所需内容。V2历史资料见归档；已被canonical替代的refit、saved、seeded和reference-check操作说明已删除，相关源码因兼容和诊断依赖保留。

|环节|文档|
|---|---|
|Git同步、服务器部署、SSH与统一运行入口|[Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)|
|研究任务、版本边界、当前状态|[V3主线](V3_MAINLINE.md)|
|主线升级后的消融、因子分析与报告写法|[报告证据衔接](V3_REPORT_EVIDENCE.md)|
|保留坡内统计、训练侧保守精简对照|[第一轮精简实验](V3_CONSERVATIVE_SELECTION.md)|
|QGIS五级分区、县域覆盖率与版式导出|[QGIS出图与评价](V3_QGIS_GRADING_AND_EVALUATION.md)|
|站点纠正与匹配|[站点匹配](V3_STATION_MATCHING.md)|
|事件前水位与折内参照日期|[水位背景](V3_WATER_BACKGROUND.md)|
|多时间尺度84维输入与五折训练|[实验运行指南](V3_EXPERIMENT_RUNBOOK.md)|
|40/48/84维控制、OOF与分级|[对照与地图口径](V3_CONTROLS_AND_MAPS.md)|
|正式图集与清理后的重建顺序|[底图图集](V3_BASEMAP_ATLAS.md)|
|组级解释与空间诊断|[组级解释](V3_EXPLAIN_S42.md)|
|84维置换、ALE与解释边界|[逐维解释方法](V3_ALL84_INTERPRETATION.md)|
|当前解释入口、模型恢复及严格核验|[Canonical加载模型](V3_EXPLAIN84_CANONICAL_MODEL.md)|
|训练/测试信息量与相关性|[IV与相关性](V3_IV_CORRELATION.md)|
|历史复现与恢复|[V2归档](../archive/v2/README.md)|
|服务器镜像选择、依赖重建与校验|[服务器环境重建](V3_SERVER_REBUILD.md)|
|概率平衡、集成子采样与单折试跑|[平衡实验](V3_BALANCE_EXPERIMENT.md)|
|AP/AUC、分级覆盖与面积、BA统一口径|[模型评价标准](V3_EVALUATION_STANDARD.md)|
|目录职责与本地辅助资料归档|[项目结构与归档](V3_PROJECT_STRUCTURE.md)|

正式图集唯一目录为`results/v3_county_borders_atlas_20261021/`。历史返回数值与诊断证据不等于可用出图；旧补丁和ZIP不作为主线入口。服务器完成状态以日志和结果核验为准。
