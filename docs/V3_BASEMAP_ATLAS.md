# V3正式底图图集

后续正式分级、制图统一使用QGIS，方法见[QGIS五级分区与评价](V3_QGIS_GRADING_AND_EVALUATION.md)。主报告改用高＋极高等级清单覆盖率及面积占比；本页现有图集和命令保留作历史复现，不作为后续正式出图入口。

唯一可用目录为`results/v3_county_borders_atlas_20261021/`。其他多轮出图、诊断绘图与对应结果ZIP已清理，不作为交付入口。该目录的68个文件原样保留，包含16张PNG和16张PDF，以及GIS配套文件、`county_selection.csv`、`map_index.json`、`metadata.json`、`manifest.json`。

## 图面口径

空间、行政两套OOF各含地形、行政、等高线、综合四张全图，以及云阳、忠县、万州、涪陵四张县域综合图。全图只画与研究区相交县域的完整原始行政边框，包含其研究区外部分；局部图不画行政边框。无水系、标题和方法说明文字。

易发性不透明度55%；局部DEM为60米网格、50米等高距、250米计曲线。县域按各协议高+极高等级面积前三取并集，选择依据保存在目录内。评分与五级自然断点未因叠加而改变，430河道单元沿用规则赋值。

行政名称沿用历史数据NAME99；现有坐标转换无新增测量控制点，图面不用于米级边界判断。OOF图是历史回溯地域迁移验证图，不是全量重拟合的未来风险预测图。

## 清理后的重建

现有正式图集可直接使用。`src.v3.map_atlas_clean`仍依赖基础评分地图和早期图集生成的描述表；这些可重建中间目录已清理，不能直接运行默认命令。需要时在新的临时目录依次生成：

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) '.geo_deps')
python -m src.v3.mapping --results results/v3_received_stats84 --out results/v3_maps_rebuild
python -m src.v3.map_atlas --maps results/v3_maps_rebuild --out results/v3_basemap_atlas_rebuild --alpha 0.55
python -m src.v3.map_atlas_clean --maps results/v3_maps_rebuild --previous results/v3_basemap_atlas_rebuild --out results/v3_county_borders_atlas_rebuild --alpha 0.55
```

上述输出目录须不存在；原始斜坡几何、DEM、行政图层和GIS依赖须可用。重建后先核对清单、评分口径与图面，再决定替换正式成果。本轮没有重新出图或训练。
