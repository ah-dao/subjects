# 五级自然断点法制图（现行分级）

按用户2026-10-20的要求，现行地图使用Fisher-Jenks自然断点法，不再使用初版等频分级。

- 对每套协议的25,509个建模单元三种子OOF连续分数分别计算五级断点。
- 使用全部单元，不抽样；每个单元计权1，不按多边形面积加权。相同分数不拆组。
- 最小化五组内部平方误差之和，采用精确一维动态规划；通过小样本全部分割穷举独立验证。
- 四个断点为前四级的含等号上界：第一类score≤b1，第二类b1<score≤b2，依此类推。
- 430个河道单元不参与断点计算，最后按既定规则固定score=0、level=1。
- 不使用滑坡标签选断点，不人为规定高易发区面积。自然断点反映分数分布，不能单独证明划出的等级就是实际发生概率或真实危险区边界。
- 25,939个单元全部保留。连续分数、AUC/AP、面积排序覆盖指标不改变，改变的是五级等级与配色。

本目录独立于旧诊断代码，运行时仅替换旧制图函数的分级策略，不编辑原文件。原服务器分组置换任务可继续运行，不需重训、重启或重新上传解释补丁。

成果目录：results/pixel_tabpfn_diagnostics_v1/natural_breaks_20261020，含每套协议的完整Shapefile/CSV、等级和分数GeoTIFF、预览PNG/PDF、断点表、分级统计表、grading_comparison.csv和核验记录。各协议分别取断点，相同等级不代表相同绝对分数阈值。

旧dist/pixel_tabpfn_initial_maps_20261020.zip是分位数历史版，现行成果包为dist/pixel_tabpfn_natural_breaks_20261020.zip；报告和GIS使用新包。

如需复现，使用本机已有GIS环境，输出目录必须不存在：

```powershell
python experiments/pixel_tabpfn_mapping_v2/mapping.py --diagnostics results/pixel_tabpfn_diagnostics_v1/initial_release --out results/pixel_tabpfn_diagnostics_v1/natural_breaks_new
```

精确Jenks单元测试：

```powershell
python -m unittest discover -s experiments/pixel_tabpfn_mapping_v2 -p test_jenks.py
```

本图继续属于历史回溯OOF验证图，非全数据重拟合图或未来时点预报。矢量覆盖所有单元，90m栅格中极小面可能不占像元，逐单元完整性以矢量/CSV为准。
