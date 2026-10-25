# V3：QGIS五级分区、县域评价与出图

本指南按QGIS 3.44官方文档编写，其他版本菜单文字可能略有不同。以后正式分级、配色和版式统一在QGIS完成；Python仅准备评分表和统计评价，不再自动生成正式地图。已有图集保留作历史成果。

## 1. 先确定评价口径

等级编码固定为：**5极高、4高、3中、2低、1极低**；图例按极高→高→中→低→极低展示。

假设自然断点的4个内部分界值从低到高为b1、b2、b3、b4。等级边界等号归较低级：

|等级|评分区间|
|---|---|
|极高，5|score>b4|
|高，4|b3<score≤b4|
|中，3|b2<score≤b3|
|低，2|b1<score≤b2|
|极低，1|score≤b1|

**“高＋极高”对应score>b3，阈值是“中—高”的分界点；“高—极高”分界点b4只用于识别极高等级。**

评价用全区OOF评分，不用训练内评分；县域只做汇总，不重新计算县内自然断点。空间与行政两套图分别分级并保存各自断点，不能假定同色对应同一绝对评分。不同模型若各用自己的断点，需要同时看高等级面积；如需相同绝对阈值对比，须另设预先固定阈值的实验，不能混用。

## 2. 主指标和配套指标

对任一县域，令N为建模单元数、P为清单滑坡单元数、A为建模面积；N_H、P_H、A_H分别为高＋极高等级内的对应数量。

|指标|公式|用途|
|---|---|---|
|高＋极高清单覆盖率R_H|P_H/P|这两个等级识别到该县多少清单滑坡|
|高＋极高面积占比S_H|A_H/A|划出多少面积作为高等级|
|高＋极高内清单正例比例D_H|P_H/N_H|划出的单元中，有多少已在滑坡清单内|
|面积富集倍数E_H|R_H/S_H|高等级内单位面积的清单正单元密度，相对全县平均的倍数|
|漏识别数|P−P_H|有多少清单滑坡落在中、低、极低等级|

例如R_H=60%、S_H=20%，E_H=3：20%的面积包含60%的清单滑坡。这里计数的是**含滑坡记录的斜坡单元**，不是滑坡点个数，也不是滑坡斑块面积；D_H不能直接等同现场核实的精确率，未标注单元不一定无滑坡。无正例县域R_H不计算，不记为0。

主报告使用R_H＋S_H，AUC/AP继续评价连续评分。C20保留为历史补充，不再作为正式分区的主指标。自然断点没有优化清单覆盖率，不能事后移动断点或类别数来提高评价结果。该覆盖率评价的是**OOF分区识别效果**，不是训练内拟合程度。

## 3. 准备QGIS评分表

在本地项目根目录运行（输出目录须不存在）：

```powershell
python tools/evaluate_v3_grades.py prepare --out results/qgis_inputs_v3
```

输出`spatial_qgis_scores.csv`和`admin_qgis_scores.csv`，各25,509行，只含建模单元。字段包括`unit_id`、`score`、`label`、`area_m2`、`county`、`is_train`；不产生断点或地图。`score`是三个种子的平均OOF评分；本脚本不读取正在运行的因子解释。

## 4. 加载几何并按ID连接

1. QGIS中新建项目，加载`data/slope_units/slope_units_final.shp`；`.shp/.shx/.dbf/.prj/.cpg`放在一起。不要因显示错位而随意修改图层原坐标系。
2. “数据源管理器→分隔文本”：加载一套评分CSV，编码UTF-8，几何选择“无几何（仅属性表）”。检查`unit_id`为整数、`score`为小数，不是文本。
3. 斜坡图层“属性→连接”，用两张表的`unit_id`连接；自定义字段前缀为`m_`。空间和行政分别用独立图层，不能把两套CSV堆叠后连接。
4. 用表达式`"m_is_train" = 1`选择建模单元，**导出所选要素**为GeoPackage，目标坐标系选EPSG:32649，命名如`v3_spatial_model_units`。确认25,509条、`m_score`无空值、`m_label`合计661。
5. 之后只在该GeoPackage图层操作。`m_area_m2`沿用训练评价的投影平面面积，保证统计分母一致；不要直接用经纬度坐标面积。若另算几何面积作核对，米制图层可用`area($geometry)`；不任意替换已核验面积。

430个被排除河道单元不参与自然断点或模型评价。需要画完整研究区时，可从原始几何提取未连接到评分表的430个单元，作为单独规则图层赋极低颜色；其原始清单记录不改，图注注明规则赋值。不要把河道补0后一起重新计算断点。

按ID连接和输出GeoPackage见[QGIS字段连接工具](https://docs.qgis.org/3.44/en/docs/user_manual/processing_algs/qgis/vectorgeneral.html#join-attributes-by-field-value)。

## 5. 全区自然断点五级分类

1. 建模单元图层“属性→符号系统→分级（Graduated）”。
2. 值选`m_score`，方式选“自然断点（Jenks）”，类别数5，点击“分类”。自然断点使同级评分尽量接近，不同级尽量分开。[官方分级说明](https://docs.qgis.org/3.44/en/docs/user_manual/working_with_vector/vector_properties.html#graduated-renderer)
3. 从低到高改标签为极低、低、中、高、极高。颜色建议依次`#dcecc9`、`#a8d18d`、`#f6e59a`、`#ee9a59`、`#c84435`；单元边线设透明或很细。
4. 记录全部4个断点及最大值、评分源、QGIS版本、分类方法和类别数；保存`.qml`样式及`.qgz`工程。不要只记录图例中经过四舍五入的数。
5. 保持这套全区断点，制作任何县域图时均不重新点“分类”。局部图只调整显示范围，不重新计算阈值。

如需读取未四舍五入断点，分类完成后选中该图层，在“插件→Python控制台”执行：

```python
layer = iface.activeLayer()
[(r.lowerValue(), r.upperValue()) for r in layer.renderer().ranges()]
```

结果依次为五级的下限/上限，前四级上限就是b1…b4。此命令只读取当前样式。软件版本和算法实现可能使断点与历史Python分级不同；后续评价以**本次QGIS实际保存的断点与等级**为准，不能直接沿用旧表数值。

## 6. 将颜色对应的等级写入属性

只改颜色不会自动生成等级字段。打开字段计算器，新建整数`level`，用下面表达式；将b1…b4替换为刚才读取的完整数值：

```sql
CASE
WHEN "m_score" IS NULL THEN NULL
WHEN "m_score" <= b1 THEN 1
WHEN "m_score" <= b2 THEN 2
WHEN "m_score" <= b3 THEN 3
WHEN "m_score" <= b4 THEN 4
ELSE 5
END
```

保存后检查`level`仅为1—5、无空值，再把符号系统改为**分类（Categorized）→level**，按5极高→1极低排列并使用同样颜色。这样图面和统计都引用同一个字段，不会出现“图里是高、表里却是中”。可另建整数`high_flag`，表达式`CASE WHEN "level" >= 4 THEN 1 ELSE 0 END`。

## 7. 输出县域评价表

从该建模图层导出CSV，包含`unit_id`和`level`，勾选导出字段名、不输出几何，**不要只导出当前选中部分**，确保25,509行。保存为例如`results/qgis_spatial_grades.csv`。

回到项目终端：

```powershell
python tools/evaluate_v3_grades.py evaluate --protocol spatial --grades results/qgis_spatial_grades.csv --out results/qgis_spatial_evaluation
```

行政图同理，改`--protocol admin`及文件名。输出：

- `county_metrics.csv`：每县R_H、S_H、D_H、E_H、漏识别数。
- `overall_metrics.csv`：全区汇总，不能用县域覆盖率的简单平均替代。
- `five_grade_statistics.csv`：按极高、高、中、低、极低分别统计单元数、面积与清单正单元数。
- `method.json`：评分、等级来源指纹和评价口径。

脚本检查ID完整、等级方向正确、同分不跨级，以及全区统一阈值；不会改变等级或模型。Python表中的面积/标签/县域来自冻结输入，避免GIS连接字段混淆。也可在QGIS按`m_county`聚合`m_label`、`m_area_m2`、高等级内标签与面积的和，按第2节公式计算；面积或标签列有空值时先修正连接，不直接统计。

## 8. QGIS版式与导出

1. 图层顺序：地形阴影在底、易发性在中、行政边界和标注在上。易发性透明度可设45%（不透明度55%）；县界用无填充细线。
2. “项目→新建打印布局”，选A4/A3横向，添加地图并调整范围；锁定地图图层与样式。
3. 添加图例、比例尺、指北针、坐标网格、数据与方法说明。图例仅保留5级，顺序极高→高→中→低→极低。
4. 图注写“V3三种子平均OOF易发性评分；全区Jenks五级；河道规则赋极低；评分非校准概率”。县域沿用全区断点。
5. 导出PDF用于排版，PNG建议300 dpi；检查中文字体、图例顺序、比例尺单位及裁切。保留GeoPackage、`.qgz`和`.qml`，地图和评价CSV一起归档。[QGIS布局导出说明](https://docs.qgis.org/3.44/en/docs/user_manual/print_layout/create_output.html)

当前`results/v3_county_borders_atlas_20261021/`仍是已认可的历史图集，本次不重绘或覆盖；新成果另建QGIS项目与输出目录。
