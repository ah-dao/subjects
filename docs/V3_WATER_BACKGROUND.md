# V3事件前水位背景与折内日期

## 本轮规则

背景统一为 **2001-01-01 ≤ 日期 < 事件/参照月月初**。例如2015年6月事件，使用2001-01-01至2015-05-31，6月不进入统计。正样本沿用study_first_landslide_date的月份；同月日期精度不一致时统一取月初，避免当月事后数据混入。

非滑坡单元按每个外折训练侧正样本的年—月联合经验分布抽参照月。空间与行政使用各自训练侧分布；每折为全部25,509单元保存参照日期与train/test角色。抽样用固定日期种子20261020及单元ID哈希，与模型种子、运行顺序和4,000条上下文选择无关。测试正样本不参与供体分布。各模型在同一协议/折必须共用同一份日期表。

## 替换的特征

|V3字段|含义|
|---|---|
|preevent_inundation_fraction|上述背景内的面积聚合日淹没频率|
|preevent_wet_dry_transition_count|从2001年起累计到事件月前的干湿状态转换次数；完整往返是两次，不再保留全期累计列|
|preevent_max_abs_monthly_level_change|背景内相邻月均水位差的最大绝对值，包含上涨与下降|
|ant_inund_days_1m|事件/参照月之前一个自然月的淹没天数，纠正旧3m字段误名|
|ant_drawdown_3m|前三个自然月水位首值减末值，负数为净上涨|

表格淹没频率/转换次数保留V2的完整DEM像元等权聚合与0.5m高程分桶；前1月淹没天数用未分桶DEM高程。像元袋端只复用旧缓存前4列静态地形及物理面积权重，水位4通道和水位—地形联合量全部按各单元参照月重算，使用全部有效像元，不抽64个像元。44维统计仍含20维静态地形、20维事件前水位分布、4维联合量。

不读取旧缓存的tab、summary、zone作为新水位特征，也不把全期历史水位名称继续写成V3事件前统计。新字段名、顺序保存在输出schema.json中。

## 缺测处理与本轮限制

清溪场在2001—2021年共缺620天，包含2010年全年与2011年大部分时段；其他站约7—9天缺测。默认只允许沿用之前已知水位最多3天（causal forward hold），并记录原始缺失、沿用天数和剩余缺失。此为短缺口估计，不是新增实测值；没有反向填充或长段插值。可用--max-forward-days 0禁用，但须新输出目录。

若2001至参照月前仍有未填缺口，累计转换次数、背景淹没频率、最大月均绝对变化及像元背景水位统计均记缺失，**不把已观测片段当作完整累计，不跨缺口直接相连计数，也不填0**。近期1/3月特征独立检查各自窗口，窗口完整时仍可计算。历史缺口未补齐时，后续年份累计特征也不能自动变完整。

因此部分清溪场匹配单元会缺少背景水位信息。下一阶段应优先寻找同站补充记录；站间校准估算需要单独验证，不能直接拿别站绝对水位替换。当前产物保留原人口，不因缺测删单元，不是已解决所有水文质量问题的最终数据集。

十份视图中，有869—1,076个单元因历史水位缺口无法完整计算背景量，其中正样本固定17个。另有458个非滑坡单元在原完整DEM像元对缓存中没有像元，表格淹没频率/交替次数亦为缺失；两类问题有部分重叠。有效地形像元袋为空的为10个，二者不是同一提取缓存。water_quality.csv明确分别记录table_dem_pixels、valid_bag_pixels及history_complete，后续只能用训练侧规则处理缺失，不能默认为0。

## 文件组织：两套五折共10份视图

```text
features/v3/fold_water/
  spatial/fold0 ... fold4/
  admin/fold0 ... fold4/
    reference_dates.csv
    reference_distribution_train_positive.csv
    water_features.csv
    water_quality.csv
    water_pixel_packet.npz
    summary.json
    DONE.json
  config.json / schema.json / fold_summary.csv / manifest.json
```

每折都有全部25,509个单元；模型只拟合其中train角色。训练/测试同折共用同一个“训练侧供体分布”，正样本始终用自己的事件前数据。正样本和静态信息可共用缓存，负样本随外折参照日期变化；不需要为每个模型种子重复提取原始GIS/时序数据。

此目录描述水位阶段，包含水位5列、44维像元统计和10份参照日期，阶段缓存的ready_for_training保留false。**后续完整组装已完成，当前训练入口为features/v3/model_inputs，详见[V3完整实验指南](V3_EXPERIMENT_RUNBOOK.md)。** 原始年度/月度矩阵继续复用，不必重新从遥感栅格提取；未混入V2动态列。

## 4,000条上下文

本轮不优化选择规则。先保持上下文预算、原训练正样本保留规则、负样本选择逻辑及模型参数，尽可能复用已记录的训练侧上下文ID，用V3数据重跑基线。先纠正数据、后对照采样，才能分辨收益来源。

日期供体使用全部外折训练正样本，不能由抽出的4,000条上下文决定。后续优化采样只能使用训练侧数据；每种采样方案、模型方案共用本轮固定的日期表，测试人口保持完整。若未来加内折调参，内折也须按自己的训练正样本建立供体日期分布，不能直接复用外折供体作为严格内折验证。

## CPU运行

本地已运行完整10份；不需要启动GPU训练。补丁供服务器复算，原始DEM对、静态像元缓存和分折文件需已存在。

本地PowerShell上传：

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

服务器准备环境和站点（已有时只verify）：

```bash
cd /3250707052/landslide
python3 -m venv .venv-v3-prep  # 已有时跳过
source .venv-v3-prep/bin/activate
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-v3-prep.txt
python -m unittest discover -s tests/v3 -p 'test_*.py'
# features/v3/station_matching尚不存在时运行build；已存在时直接verify
python -m src.v3.station_matching build
python -m src.v3.station_matching verify
```

构建水位视图（如服务器静态缓存实际在别处，修改--prepared；不更改TabPFN环境）：

```bash
mkdir -p logs
nohup python -u -m src.v3.prepare_water build \
  --prepared results/pixel_bag_repr_v1/prepared \
  > logs/v3_water_background.log 2>&1 < /dev/null &
```

同配置中断可加--resume重试，逐折校验DONE后跳过；改变来源、代码或缺测策略须新--out。完整结束打印ALL DONE后：

```bash
python -m src.v3.prepare_water verify
```

所有均为CPU计算，不需要新torch/CUDA，不训练TabPFN。补丁不含原始数据、大像元数组或本地结果，也不会覆写冻结V2实验代码。
