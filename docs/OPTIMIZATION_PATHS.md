# 滑坡易发性分析优化路径（斜坡单元方案）

本文档描述从当前栅格 Patch 模型转向**斜坡单元 + GraphSAGE + Transformer** 的完整技术路线，涵盖数据准备、时序特征工程、模型架构与训练策略。

研究区：三峡库区消落区；斜坡单元数：26068；滑坡标签：2000-2021，含准确年月日。

---

## 一、总体架构转变

### 1.1 从栅格 Patch 到斜坡单元

| 维度 | 当前方案（栅格 Patch） | 目标方案（斜坡单元） |
|------|----------------------|---------------------|
| 样本单位 | 256×256 矩形 | 不规则多边形 |
| 样本数量 | ~22000 个 patch | 26068 个斜坡单元 |
| 空间关系 | 隐式（CNN 感受野） | 显式（图邻接） |
| 特征来源 | 逐像素栅格值 | 单元内聚合统计 |
| 邻域建模 | CNN 卷积核 | GraphSAGE 消息传递 |
| 模型骨干 | CNN+CBAM+Transformer+SPP | GraphSAGE+Transformer |
| 输出形式 | 每个像素的概率 | 每个斜坡单元的概率 |

### 1.2 为什么选择斜坡单元

1. **物理意义**：斜坡单元是地形的自然分割单元，对应真实的水文与地质边界，比规则网格更符合滑坡机制
2. **样本量充足**：26068 个单元与现有 patch 数量级持平，不存在小样本过拟合问题
3. **空间关系显式建模**：图神经网络直接编码斜坡单元间的邻接关系，捕获滑坡的空间传导效应
4. **特征工程更灵活**：节点特征可以是任意维度的统计向量，不受卷积核尺寸限制

---

## 二、已知数据统计

### 2.1 斜坡单元

| 指标 | 数值 |
|------|------|
| 斜坡单元总数 | 26068 |
| 无效几何 | 28（0.1%，已用 `shapely.make_valid` 修复） |
| 有效单元 | 26040（修复后） |

### 2.2 滑坡点与斜坡单元关联（QGIS 统计结果）

| 指标 | 数值 |
|------|------|
| 滑坡点总数 | 986（2000-2021，含精确年月日） |
| 有滑坡的斜坡单元（≥1） | **846 处** |
| 单次滑坡单元（=1） | 727 处 |
| 复发滑坡单元（≥2） | **119 处** |
| 复发占比 | 119/846 = **14.1%** |
| 无效几何单元 | 28 处（已排除） |

### 2.3 复发单元统计

| 指标 | 数值 |
|------|------|
| 复发单元数 | 119 |
| 复发单元贡献的滑坡事件 | 986 - 727 = 259 个 |
| 平均复发次数 | 259/119 ≈ 2.18 次/单元 |

**结论**：14.1% 位于 10%-30% 区间，需在节点特征中加入 4 维复发相关特征。

---

## 三、数据准备

### 3.1 斜坡单元数据与图构建

- 已有：研究区斜坡单元 shapefile，共 26068 个多边形
- 几何修复：用 `fix_slope_units.py`（`shapely.make_valid`）修复 28 个无效几何
- 导入 GEE：打包 `.shp/.shx/.dbf/.prj`（必须含投影文件）→ 上传为 Asset
- 坐标系统：CGCS2000 → 统一转为 WGS 84（EPSG:4326），用 QGIS"重新投影"工具而非直接改属性
- 图构建（Python 端）：基于多边形邻接关系或 Delaunay 三角剖分生成 `edge_index`

### 3.2 关联滑坡点与斜坡单元（QGIS 流程）

工具：`矢量 → 分析工具 → 计算点在多边形内`

```
前提条件：
  1. 滑坡点 CSV 必须正确加载为点图层
     - 菜单：图层 → 添加图层 → 添加文本数据图层
     - X 字段选经度列，Y 字段选纬度列
     - 几何坐标参考系：EPSG:4326
     - 如加载后看不到点，先检查坐标系是否与斜坡单元一致
  2. 两个图层坐标系必须一致（看右下角 EPSG 标识）
  3. CSV 图层图标必须是"三个点连线"而非"表格"，否则需重新加载或先另存为 .shp

操作步骤：
  1. 输入 Polygons：斜坡单元（修复后的 .shp）
  2. 输入 Points：滑坡点图层
  3. 计数字段名：landslide_count
  4. 输出：slope_units_count.shp

结果验证：
  - 打开属性表，确认新增 landslide_count 字段
  - 选中 landslide_count >= 2 → 查看复发单元数量（119）
  - 选中 landslide_count >= 1 → 查看总滑坡单元数量（846）
```

### 3.3 静态地形因子

5 个环境因子，从已有 GeoTIFF 中按斜坡单元提取 zonal 统计：

| 因子 | 数据源 | 统计量 |
|------|--------|--------|
| elevation | SRTM 30m | mean, std, min, max |
| slope | SRTM 派生 | mean, std |
| aspect | SRTM 派生 | mean |
| TRI | SRTM 派生 | mean |
| curvature | SRTM 派生 | mean |

**提取方式**：GEE 导出多波段栈（1 次 Export），本地用 `rasterstats` 做 zonal stats（几秒/波段）。

### 3.4 静态几何特征

从 shapefile 直接计算：

| 特征 | 计算方式 |
|------|---------|
| area | 多边形面积 |
| compactness | 4π·area / perimeter² |
| shape_index | perimeter / (2√(π·area)) |

### 3.5 时序数据：分层时间分辨率策略

消落区滑坡的核心触发因子是水位变化，不同因子的时间分辨率应匹配其物理响应时滞：

| 因子 | 时间分辨率 | 理由 |
|------|-----------|------|
| **水库水位** | **月度** | 145→175m 的涨落是月内事件，年均水位无意义；消落期 3 个月累计降幅是滑坡的直接触发因子 |
| NDVI | 年度（生长季均值） | 植被变化缓慢，年均 NDVI 是文献标准做法 |
| 降雨 | 年度（年最大日降雨/暴雨频次） | 触发滑坡的是极端事件，CHIRPS 5km 分辨率在月尺度噪声大 |

### 3.6 GEE 数据导出策略：多波段栈一次导出

**核心优化**：避免对 26K 单元逐个 `reduceRegion`（会超时），改为导出完整栅格栈，在 Python 端用 `rasterstats` 做本地 zonal stats。

```javascript
// NDVI：32 年堆叠为 32 波段，一次导出
var yearlyNdvi = ee.ImageCollection(
  ee.List.sequence(1990, 2021).map(function(year) {
    return landsatCol
      .filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 9, 30))
      .median()
      .select('NDVI')
      .rename('ndvi_' + ee.Number(year).format('%d'));
  })
);

Export.image.toDrive({
  image: yearlyNdvi.toBands(),
  description: 'ndvi_32yr_stack',
  scale: 30,
  region: studyArea,
  maxPixels: 1e13
});
```

降雨同理导出为 32 波段栈（年最大日降雨 + 年累计降雨 = 64 波段）。

| 环节 | 逐单元 reduceRegions | 多波段栈导出 + 本地 zonal stats |
|------|---------------------|-------------------------------|
| GEE 任务数 | 64+ 次 Export | **3 次 Export** |
| GEE 超时风险 | 高（26K 单元） | 无 |
| 本地处理时间 | — | ~10 秒/波段 |
| 总人工干预 | 反复提交任务 | **10 分钟** |

### 3.7 水位数据（逐日 Excel）

水位数据是整个水库统一的值，不需要 GEE：

| 时期 | 水位 | 说明 |
|------|------|------|
| 1990-2003 | 天然河道水位（~60m 变幅） | 未蓄水 |
| 2003-2006 | 135-139m | 围堰蓄水期 |
| 2006-2008 | 156m | 初期蓄水期 |
| 2008-至今 | 145-175m 周期调度 | 正常运行期 |

数据源：长江三峡集团调度公报 / 长江水文网（宜昌站逐日水位）。

**水位特征提取脚本**：`extract_water_features.py`，从 Excel 读取逐日水位 → 聚合为月尺度 → 按滑坡日期截断 → 输出 7 个全局水位特征。

---

## 四、时序特征工程：因果对齐

### 4.1 核心原则

滑坡标签有准确年月日，时序特征必须**按滑坡日期截断**，禁止使用滑坡发生后的数据（数据泄漏）。

```
单元 A: 2008-06-15 滑坡 → 只用 [1990, 2008-06] 的数据
单元 B: 2015-03-20 滑坡 → 只用 [1990, 2015-03] 的数据
单元 C: 未滑坡          → 用 [1990, 2021] 全窗口数据，标签 = 0
```

### 4.2 时序特征设计

#### NDVI 年度时序（按截止年截断后提取）

| 特征 | 计算方式 | 物理意义 |
|------|---------|---------|
| long_trend_slope | 截止年内 NDVI 线性回归斜率 | 植被长期退化/恢复趋势 |
| long_cv | 标准差 / 均值 | 植被稳定性 |
| pre_event_2yr_drop | 截止前 2 年 NDVI 均值 - 长期均值 | 滑坡前植被异常下降 |
| max_interannual_change | max(\|diff\|) | 最大年际突变 |

#### 降雨年度时序

| 特征 | 计算方式 | 物理意义 |
|------|---------|---------|
| annual_max_rain_mean | 截止年内年均最大日降雨 | 极端降雨强度 |
| heavy_rain_trend | 暴雨日数线性趋势 | 极端降雨频率变化 |
| pre_event_2yr_extreme | 截止前 2 年最大日降雨 | 滑坡前短期极端降雨 |
| antecedent_30d_max | 截止前最大 30 日累计降雨 | 前期降雨触发信号 |

#### 水位月度时序（按截止月截断后提取）

| 特征 | 计算方式 | 物理意义 |
|------|---------|---------|
| exposure_months | 截止月前的淹没月数 | 累计暴露时长 |
| max_monthly_drawdown | max(\|月降幅\|) | 历史最大单月水位降幅 |
| drawdown_3m_cumulative | 截止前 3 个月累计降幅 | **消落期核心触发指标**（2-5 月 175→145） |
| drawdown_1m_prior | 截止前 1 个月降幅 | 最近水位变化 |
| rapid_drawdown_events | 月降幅 >3m 的次数 | 骤降事件计数 |
| mean_drawdown_rate | 月均降幅 | 平均水位下降速率 |
| last_month_level | 截止月水位值 | 当前水位状态 |
| water_range_total | 截止期内水位波动总幅度 | 该单元经历的最大水位变幅 |

### 4.3 多次滑坡单元的处理：首次事件法

对于同一斜坡单元多次滑坡（如 2005 和 2017 各一次）：

```
策略：标签 = 1（发生滑坡）
      特征截断到首次滑坡日期（2005）
      后续滑坡信息编码为复发特征（见 4.4）

理由：
  1. 首次滑坡决定了该单元"先天不稳定"的特性
  2. 后续滑坡更多是水位周期性触发，不是单元属性变化
  3. 在 GNN 中每个 unit_id 只能是一个节点，不能重复入图
```

### 4.4 复发特征（4 维，因复发占比 14.1%）

| 特征 | 计算方式 | 物理意义 |
|------|---------|---------|
| recurrence_count | 该单元历史滑坡总次数 | 越高越不稳定 |
| recurrence_flag | 是否复发（1/0） | 二值标记 |
| years_to_recurrence | 首次→末次滑坡的时间跨度（年） | 短=快速复发，持续不稳定 |
| inter_event_drawdown | 两次滑坡之间累计水位降幅 | 水位反复冲击导致复发 |

`years_to_recurrence` 和 `inter_event_drawdown` 需要首次/末次滑坡日期，通过 QGIS"以位置连接属性"的聚合模式（一对多汇总，`min(date)` + `max(date)`）提取。

### 4.5 未滑坡单元的处理

未滑坡单元使用全窗口数据（1990-2021），标签 = 0。"32 年观测期内未发生滑坡"是强负信号。复发特征全部置 0。

### 4.6 最终节点特征表

| 类别 | 维度 | 特征 |
|------|------|------|
| 静态地形 | 5 | elevation_mean, slope_mean, aspect_mean, TRI_mean, curvature_mean |
| 静态几何 | 3 | area, compactness, shape_index |
| NDVI 年度时序 | 4 | long_trend_slope, long_cv, pre_event_2yr_drop, max_interannual_change |
| 降雨年度时序 | 4 | annual_max_rain_mean, heavy_rain_trend, pre_event_2yr_extreme, antecedent_30d_max |
| 水位月度时序 | 8 | exposure_months, max_monthly_drawdown, drawdown_3m_cumulative, drawdown_1m_prior, rapid_drawdown_events, mean_drawdown_rate, last_month_level, water_range_total |
| 复发特征 | 4 | recurrence_count, recurrence_flag, years_to_recurrence, inter_event_drawdown |
| **合计** | **~28** | |

---

## 五、模型架构

### 5.1 总体设计：GraphSAGE + Transformer（节点级预测）

```
输入: 图 G(V, E)
  V = 26068 个斜坡单元，每个节点 ~28 维特征
  E = 多边形邻接 / Delaunay 三角剖分定义的边

  节点特征 X (26068 × 28)
    ↓
  Linear Projection (28 → 64)           # 特征维度对齐
    ↓
  GraphSAGE Layer 1 (64 → 64)           # 邻域消息传递，1 跳
  GraphSAGE Layer 2 (64 → 64)           # 2 跳覆盖范围
    ↓
  Transformer Conv (2 层, 4 头)          # 全局长程依赖
    ↓
  FC (64 → 32 → 1) + Sigmoid            # 逐节点概率输出
```

### 5.2 为什么用 GraphSAGE 而非 GAT

| 对比 | GAT | GraphSAGE |
|------|-----|-----------|
| 学习方式 | 直译式（Transductive） | 归纳式（Inductive） |
| 泛化能力 | 无法处理未见节点 | 可泛化到新斜坡单元 |
| 计算效率 | 注意力计算开销大 | 均值/最大值聚合高效 |
| 小样本稳定性 | 注意力权重易过拟合 | 聚合操作更稳定 |
| 适用性 | 消落区邻接关系相对均匀 | 均值聚合已足够 |

消落区斜坡单元间的邻接关系相对均匀（共享边界），GAT 的注意力机制增益有限。GraphSAGE 的归纳能力和训练稳定性更实用。

### 5.3 Transformer 的作用

GraphSAGE 的消息传递本质是局部的（2 跳 = 2 层），Transformer 补充全局长程依赖：

- 捕获远距离斜坡单元间的相似性（如相同高程带的单元可能有相似的滑坡模式）
- 学习全局空间模式（如消落区下部单元群的整体失稳趋势）

### 5.4 模型实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv

class SlopeUnitGNN(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=64, num_heads=4, 
                 dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.transformer = TransformerConv(
            hidden_dim, hidden_dim // num_heads, 
            heads=num_heads, dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = F.relu(self.sage1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.sage2(x, edge_index))
        x = self.dropout(x)
        x = self.transformer(x, edge_index)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))
```

---

## 六、训练策略

### 6.1 数据划分：空间 K-Fold 交叉验证

**为什么不用时间交叉验证**：2015-2021 每年仅约 10 起滑坡，年度滑动窗口测试集正样本过少（10 个/年），AUC 置信区间极宽（±0.15），无统计意义。986 个滑坡点集中在 2003-2010 蓄水期，后期稀疏。

采用**空间划分**：

| 策略 | 做法 | 优缺点 |
|------|------|--------|
| 空间 K-Fold | 按子流域/行政区划 5 折，每折 4:1 训练:测试 | 最稳健，消除空间自相关，适合论文 |
| 随机划分 | 70/15/15 | 简单但可能高估 AUC（空间泄漏） |

**推荐**：5 折空间交叉验证，汇报平均 AUC ± 标准差。

### 6.2 类别不平衡处理

滑坡正样本 846 个，总单元 26040 个，正样本占比约 3.2%，严重不平衡：

- **加权 BCE Loss**：`pos_weight = num_negatives / num_positives ≈ 30`
- **Focal Loss**（可选）：对易分类样本降权，聚焦难样本

### 6.3 正则化

- Dropout = 0.3（GraphSAGE 和 FC 层之间）
- Weight Decay = 1e-4
- Early Stopping（patience=20，监控 val AUC）
- 归一化：全局 MinMax 归一化到 [0, 1]

### 6.4 超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| hidden_dim | 64 | 节点特征维度，26K 节点足够 |
| input_dim | 28 | 最终节点特征维度 |
| GraphSAGE 层数 | 2 | 2 跳覆盖邻域，更多层易过平滑 |
| Transformer 层数 | 2 | 全局增强，不宜过多 |
| num_heads | 4 | 多头注意力 |
| dropout | 0.3 | 正则化 |
| learning_rate | 1e-3 | Adam |
| batch_size | 全图 | 26K 节点可全图训练（Full-batch） |

### 6.5 训练开销估算

| 环节 | 耗时 | 硬件 |
|------|------|------|
| 图构建（Delaunay, 26K 节点） | ~5 秒 | CPU |
| 训练（200 epochs, Full-batch） | ~2-5 分钟 | Colab T4（免费） |
| 5 折交叉验证 | ~25 分钟 | Colab T4 |
| 推理 | ~0.1 秒 | CPU 即可 |
| 水位特征提取 | ~30 秒 | CPU |

**相比现有 CNN 方案，训练开销降低 10-20 倍。**

---

## 七、预测与出图

### 7.1 预测流程

```
1. 加载训练好的模型 + 归一化参数 + edge_index
2. 输入全图 (26040 节点 + 邻接边)
3. 前向传播 → 每个节点的滑坡概率 [0, 1]
4. 按概率分 5 级：极低(0-0.2)/低(0.2-0.4)/中(0.4-0.6)/高(0.6-0.8)/极高(0.8-1.0)
5. 回填到 shapefile 属性表
6. QGIS 矢量分级填色出图
```

### 7.2 与现有出图流程的对接

| 对比 | 现有（栅格） | 新方案（矢量） |
|------|------------|--------------|
| 输出格式 | GeoTIFF (像素级概率) | Shapefile (多边形级概率) |
| 渲染方式 | 栅格配色 | 矢量分级填色 |
| 空间精度 | 像素级 | 单元级 |
| 出图工具 | matplotlib / QGIS | QGIS（推荐 RdYlGn_r 配色） |

---

## 八、实施路线与配套脚本

### 8.1 已完成的准备工作

| 步骤 | 工具 | 完成情况 |
|------|------|---------|
| 斜坡单元 shapefile 加载 | QGIS | ✅ 26068 个单元 |
| 几何修复 | `fix_slope_units.py` | ✅ 28 个无效几何已修复 |
| 滑坡点 CSV 提取 | `extract_landslide_points.py` | ✅ 986 个滑坡点（2000-2021） |
| 斜坡单元滑坡计数 | QGIS（计算点在多边形内） | ✅ 846 有滑坡，119 复发 |
| 水箱数据准备 | Excel | 待确认 |

### 8.2 分阶段实施

```
阶段 1: 数据准备（GEE 导出 + 本地 zonal stats）
  ├─ 上传斜坡单元 shapefile 到 GEE
  ├─ 导出 32 波段 NDVI 栈 + 64 波段降雨栈（各 1 次 Export）← 待执行
  ├─ 本地 rasterstats 提取静态地形 + NDVI/降雨时序特征
  ├─ 水位月度特征提取（extract_water_features.py）
  ├─ QGIS 提取首次/末次滑坡日期（一对多汇总 join）
  └─ 按滑坡日期截断时序，构建节点特征表 (26040 × 28)

阶段 2: 基线验证（XGBoost, CPU, ~10 秒）
  ├─ 用 28 维特征训练 XGBoost
  ├─ 查看 AUC 和特征重要性
  ├─ 确认水位特征是否排前列（验证消落区假设）
  └─ AUC > 0.7 则继续

阶段 3: GraphSAGE + Transformer 模型
  ├─ 构建图（多边形邻接 / Delaunay）
  ├─ 训练模型（Colab T4, ~5 分钟）
  ├─ 5 折空间交叉验证（~25 分钟）
  ├─ 对比 XGBoost 基线
  └─ 评估指标：AUC + Recall@Top10%

阶段 4: 预测与出图
  ├─ 全图推理 → 每个单元的概率
  ├─ 回填 shapefile → QGIS 矢量分级出图
  └─ 与现有 CNN 栅格结果对比分析
```

### 8.3 配套脚本清单

| 脚本 | 用途 | 状态 |
|------|------|------|
| `extract_landslide_points.py` | 从 Excel 提取含日期滑坡点 → CSV | ✅ 已创建 |
| `fix_slope_units.py` | 修复 shapefile 无效几何 | ✅ 已创建 |
| `extract_water_features.py` | 从水位 Excel 提取水位月度特征 | ✅ 已创建 |
| `extract_terrain_features.py` | 从 GeoTIFF + shapefile 提取静态地形特征 | 待创建 |
| `extract_temporal_features.py` | 从 NDVI/降雨栈提取时序特征并截断 | 待创建 |
| `build_graph.py` | 构建图邻接关系（Delaunay/共享边界） | 待创建 |
| `train_gnn.py` | GraphSAGE + Transformer 训练 | 待创建 |
| `predict_gnn.py` | 全图推理 + shapefile 回填 | 待创建 |

### 8.4 优先级总结

| 优先级 | 任务 | 预期提升 | 难度 | 开销 |
|--------|------|---------|------|------|
| 1 | 斜坡单元 + 静态地形特征 → XGBoost 基线 | 建立基线 | 低 | CPU, 10 秒 |
| 2 | 加入时序统计特征（NDVI + 降雨年度） | AUC +0.03~0.05 | 低 | GEE 2 次导出 |
| 3 | 加入水位月度时序特征 | AUC +0.05~0.10 | 中 | 水位 Excel + 脚本 |
| 4 | 加入复发特征（4 维） | AUC +0.01~0.03 | 低 | QGIS 聚合 join |
| 5 | GraphSAGE + Transformer 图模型 | AUC +0.03~0.08 | 中 | Colab T4, 5 分钟 |
| 6 | 空间 K-Fold 交叉验证 + 超参数调优 | 稳健性提升 | 中 | 多次训练 |
| 7 | 矢量出图 + 与栅格方案对比 | 论文完整性 | 低 | QGIS |

---

## 九、开源工具与依赖

### 9.1 Python 库

| 库 | 用途 | 安装 |
|----|------|------|
| `torch_geometric` | GraphSAGE, TransformerConv | `pip install torch-geometric` |
| `rasterstats` | 栅格按多边形 zonal stats | `pip install rasterstats` |
| `geopandas` | shapefile 读写与空间操作 | `pip install geopandas` |
| `shapely` | 几何修复（make_valid） | `pip install shapely` |
| `scipy.spatial` | Delaunay 三角剖分构建图 | scipy 内置 |
| `xgboost` | 基线模型 | `pip install xgboost` |
| `xlrd` | 读取 .xls 格式 Excel | `pip install xlrd` |

### 9.2 GEE 数据源

| 数据 | GEE ID | 用途 |
|------|--------|------|
| Landsat 5/7/8 NDVI | `LANDSAT/LT05/C02/T1_L2` 等 | 年度 NDVI 栈 |
| CHIRPS 降雨 | `UCSB-CHG/CHIRPS/DAILY` | 年度降雨栈 |
| SRTM DEM | `USGS/SRTMGL1_003` | 静态地形因子 |

### 9.3 桌面工具

| 工具 | 用途 | 备注 |
|------|------|------|
| QGIS LTR 3.40 | 空间操作、滑坡统计、出图 | 推荐 LTR 版避免智能应用控制拦截；设置→选项→语言可切换中文 |
| Google Earth Engine | 栅格导出 | 免费使用 |

### 9.4 云服务器

| 平台 | GPU | 用途 | 费用 |
|------|-----|------|------|
| Google Colab | T4 16GB | 模型训练 | 免费 |
| AutoDL | 3090 24GB | 大规模实验 | ~2-3 元/小时 |
| 本地 | CPU | XGBoost 基线 + 数据处理 | 零 |

**模型训练在 Colab 免费 T4 上完全够用，无需付费 GPU。**
