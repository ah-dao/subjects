# 滑坡易发性预测精度提升方案

## 当前状态

| 项目 | 当前值 | 问题 |
|---|---|---|
| 模型架构 | `LandslideProbabilityModel`（patch 分类） | 输入 256×256 → 输出 1 个概率值，同一 patch 内所有像素共享相同概率 |
| 标签格式 | 标量 0/1，shape=(1,) | 无法表达 patch 内部空间差异 |
| 图像分辨率 | 30m 原始，降采样到 60m 使用 | Colab 免费版 12GB RAM 无法加载 7901×13568 全分辨率 |
| 字体 | 已修复（自动下载 Noto Sans CJK SC） | — |

---

## 目标

30m 全分辨率、逐像素概率输出、汉字正常显示。

---

## 路线一：模型升级 —— Patch 分类 → 像素级分割

### 原理

当前 `LandslideProbabilityModel`：256×256 patch → CNN → Transformer → SPP → FC → **1 个标量概率**。

已有 `LandslideSegmentationModel`：256×256 patch → CNN → Transformer → Decoder → **256×256 像素概率图**。

```
当前:  (batch, 5, 256, 256) → SPP → FC → Sigmoid → (batch, 1)      【粗粒度】
目标:  (batch, 5, 256, 256) → Decoder → Upsample → Softmax → (batch, 2, 256, 256)  【逐像素】
```

### 需要修改的代码位置

| 文件 | 改动 |
|---|---|
| `src/model_segmentation.py` | 训练时使用 `LandslideSegmentationModel`（已存在，无需新写） |
| `src/dataloader.py` | `__getitem__` 标签从 `(1,)` 改为 `(256, 256)` mask |
| `main.py` 或训练脚本 | 模型初始化改为 `LandslideSegmentationModel`，loss 改为 `CrossEntropyLoss` |
| `predict.py` | 预测时每个 patch 输出 256×256 概率图，直接拼回完整概率图（无需滑窗平均） |

### 预测流程对比

```
当前（patch 分类）:
  滑窗切 patch → 每个 patch 1 个概率 → 写入中心区域 → 重叠区域取平均
  缺点: 单 patch 内无空间变化，依赖重叠平均

目标（像素分割）:
  滑窗切 patch → 每个 patch 输出 256×256 概率图 → 直接写入对应位置 → 边界重叠区域取平均
  优点: 每个像素独立预测，边界平滑
```

---

## 路线二：标签数据准备 —— 栅格化滑坡点

### 输入

- 滑坡点矢量数据（shapefile / GeoJSON / CSV 经纬度）
- 与训练 GeoTIFF 相同的 CRS 和分辨率

### 处理步骤

1. **读取滑坡点坐标**（lat, lon）
2. **转换为图像像素坐标**（利用 GeoTIFF 的 `transform`）
3. **以每个滑坡点为中心，生成缓冲区 mask**（例如半径 30m = 1 像素，或 90m = 3 像素半径圆）
4. **输出**：每个训练 patch 对应一个 `(256, 256)` 的二值 mask（滑坡=1，非滑坡=0，研究区外=忽略）

### 伪代码示例

```python
import rasterio
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import Point, box

def make_slide_mask(geotiff_path, slide_points, patch_bounds, patch_size=256):
    """
    geotiff_path: 训练数据 GeoTIFF
    slide_points: [(lon, lat), ...] 滑坡点坐标列表
    patch_bounds: 每个 patch 的地理范围
    """
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        
        masks = []
        for (x_min, y_max, x_max, y_min) in patch_bounds:  # 每个 patch 的地理包围盒
            # 筛选落在该 patch 内的滑坡点
            patch_points = [
                Point(lon, lat) for lon, lat in slide_points
                if x_min <= lon <= x_max and y_min <= lat <= y_max
            ]
            
            if not patch_points:
                mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
            else:
                # 栅格化滑坡点为 patch 内的像素 mask
                shapes = [(p, 1) for p in patch_points]
                mask = rasterize(
                    shapes,
                    out_shape=(patch_size, patch_size),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
            masks.append(mask)
    
    return masks
```

### 标签格式变化

```
当前:  label.npy shape=(1,), dtype=int64, 值=0 或 1
目标:  label.npy shape=(256, 256), dtype=uint8, 值: 0=非滑坡, 1=滑坡, 255=忽略(研究区外)
```

---

## 路线三：硬件方案 —— 30m 全分辨率

### 方案对比

| 方案 | 内存 | 能否跑 30m 全分辨率 | 成本 |
|---|---|---|---|
| AutoDL 最低配（16G 显存 T4 / 32G RAM） | 32GB | 能（7901×13568 ≈ 2.1GB × 5 通道 = 10.5GB，余量足够） | ~0.5 元/小时 |
| AutoDL 中配（24G 显存 3090 / 48G RAM） | 48GB | 能，且推理更快 | ~1.5 元/小时 |
| Colab Pro+（高内存运行时） | 50GB | 能 | ~$50/月 |
| Colab 免费版（标准运行时） | 12GB | 不能（需降采样到 60m 或分块） | 免费 |

### AutoDL 使用流程

1. 注册 [AutoDL](https://www.autodl.com/)，充值 10 元
2. 租用「RTX 3090 / 24G」或「T4 / 16G」实例
3. 实例创建后获得 SSH 连接信息
4. VS Code Remote-SSH 连接 → `git clone` → 安装依赖 → 直接跑
5. 用完记得**关机**（不删实例），下次开机代码和数据都在

### 如果坚持 Colab 免费版 + 30m：分块方案

将 7901×13568 切成 16 块（4×4），逐块处理：

```
┌────┬────┬────┬────┐
│ B1 │ B2 │ B3 │ B4 │  每块 ~2000×3400
├────┼────┼────┼────┤
│ B5 │ B6 │ B7 │ B8 │  每块内存 ~ 2000×3400×5×4B = 136MB
├────┼────┼────┼────┤  远低于 Colab 12GB 限制
│ ...                     
└────┴────┴────┴────┘
```

边界预留 `patch_size` 的重叠区域，拼接时取平均。实现脚本约 30 行。

---

## 路线四：环境因子工程 —— 提升 AUC

### 当前因子（5 通道）

| 通道 | 因子 | 问题 |
|---|---|---|
| 0 | Elevation | 正常，值域 0~1663 |
| 1 | Slope | 正常 |
| 2 | Aspect | **角度值 0~359°，0°和 359° 实际上是同一方向但数值差很大** |
| 3 | TRI | 值域不稳定（max 可达 5289） |
| 4 | Curvature | 与 TRI 共线性高 |

### 建议改造

```
当前 5 通道:
  [elevation, slope, aspect, TRI, curvature]

建议 8 通道:
  [elevation, slope, sin(aspect), cos(aspect), TRI_normalized, curvature_normalized, distance_to_river, distance_to_fault]
```

| 改动 | 原因 |
|---|---|
| `aspect` → `sin(aspect)`, `cos(aspect)` | 消除角度循环断裂（0°↔360°） |
| `TRI`, `curvature` 单独归一化 | 避免被 elevation 的大数值淹没梯度 |
| 加 `distance_to_river` | 消落区必备因子 |
| 加 `distance_to_fault` | 地质结构控制因子 |

### GEE 导出注意事项

新增因子需重新在 GEE 中计算并导出，与现有 5 通道合并为一个多波段 GeoTIFF。导出 scale 保持 30m。

---

## 推荐实施顺序

```
第 0 步（立即可做）: 提交当前代码，Colab git pull 验证中文字体修复

第 1 步: 准备像素级标签
  - 栅格化滑坡点为 256×256 mask
  - 修改 dataloader 支持 mask 标签

第 2 步: 切换分割模型
  - 训练脚本改为 LandslideSegmentationModel
  - loss 改为 CrossEntropyLoss + 忽略研究区外像素

第 3 步: AutoDL 上训练 + 30m 全分辨率预测
  - 租用 16G/32G 实例
  - 训练 ~30 epoch
  - 预测 → 出图

第 4 步（可选）: 扩展环境因子
  - GEE 导出 distance_to_river, distance_to_fault
  - aspect 拆分 sin/cos
  - 重新训练
```

---

## 不修改代码即可提升的临时方案

| 方案 | 命令 | 效果 |
|---|---|---|
| 减小步长 | `--stride_factor 0.0625` | patch 密度翻倍，概率图更平滑 |
| 调低温度 | `--temperature 1.5` | 概率值分散度降低，等级区分更保守 |
| 换分等方法 | `--method quantile` | 等频分等，适合偏态分布 |
| 小 batch | `--batch_size 4` | 降低内存峰值 |

---

## 相关文件索引

| 文件 | 作用 |
|---|---|
| `src/model_segmentation.py` | 分割模型定义（`LandslideSegmentationModel` 已存在） |
| `src/dataloader.py` | 数据加载 + 归一化 |
| `predict.py` | 预测 + 出图主逻辑 |
| `src/visualization.py` | 绘图 + 中文字体 |
| `main.py` | 训练入口 |
| `src/debug_config.py` | 训练/推理超参数配置 |
| `docs/OPTIMIZATION_PATHS.md` | 之前的技术路线文档 |
