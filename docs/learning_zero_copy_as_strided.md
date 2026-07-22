# 零拷贝切片：as_strided 学习笔记

## 1. 问题背景

在 `predict.py` 中，需要将一张大尺寸 GeoTIFF（如 5000×6000）切成 N 个 256×256 的 patch，送入模型逐 patch 推理。

**传统做法**（双重循环）：

```python
patches = []
for i in range(h_steps):
    for j in range(w_steps):
        patch = data[:, i*stride:i*stride+256, j*stride:j*stride+256].copy()
        patches.append(patch)
```

- patch_size=256, stride=32, 5000×6000 图像 → 约 **22000 个 patch**
- 每个 patch `5×256×256×4byte = 1.25MB`
- 总共 **~27GB 内存** → Colab 12GB 直接崩溃

## 2. as_strided 核心思想

**不复制数据，只换一种"看"数据的方式。**

用书架比喻：
- **传统方法** = 把一本书复印 22000 份，每份剪下一个小方块 → 内存爆炸
- **as_strided** = 用手指在原书上比划 22000 个框，要用哪个才复印哪个 → 零内存

## 3. 理解 strides

### 3.1 什么是 strides

numpy 数组在内存中是一段**连续的字节**。`strides` 告诉你"沿着每个维度走一步，需要跳过多少字节"。

```python
data.shape  = (5, 5000, 6000)    # (通道, 行, 列)
data.strides = (120000000, 24000, 4)
#                ↑通道       ↑行    ↑列
#                跳到下一通道   跳到下一行 跳到下一列
#                要跳过         要跳过      要跳过
#                5000×6000×4B  6000×4B     4B(1个float32)
```

### 3.2 as_strided 的原理

`as_strided(data, shape, strides)` 给 numpy 一个**新形状 + 新 strides**，让 numpy 用不同的方式解读同一块内存。

## 4. predict.py 中的实际用法

### 4.1 小例子演示

假设一张 3×3 图，1 通道，patch_size=2，stride=1：

```
原始数据:
┌──────────┐
│ 1  2  3  │  shape = (1, 3, 3)
│ 4  5  6  │  strides = (36, 12, 4)
│ 7  8  9  │
└──────────┘
```

要从 3×3 中切出所有 2×2 的 patch，stride=1：

```
patch[0,0] = ┌───┐   patch[0,1] = ┌───┐
             │1 2│                │2 3│
             │4 5│                │5 6│
             └───┘                └───┘

patch[1,0] = ┌───┐   patch[1,1] = ┌───┐
             │4 5│                │5 6│
             │7 8│                │8 9│
             └───┘                └───┘
```

### 4.2 构建新 shape

```python
h_steps = (3 - 2) // 1 + 1 = 2    # 可以取2行patch
w_steps = (3 - 2) // 1 + 1 = 2    # 可以取2列patch

patch_view_shape = (h_steps, w_steps, channels, patch_size, patch_size)
                 = (2, 2, 1, 2, 2)
```

### 4.3 构建新 strides（这是关键）

```python
patch_view_strides = (
    stride * data.strides[1],   # 维度0 (h_steps): 跳 stride 行
    stride * data.strides[2],   # 维度1 (w_steps): 跳 stride 列
    data.strides[0],            # 维度2 (通道): 不变
    data.strides[1],            # 维度3 (patch内行): 不变
    data.strides[2],            # 维度4 (patch内列): 不变
)
# = (1×12, 1×4, 36, 12, 4)
# = (12, 4, 36, 12, 4)
```

#### 解读每个 stride

| 维度 | stride 值 | 含义 |
|------|----------|------|
| **h_steps** | `stride × data.strides[1]` = 12 | `all_patches[i+1,j]` 比 `all_patches[i,j]` 在原数据中往下偏移 `stride`(1) 行，内存中跳过 1 行 = 12 字节 |
| **w_steps** | `stride × data.strides[2]` = 4 | `all_patches[i,j+1]` 比 `all_patches[i,j]` 在原数据中往右偏移 `stride`(1) 列，内存中跳过 1 列 = 4 字节 |
| **通道** | `data.strides[0]` = 36 | patch 内跨通道和原数据一致 |
| **patch内行** | `data.strides[1]` = 12 | patch 内跨行和原数据一致 |
| **patch内列** | `data.strides[2]` = 4 | patch 内跨列和原数据一致 |

#### 可视化

```
访问 all_patches[0, 0, 0, 0, 0]:
  = data[0, 0] → 1  ← 左上角

访问 all_patches[0, 1, 0, 0, 0]:
  = data[0, 1] → 2  ← 第1列patch从原图第1列开始

访问 all_patches[1, 0, 0, 0, 0]:
  = data[1, 0] → 4  ← 第1行patch从原图第1行开始
```

### 4.4 展开为 (N, C, H, W)

```python
all_patches = as_strided(data, shape=patch_view_shape, strides=patch_view_strides)
# shape: (h_steps, w_steps, C, H, W)

all_patches = all_patches.reshape(-1, channels, patch_size, patch_size)
# shape: (N, C, H, W) ← 方便用 indices 直接索引
```

## 5. 为什么能"零拷贝"

`all_patches[0,0]`、`all_patches[0,1]` 等所有 patch **共享 data 的同一块内存**，只是起始位置和读取范围不同。

传统的 `view` 操作（reshape、transpose、slice）返回的也是视图，但**只能表示连续的、规则的范围**。`as_strided` 的威力在于可以表示**任意间距的滑窗**——通过修改 strides 中的第一步来实现"偏移 stride 行/列"。

## 6. 零拷贝的代价和注意事项

### 6.1 不是"免费"的

- `as_strided` **不会检查越界**：如果 strides 设置不当，可能读到内存中不属于原数组的数据（缓冲区溢出风险）
- 需要使用者保证 shape 和 strides 的合法性

### 6.2 使用时必须 .copy()

```python
# predict.py 第 163 行
batch_patches = all_patches[indices].copy()  # 送入模型前必须拷贝
```

原因：
- 后续要对 patch 做归一化（原地修改数据）
- 多个 patch 共享内存，改一个会污染其他 patch
- 所以只在真正需要时才 `.copy()` 一份独立数据

### 6.3 性能对比

| 方法 | 内存占用 | 时间 |
|------|---------|------|
| 双重循环 | ~27GB (22000个独立数组) | ~11 分钟 |
| as_strided | ~0 (零拷贝视图) | < 1 秒 |

## 7. 总结

> **传统方法**：把整本书复印 22000 份，每份剪下一个小方块 → 内存爆炸
>
> **as_strided**：用手指在原书上比划 22000 个框，要用哪个才复印哪个 → 零内存

核心公式：

```python
patch_view_strides = (
    stride * data.strides[1],   # 相邻patch沿行跳 stride 行
    stride * data.strides[2],   # 相邻patch沿列跳 stride 列
    data.strides[0],            # 通道不变
    data.strides[1],            # patch内行不变
    data.strides[2],            # patch内列不变
)
```

前两维是"哪一片 patch"，后三维是"patch 内部怎么读"。patch 内部的读取方式跟原数据毫无区别，唯一的变化就是不同 patch 之间在原数据中的偏移量。
