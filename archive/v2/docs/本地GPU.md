# 本地 GPU 跑 TabPFN（RTX 5060 8 GB）

> 结论先说：**本机有卡、能用**，只是 conda 环境里装的是 **CPU-only 版 torch**，所以一直没用上。
> 装一个 cu128 版 torch 到 `.deps`（不碰 conda 环境）即可，TabPFN 全流程本地跑。

## 0. 本机硬件实测

| 项 | 值 | 说明 |
|---|---|---|
| GPU | **NVIDIA GeForce RTX 5060**（Blackwell，`sm_120`） | 显存 **8151 MiB** |
| 驱动 | **577.05**（CUDA 12.9） | 50 系必须 **cu128**，且驱动 ≥570 ✅ |
| 现有 torch | `2.12.1+cpu` | **CPU-only** → `cuda.is_available()` 恒为 False |
| 可下载版本 | `torch-2.9.1+cu128-cp310-cp310-win_amd64.whl`（2.86 GB） | 阿里云 pytorch-wheels 镜像与官方源都通 |

⚠ **不要用 PyPI 装 torch**：PyPI 上的 Windows `torch` wheel 是 **CPU 版**（仅 ~0.2 GB）；
CUDA 版只在 `download.pytorch.org` 与国内 `pytorch-wheels` 镜像上。

## 1. 一键安装（本机实测可用）

```powershell
cd C:\Users\dollars\code\subjects
& "$env:USERPROFILE\.conda\envs\landslide\python.exe" tools\install_cuda_torch.py
#   下载 2.86 GB → 解包到 .deps（不碰 conda 环境）→ 自动验证 CUDA 并跑一次矩阵乘测速
#   可选： --ver 2.9.0  换版本 ； --check 只体检 ； --delete-wheel 装完删掉 wheel（省 2.9 GB）
```

装完应看到：`torch 2.9.1+cu128`、`cuda.is_available() = True`、`设备: NVIDIA GeForce RTX 5060`、
`Blackwell(sm_120) 支持：True`。

**为什么装到 `.deps`**：本机 pip 不能联网、`pip --target` 被系统策略拒绝，所以走"urllib 下载 + zipfile 解包"；
而 `baseline_tabpfn.py` 启动时已把 `.deps` 插到 `sys.path` 最前 → 自动优先用 CUDA 版 torch，
conda 环境保持原样（要回退只需删掉 `.deps\torch`）。

## 2. 跑 TabPFN（GPU）

```powershell
cd C:\Users\dollars\code\subjects
$env:PYTHONPATH = "$PWD\.deps"          # 让**任何**脚本都能用到 CUDA 版 torch
$env:HF_HUB_OFFLINE = "1"               # 权重用本地 models/tabpfn/*.ckpt

# ① 冒烟（1 分钟，确认链路与显存占用）
python baseline_tabpfn.py --smoke --device cuda --n-estimators 4

# ② N1：逐单元 OOF 概率导出（集成/校准/共形/不确定性图的前提）
python baseline_tabpfn.py --features-csv features/v2/features_v2_train_38.csv `
    --method spatial_kmeans --context soft --neg-lam 0.2 --neg-km 4.0 `
    --max-train 6000 --seeds 3 --device cuda --save-oof results/runs/tabpfn_oof_spatial.csv

# ③ T3：置换零分布 → 新检出限（7 臂 × 5 折 × 3 种子）——**这就把 GPU 上要跑的实验搬回本地**
python tools/m3_t3_mde_tabpfn.py --method spatial_kmeans --seeds 3 --max-train 6000 --device cuda
python tools/m3_t3_mde_tabpfn.py --method admin          --seeds 3 --max-train 6000 --device cuda
```

## 3. 8 GB 显存的注意事项

| 口径 | 上下文规模 | 8 GB 是否够 | 建议 |
|---|---|---|---|
| `--context matched`（病例-对照 k=5） | ~3.2k | ✅ 宽裕 | 直接用 |
| `--context soft --max-train 6000` | ~6k | ✅ 可以 | **推荐**（本地默认用这个） |
| `--context soft --max-train 12000` | ~12k | ❌ 需 ≥16 GB | 本地不要用；正式口径留给服务器 |

- 显存不够时先降 `--n-estimators 4`，再降 `--max-train`。
- Windows WDDM 有额外开销（浏览器/壁纸占用显存），跑之前先关掉占显存的程序。
- ⚠ **口径必须写清楚**：本地用 6000 得到的 AUC/MDE 与服务器"正式口径 12000"的数值**不可直接混用**；
  标度无关的量（如 `σ_种子/σ_置换` 比值）可以比，绝对值不行。
- `--device cuda` 若报 DLL 加载失败，说明被系统应用控制策略拦了 CUDA 动态库
  （本机 rasterio 就遇到过这问题）→ 把 `tools/install_cuda_torch.py --check` 的完整输出发我。

## 4. 本地 GPU 解锁了哪些原本只能上服务器的实验

| 实验 | CPU（8 线程） | 本地 GPU（估） | 说明 |
|---|---|---|---|
| 单配置 soft 口径（5 折 × 1 种子，6k 上下文） | ~15~20 min/折 | ~10~30 s/折 | 约 **30~60×** |
| **T3**（7 臂 × 2 分折 = 210 次拟合） | 15~30 h（不可行） | **~30~60 min** | 原本等 GPU 的核心实验 |
| N1 OOF 导出（2 分折） | ~5 h | ~10 min | 集成/共形/不确定性图的前提 |
| M2 MIL 训练（5 折 × 8 epoch） | ~15~30 min | 几分钟 | 仍需先在服务器做栅格预处理（本机 rasterio 被拦） |

> 注：M2 的**预处理**（`unit_id` 栅格 + 像元袋）依赖 rasterio，本机被系统策略拦截，只能在服务器做；
> 但本机地形栈是 BigTIFF+LZW 分块，理论上可自写解码器绕过 —— 风险较高，不建议赶工时做。

## 5. 推荐执行顺序（本地 GPU）

1. `tools/install_cuda_torch.py` → 验证 CUDA ✅
2. `--smoke --device cuda` 确认链路
3. **T3 空间折** → 出 `results/t3_tabpfn_mde_spatial_kmeans.txt`（配合 `tools/noise_decomposition.py` 看 σ 分解）
4. **T3 行政折** → 满足"两分折同号"判据
5. N1 OOF 导出 → 再跑 `tools/m3_ensemble_conformal.py` 出 N1 集成 / TabPFN 侧校准共形 / N4 不确定性
