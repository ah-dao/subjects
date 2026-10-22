# 失效分析与注意力导出

这次新增诊断脚本，不改 run.py、数据准备、训练设置或已保存模型，不重新训练。

## 已在本地完成

`failure_analysis.py` 对两种划分下全部单元的 OOF 预测做分组诊断，连接站点、县域、38 维特征和像元数量；输入指纹与服务器缓存清单一致。结果在 `results/pixel_bag_v1/failure_attention_20261015/`。

注意：分组主表采用三种子概率集成后评价，不能和三种子各自指标的均值混用。面积预算命中以全区统一排序为准；没有在小分组内重新分配预算。站点、区域等差异只是描述性证据，不能作为病因判断。

## 服务器操作

把 `dist/pixel_bag_diagnostics_patch.zip` 上传到服务器项目根目录。压缩包只包含本说明、新增导出脚本和 30 个案例列表；依赖服务器已有的原版 `experiments/pixel_bag_v1/run.py`。

```bash
cd /3250707052/landslide/mil_prepare
unzip -n pixel_bag_diagnostics_patch.zip -d .
/3250707052/.venv-mil/bin/python -u experiments/pixel_bag_v1/export_attention.py --device cuda
```

沿用现有 Python 环境，不新增软件包。使用 GPU 推理，不重新训练；耗时取决于磁盘读取和案例袋大小。修正版按原测试人口顺序恢复原始批次，包含最后一个不足批次，并保持原概率平均的浮点运算顺序。默认 `--device auto` 在 CUDA 可用时使用 GPU；为了与此次服务器训练一致，上述命令显式指定 CUDA。

如果已安装旧诊断脚本，使用 `pixel_bag_attention_replay_fix.zip`，执行 `unzip -o pixel_bag_attention_replay_fix.zip -d .` 更新。这里需要 `-o`，否则此前的 `-n` 会跳过旧脚本，修复不会生效。修复包只更新诊断脚本及其说明，不含训练代码、权重或数据。

默认读取以下原文件；如果之前改过目录，通过对应选项传入真实路径：

| 文件 | 参数 |
| --- | --- |
| `results/pixel_bag_v1/prepared` | `--prepared` |
| `features/v2/unit_bags.npz` | `--bags` |
| `results/pixel_bag_v1/spatial_s42_e12` | `--reference` |
| `results/pixel_bag_v1/followup_v11` | `--followup-root` |

模型目录中必须保留 `fold{fold}_seed{seed}_{arm}/weights.pt` 及 `scaler.npz`。本次读取地形 MIL 和 concat_mil，空间与按县两种划分，seed42/43/44；不读取 XGBoost 权重。

如果模型或参数文件缺失，先确认是否移动或清理，不要为导出解释直接重新训练。输出目录若已存在，脚本拒绝覆盖；需要再次导出时使用一个新的 `--out` 路径。

## 导出内容与核验

1. 从原始袋恢复像元行列号，对齐缓存像元顺序并逐案例核验地形特征。
2. 只读取该单元所属外层测试折的模型，防止用训练内预测解释测试性能。
3. 在 GPU 上按原测试批次复现 K=64、3 次采样的 OOF 概率，与保存预测对照。绝对差超过 2e-5 立即停止，不输出“成功”的 ZIP；同时写出 `reproduction_failure.json`，记录环境、原始批次、误差及模型指纹，便于进一步定位。阈值未放宽。
4. 导出三次原始采样中实际使用的像元及注意力；重复采样的像元以不同抽样位置保留。
5. 额外对完整像元袋计算一次注意力，便于绘图和跨种子比较。完整袋改变输入大小，属于额外诊断，其预测不能替代原始实验结果。
6. 导出注意力有效像元数、前 10% 像元的权重占比、陡坡及水位过渡区的权重与面积占比，并计算三个种子间的 Spearman 相关和前 10% 像元 Jaccard。相关可能部分来自共同的面积权重先验。

脚本在本地用合成短袋/长袋、打乱的原始像元顺序、两种划分、三个种子的模型文件做了完整导出测试；修正版还检查原测试顺序、最后一个不足批次、真实批次大小及浮点平均顺序，同时验证故意篡改 OOF 概率会被拒绝。两个测试通过。本地仅有 CPU，尚未验证服务器真实 GPU 模型；没有在本地训练真实模型。

## 返回结果

成功后把以下 ZIP 下载到本地项目根目录：

```text
results/pixel_bag_v1/attention_diagnostics_20261015.zip
```

文件含 `case_summary.csv`、`seed_attention_stability.csv`、各案例的 `samples/*.npz` 与 `full/*.npz`、案例列表、来源配置和校验和；不包含模型权重、全部原始像元袋或缓存。

本地收到后可生成真实的注意力空间图，对照失败与改善案例。现在本地缺少权重，因此尚未生成或解释真实注意力图。

注意力只是模型内部相对权重，不是像元滑坡概率，也不直接等于因果贡献。30 个案例按预先固定规则选取：每种划分每折各一个正样本排名下降、排名上升、高分负样本；用于诊断，不能作为总体有效性的独立验证。
