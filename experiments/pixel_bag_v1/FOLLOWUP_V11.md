# 后续实验 v1.1：按顺序验证稳定性、区域泛化与贡献来源

## 本次改变及保持项

修改仅在 `experiments/pixel_bag_v1/`。首轮 `run.py` 的原始字节保存在 `run_v1_frozen.py`，SHA256 应为 `14f622a6955ffe36ebd86e0f57e152df9354e67f7800bc2a902a810fa700a31b`。它作为历史复核快照，不作为新的默认入口。

原5臂的网络构造顺序、初始化、前向计算、优化器、损失、袋采样、训练轮数/树数/验证采样不变。数据缓存格式仍是 `pixel_bag_v1.0`，旧的 prepared 目录、原始像元袋和 seed42 结果继续复用。**无需重做 prepare、无需重跑 seed42 的原5臂。**

新增3个控制臂：

| 名称 | 定义 | 配对目的 |
|---|---|---|
| table_mlp | 将池化表示固定为零，保留 MIL 同样的两条表格预测路径及空袋回退；不读取像元值或局部水位条件 | mil/concat 相对神经表格基线的增量 |
| mean_mil | 地形局部编码不变，将注意力替换为固定面积加权均值 | 普通 mil 中注意力的作用 |
| concat_mean_mil | 地形+水位直接拼接的局部编码不变，将注意力替换为固定面积加权均值 | concat_mil 中注意力的作用 |

table_mlp 仍有所有臂共同的像元可用性/数量变量，不能称完全不使用像元元信息；没有使用像元地形值。均值臂保留但不使用注意力模块的初始化，以保持对应模型初始化顺序一致；不应把未参与前向的参数算成有效模型容量。

## 固定实验设置

空间/行政5折；模型种子42/43/44；fold-seed42；sampling-seed731；soft λ0.2、4km；300棵树；12轮；袋64；hidden32；batch128；学习率0.001；验证袋重复3次。线程默认4，设备默认cuda。

这是基于首轮探索结果确定的后续验证计划，不称为首轮实验之前的预注册。代码不根据验证成绩选择模型/轮数/阈值；保留所有臂与所有种子。

阶段1和空间控制臂将重新计算的折逐单元与首轮 `folds.csv` 比对；不一致立即停止。行政控制臂与阶段2的行政折比对。不同分折不能混合汇总。

## 上传和检查

上传 **`dist/pixel_bag_v11_patch.zip`** 到服务器项目目录；本包只更新 experiments，不含主线文件和数据。

```bash
source /3250707052/.venv-mil/bin/activate
cd /3250707052/landslide/mil_prepare
unzip -o pixel_bag_v11_patch.zip
python experiments/pixel_bag_v1/verify_sources.py
python experiments/pixel_bag_v1/test_pipeline.py
```

必须保留：

```text
results/pixel_bag_v1/prepared/
results/pixel_bag_v1/spatial_s42_e12/
```

第二个目录是已完成的 seed42 五臂完整结果，不能指向 smoke。目录变动时，所有命令通过 `--reference <实际完整结果目录>` 和 `--prepared <实际缓存目录>` 显式指定。

先查看计划，不启动训练：

```bash
python experiments/pixel_bag_v1/followup.py run --stage all --dry-run
```

## 阶段1：补种子43、44

```bash
python -u experiments/pixel_bag_v1/followup.py run --stage 1
```

运行原五臂的两个新种子，5折。输出：

```text
results/pixel_bag_v1/followup_v11/spatial_s43_s44/
results/pixel_bag_v1/followup_v11/logs/spatial_s43_s44.log
```

首轮 seed42 不会被覆盖或重算。命令验证参考结果已完整完成，且参数、缓存指纹与本轮相符。

## 阶段2：行政整县验证

```bash
python -u experiments/pixel_bag_v1/followup.py run --stage 2
```

原五臂，种子42/43/44，行政5折，输出 `followup_v11/admin_s42_s43_s44/`。开始前要求阶段1完成。不会根据阶段1成绩自动删除任何臂。

## 阶段3：归因对照

```bash
python -u experiments/pixel_bag_v1/followup.py run --stage 3
```

新增3臂分别运行空间与行政5折，各3种子。输出 `spatial_controls_s42_s43_s44/` 与 `admin_controls_s42_s43_s44/`。阶段1、2必须完成；原5臂结果复用。

也可一次严格按顺序执行以上所有阶段：

```bash
python -u experiments/pixel_bag_v1/followup.py run --stage all
```

运行不使用新依赖、不修改缓存。成功完成的目录会验证配置与每个臂/种子的完整人口后跳过；未完成目录不覆盖、不删除，出现失败先保留日志处理。租用实例释放前保存日志与权重，勿只保存汇总数字。

查看正在运行阶段的进度（另一个终端）：

```bash
tail -f results/pixel_bag_v1/followup_v11/logs/spatial_s43_s44.log
```

后两阶段将日志名改为对应目录名。

## 汇总与返回

阶段1后即可导出部分结果；完成全部阶段后再做一次新的完整汇总：

```bash
python experiments/pixel_bag_v1/followup.py summarize
python experiments/pixel_bag_v1/followup.py export \
  --out pixel_bag_followup_v11_return.zip
```

**建议全部完成后只汇总一次。** 汇总目录已存在时不覆盖；若已经在阶段1后汇总，需要新的输出目录，可用下面的底层汇总器分别生成报告：

```bash
python experiments/pixel_bag_v1/followup_summary.py \
  --runs results/pixel_bag_v1/spatial_s42_e12 \
         results/pixel_bag_v1/followup_v11/spatial_s43_s44 \
         results/pixel_bag_v1/followup_v11/spatial_controls_s42_s43_s44 \
  --out results/pixel_bag_v1/followup_v11/reports/spatial_complete
```

行政汇总的 `--runs` 使用两份 admin 目录。导出器会包含 reports 下所有 CSV/JSON。

下载返回包放到本地项目根目录后交给助手复核。默认含首轮参考预测、已完成阶段预测、参数、逐折/种子指标、日志、来源与代码快照；不重复传原始袋、大缓存或权重。不同于首轮扁平 ZIP，本包使用 `runs/<阶段>/` 结构。

## 新指标文件怎样读

- `seed_metrics.csv`：每个训练种子的合并OOF指标。
- `seed_variability.csv`：上述指标的种子均值、标准差与数量。只有1个种子时标准差为空。
- `fold_metrics.csv`：逐折逐种子指标。
- `macro_metrics.csv`：每种子逐折等权平均，必须与合并OOF同时读。
- `ensemble_metrics.csv`：先对列出的种子预测取平均，再评价一次；与种子指标均值不同。
- `paired_seed_differences.csv`：同种子的配对差值、宏平均AUC差和AUC提高的折数。
- `paired_seed_variability.csv`：配对差值的种子均值/标准差。

默认 `summary.json` 仍保留旧兼容语义：多个种子概率平均后的集成结果。不要把它当成种子平均性能。

汇总器拒绝不完整OOF、重复单元/种子/臂、不同缓存、不同环境版本、不同超参数或不相同的折。不同种子不代表新数据集，不把15个种子×折当作15份独立样本输出p值。

重点报告消落带内逐折AUC与合并OOF、全域/带内固定面积覆盖，并检查区域失败。多种子/行政折仍不能稳定超过基线时，应保留阴性结果；不要仅凭集成AUC或最佳种子宣称创新成立。
