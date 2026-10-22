# 原外层fold4：18次拟合；TabPFN后续位置

复用当前run.py和派生缓存，不修改第一轮参数。fold4.py是固定参数启动器，只选择原空间五折的fold4作为测试，其余四折训练。不会加载第一轮开发训练权重，而是从头训练18个模型。该测试仍有寸滩站点对应单元参与训练，不是留出整个寸滩站点。

## 上传与执行

将 dist/pixel_bag_repr_fold4_patch.zip 上传到服务器项目目录，解压仅新增 fold4.py 和本说明，不覆盖旧run.py：

```bash
source /3250707052/.venv-mil/bin/activate
cd /3250707052/landslide/mil_prepare
unzip -n pixel_bag_repr_fold4_patch.zip
python experiments/pixel_bag_repr_v1/fold4.py plan
mkdir -p logs
nohup python -u experiments/pixel_bag_repr_v1/fold4.py train \
  > logs/pixel_bag_repr_outer_fold4.log 2>&1 < /dev/null &
```

```bash
tail -n 40 logs/pixel_bag_repr_outer_fold4.log
```

旧空间运行目录不是spatial_s42_e12时，在train命令追加 `--reference 实际空间五折目录/folds.csv`。启动器先校验原训练代码/依赖指纹、25,509人口、测试3,770单元及47正样本；run.py还会重新生成五折并逐位核对reference，因此不会默默使用行政折或开发3折。保持旧package_manifest.json不变。

内部命令关键是 `--full-cv --folds 5 --frozen-folds 原折表 --only-fold 4`，**没有 --development-folds，也没有 --exclude-fold**。这里full-cv表示使用全人口建立原外层协议；only-fold=4使实际只训练/测试这一折。输出summary的complete_oof=false正常，不表示失败。

完成条件：日志出现DONE；六臂×三种子，共18条fold_metrics记录；predictions.csv有67,860行（3,770×6×3）。新目录默认outer_fold4_s42_s43_s44，存在则拒绝覆盖。若确需重试，train与export都用相同的新 --out。

结束后导出：

```bash
python experiments/pixel_bag_repr_v1/fold4.py export
```

回传pixel_bag_repr_outer_fold4_return.zip及日志。不要把它与开发3折合并冒充全人口OOF，它们训练协议不同。该折内部10/20%面积指标，不等于旧全域预算下在此折命中的比例。此处不选最佳种子或调参；是已诊断区域的固定配置复验，不是从未见过的独立测试。

## TabPFN没有被否定，应恢复为强对照

docs/当前模型方案.md §11.6、results/m3_entry_report.txt及同人群对照记录：38维soft口径中，TabPFN空间/行政AUC约0.9122/0.9163，同上下文XGB约0.9059/0.9099。历史结果支持其在该设置下的价值；matched口径没有同样明确优势。这些旧人口/折/上下文结果不能直接与本轮分数比较，也不沿用将15个共享种子—折视为独立的显著性说法。

本轮MIL已使用原38维表格加2个可用性变量（40维），与像元池化表示共同预测。当前没有TabPFN调用；保留XGB是为了控制像元表示消融，不能把击败XGB等同于击败项目最强模型。前一版下一轮规划遗漏了TabPFN，应修正。

恢复次序：

1. 先完成已确定18次拟合，不中途改变实验臂。
2. 在相同原空间折表和评估人口上补TabPFN对照：原38维复现桥梁、共同40维表格、共同40维+固定像元/水位统计量。至少对后两套输入增加同上下文XGB，保持每种子上下文行ID完全相同并保存；另保留全训练人口加权XGB。这样区分模型、输入与上下文效应。
3. 若像元表示仍有必要，再检验 `共同40维表格 + MIL池化向量 → TabPFN`。每单元一个定长向量，不把所有原始像元展开当独立标签行。

监督MIL向量不能从全人口训练的编码器导出后再随机划分。每个外层训练区内完成编码器训练、表示选择及TabPFN训练，测试区仅变换/预测。若同一训练标签既用于编码器又用于第二级分类器，存在训练/测试表示差异及过拟合风险；用训练区内部嵌套验证评估，不声称仅加一个OOF步骤即可解决。不同内折编码器的潜在坐标不自动对齐，不能随意把其embedding拼成一张OOF表。

更容易严谨实现的融合起点是预测分数融合：TabPFN(table)与MIL的两个分数，融合参数仅在外层训练区的内层OOF上确定，外层测试仅用训练区重训模型预测。分数有统一语义，避免跨编码器潜在维度对齐问题。融合也不保证优于两者。

当前服务器MIL环境是否安装TabPFN及对应权重尚未确认；本18次任务不需要安装TabPFN。后续应固定既有实验的包版本、具体模型权重/配置和上下文构造，不混用升级后的默认模型。
