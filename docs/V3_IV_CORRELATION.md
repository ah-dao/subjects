# V3信息量与相关性分析

独立CPU流程，使用现有V3特征与seed42训练上下文。无需GPU、Torch/TabPFN导入或新增GIS/绘图依赖，不读取/修改正在运行的置换结果，也不重新训练模型。需要先前全部84维解释补丁中的explain84_design与explain_schema。新增文件，不覆盖冻结模型、特征、旧解释代码。

## 分析口径

- 每个空间/行政外折分别使用完整四折训练池与一折测试人口，84维全覆盖。430个河道规则单元不参加统计；每套测试人口为25509单元、661清单正单元。
- 分箱、粗地形层边界只在该折原seed42的4000训练上下文确定，复用逐维解释规则；信息量计数使用完整四折训练池，避免上下文正例富集造成密度偏差。
- 全体、县域、粗地形层、水位背景分别计算。FR=(分箱清单正率)/(同层总体清单正率)，IV=ln(FR)，按单元数量而非面积。正值代表相对集中，不是因果效应。
- 明确分开train_pool训练关联与heldout测试描述。测试标签不参与训练信息量或分箱拟合；本轮不将IV作为新模型输入、特征选择或融合权重。训练池重叠，不叠加五折训练计数；各折边界不同，主表保留逐折结果。
- 原始IV零正例为负无穷，以状态列标注而非将其冒充0；平滑IV为ln(((p+0.5)/(n+1))/((P+0.5)/(N+1)))。空组和单类别层不估计。缺失单列，0是实际取值；训练常量、新水平、无训练支持明确标记。小样本标记不是显著性检验。
- 相关性仅用原始有限值，逐变量对剔除缺失，不使用训练填补值、不混入标签或模型评分。输出训练/测试84×84 Spearman矩阵、各对有效样本数、全部变量对和绝对值>=0.9清单；常量/不足有效对为NaN。OOF矩阵仅用于描述，不据此事后删特征。
- 信息量范围不是可比较的模型重要性，尤其不能按最大IV直接排名。与置换AP/C20/AUC、效应曲线及相关冗余一起解释，保留样本支持、县域差异和非单调情况。

## Git同步与运行

本地PowerShell上传（沿用已有主机地址、端口与V3目录，变化时替换）：

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

服务器上，待当前任务完成后运行：

```bash
cd /3250707052/subjects-git
source /3250707052/.venv-tabpfn-cu124/bin/activate
mkdir -p logs
nohup bash scripts/run_v3_iv_correlation_s42.sh > logs/v3_iv_correlation_s42.log 2>&1 < /dev/null &
```

默认输出results/v3_iv_correlation_s42，完成后自动校验打包为results/v3_iv_correlation_s42_return.zip。不允许覆盖已有输出；失败后保留旧目录，手动run指定新的--out，不删除训练结果。本CPU流程不做折级断点续跑。

若计算完成但自动打包未完成，可手动导出（ZIP存在时拒绝覆盖）：

```bash
python tools/v3_iv_correlation.py export --out results/v3_iv_correlation_s42
```

本地PowerShell下载：

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

## 文件

information_value_detail.csv：每折/训练或测试/全部84维/分层/分箱的FR、原始与平滑IV、支持与缺失状态。

training_context_bin_plans.csv：训练侧分箱与地形边界。

feature_support.csv：每折各变量有限/缺失/不同值数量。

correlation_pairs_all.csv、correlation_pairs_abs_ge_0p9.csv：逐折变量对结果及高相关清单。

spatial_fold0等目录：各折训练/测试相关矩阵及有效对数矩阵。

spatial/admin_oof_descriptive_*：完整测试人口拼接的描述性矩阵。

information_value_support_summary.csv：逐折IV范围及支持检查，不是变量重要性排名。

config.json、manifest.json：输入、代码、训练上下文指纹及完整输出校验。
