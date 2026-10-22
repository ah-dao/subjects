# 本地验证范围

CPU landslide环境：numpy/pandas/sklearn/XGBoost/PyTorch。test_pipeline.py已执行合成数据端到端检查：

- 上下文只来自训练候选、保留全部训练正样本；修改外层测试标签不会改变上下文。
- 38/40/84维输入、TabPFN调用位置与同上下文XGB，以及可选soft加权XGB的完整运行、指标与ID导出。
- 真实小型MIL训练：内层3折OOF、外层重训、四臂外层预测、alpha=0回退及并列选较小alpha。
- 翻转外层测试标签后重新运行，权重选择和外层预测分数完全不变。
- 导出ZIP CRC/逐项哈希、可信本地代码复算指标与融合权重；拒绝覆盖输出及缓存篡改。

本地未安装TabPFN；测试在进程内将其替换为LogisticRegression，并明确标记版本为test-double。这只验证数据隔离、调用接口和实验编排，不验证TabPFN 8.5.0真实拟合、CUDA、显存或科学效果。生产CLI没有替身模式。真实版本、权重、构造器在服务器check中核验，首次拟合由README的smoke完成。

没有跑本项目真实数据训练；不声称融合已带来收益。旧实验的run.py与其package_manifest.json保持原样。
