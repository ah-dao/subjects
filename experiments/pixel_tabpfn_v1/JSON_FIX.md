# 完整CV的NumPy整数记录修复

报错位置为fit_tabular写fit.json。未指定--only-fold时，外层fold来自NumPy整数数组；Python种子加NumPy fold得到numpy.int64，标准json.dumps不能序列化。原测试只使用显式only-fold，未覆盖这个分支。

修复：

- 完整CV及指定单折的outer_folds统一转为Python int。
- fit_tabular入口将模型种子转为Python int，避免序列化及下游随机种子API的类型问题。
- bag_core.write_json支持NumPy标量和数组，保留JSON数值/布尔类型；仍拒绝NaN/Infinity和不支持对象。
- 回归测试增加实际完整4折合成数据分支，核对每折fit.json种子和最终完整覆盖，而非只测试单折。

源码清单随补丁更新。此修复不改变已有4k/2估计器/预测批次32的设置，不重建缓存、不改折位。仅改变元数据类型规范化；不保证与不同CUDA环境结果逐位一致。

在包含experiments/和splits/的项目根目录解压覆盖本补丁，执行package.py verify和test_pipeline.py。已失败运行的当前模型虽完成过预测，但尚未完整保存，入口不支持恢复该次拟合；保留失败目录并换新out重跑。日志显示该次只进入OUTER0，并非已经完成全部实验。

测试使用CPU合成数据和TabPFN调用替身，不是服务器真实TabPFN验证。
