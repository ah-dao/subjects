# 明确以保存文件加载出的模型为解释对象

这是当前V3逐维解释的唯一运行入口。解释方法与边界见[V3全部84维解释](V3_ALL84_INTERPRETATION.md)，正式出图见[V3底图图集](V3_BASEMAP_ATLAS.md)。旧refit、saved、seeded和reference-check过程指南已删除，相关源码与模型恢复依赖保留。本文记录方案与操作方法，不代表服务器已经完成全部模型核验或解释。

## 真实诊断结果

2026-10-22的加载失败诊断：7357单元仅47个误差>2e-5；平均绝对差2.217156e-6，最大0.001704574；两次加载后预测（同实例）差0。AUC保存前/加载后0.923568890/0.923571377，AP0.146355611/0.146357679，20%面积捕获均为97/111。评分Spearman约0.999999660。不能因此宣称所有折序列化无损，也没有证实具体底层误差来源。当前自动混合精度为auto/use_autocast=true，仅凭该标志不能认定混合精度就是原因。

旧流程把保存前对象的逐点评分作为加载后对象的严格门槛；当前误差阻断了它。此版不放大2e-5门槛，而是明确更换解释对象和参照基线：以官方保存文件每次加载出的模型定义固定解释模型。保存前、历史原始评分都完整保留作对照，不把加载对象说成原对象无损复现。

## 新基线的严格校验

1. 首先恢复原saved目录已有的.tabpfn_fit。旧模型字节与旧参考不改，新目录有明确来源指纹。
2. 在相同环境中加载两次独立模型，每次完整测试预测两次；各自重复和两个加载实例间均须<=2e-5。顺序加载，只有一个实例驻留GPU。
3. 独立加载一致后生成loaded_baseline/predictions.npz和CANONICAL.json，记录真实加载基线与模型/配置绑定；roundtrip_difference.json单独记录保存前后差异。原fitted_model/reference.npz不修改。
4. 新Python进程verify-existing再次加载已恢复模型并严格匹配加载基线，前两阶段绝不fit。通过后才准备其余未运行折。每个新折仍保留原两次拟合稳定性检查，保存后建立同样的加载基线。
5. 所有模型完成后，新进程verify-models验证10个加载模型；PFI与ALE用同一个加载实例；续跑仍加载保存模型并核对加载基线，不能重建/刷新基线掩盖后续差异。

如果两个独立加载或跨进程重载不一致，立即停止。不是对所有误差开绿灯：保存前后差异不再被冒称精确复现，独立加载模型自身复现仍严格2e-5。解释报告为新版本，不能覆盖旧地图、旧评价或混用旧模型贡献。

legacy_vs_refit_fullfold_metrics.csv同时列历史参考、保存前拟合、canonical加载模型三个对象的逐折AUC/AP/C20。全部结果回来后先对照这个表；目前只检查空间fold0，不能推定剩余折同样接近。

## 运行

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

```bash
cd /3250707052/subjects-git
source /3250707052/.venv-tabpfn-cu124/bin/activate
mkdir -p logs
nohup bash scripts/run_v3.sh canonical > logs/v3_explain84_canonical_s42.log 2>&1 < /dev/null &
```

恢复源是第一次保存模型的results/v3_explain84_saved_s42，不是seeded失败目录。若原任务自定义目录，需要修改串行器recover-from，不能删除原结果。新结果目录为results/v3_explain84_canonical_s42。

```bash
tail -n 40 logs/v3_explain84_canonical_s42.log
```

先通过恢复和独立进程verify-existing，才进入未运行模型的prepare；两阶段通过之前没有新拟合。随后做84维全局/地形层内置换及效应分析。Git入口不自动导出，需要时手动执行`python -m src.v3.explain84_canonical export --out results/v3_explain84_canonical_s42`。完整结果含模型状态；基础.ckpt仍须另行保留。

成功后PowerShell下载：

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

若仍在独立加载校验中失败，自动输出小诊断包，不再尝试放宽门槛：

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

本地审计/作图：

```bash
python -m src.v3.explain84_canonical review --archive v3_explain84_canonical_s42_return.zip --out results/v3_received_explain84_canonical_s42
python tools/analyze_v3_explain84_canonical.py
```

CPU替身测试覆盖完整流程、独立进程、旧文件不变与无fit恢复，并明确测试“保存前后有固定差异但独立加载一致”以及“后续加载不同必须拒绝”。这不代替真实GPU验证，因此服务器先校验现有模型再决定能否继续。
