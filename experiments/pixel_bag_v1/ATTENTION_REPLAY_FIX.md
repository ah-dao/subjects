# 注意力导出：修正原 OOF 推理复现方式

用户返回 unit8250、空间 fold0、seed42、mil 的复现误差：

- 已保存概率：0.1568111032247543
- 旧诊断脚本：0.15677919487158457
- 绝对差：0.00003190835316973，大于原检查阈值 0.00002

旧脚本在 CPU 上逐单元前向，而训练脚本在 GPU 上以原测试集顺序每批 128 单元计算。设备、批次形状会影响浮点结果，因此这是一项明确的复现方式缺陷；是否完全解释该服务器误差，需要修正版在真实检查点上验证。此外，旧脚本使用 Python 双精度概率求均值，而原评估先对 float32 概率向量除以采样次数，再加至 float64 累加器。

修正：

1. 增加 `--device auto/cuda/cpu`，auto 优先使用 GPU。
2. 使用保存的 folds.csv 和 prepared 单元顺序恢复原批次；包含未选为解释案例的其他测试单元，保持最后一个不足批次的大小。
3. 注意力导出与原概率复现均来自该原批次，不再用逐单元前向充当原评估结果。
4. 保持 float32 除法后再累加的顺序。
5. 保存设备、CUDA、TF32 配置等运行信息；若仍失败，写出 `reproduction_failure.json` 后停止，不放宽 2e-5 阈值。

完整袋解释仍是额外诊断，使用实际完整袋大小，其概率不替代原 OOF 概率。

## 安装并重新运行

将 `pixel_bag_attention_replay_fix.zip` 上传至服务器项目根目录：

```bash
cd /3250707052/landslide/mil_prepare
unzip -o pixel_bag_attention_replay_fix.zip -d .
/3250707052/.venv-mil/bin/python -u \
  experiments/pixel_bag_v1/export_attention.py \
  --device cuda \
  --out "results/pixel_bag_v1/attention_replay_$(date +%Y%m%d_%H%M%S)"
```

这次须用 `unzip -o` 更新旧诊断脚本；`-n` 会跳过同名旧文件。压缩包不包含 run.py、数据、权重或训练代码，不重新训练，不删除已有输出。

启动应打印 `Replay runtime:` 并显示 `device: cuda`。成功时打印 `DONE:`，下载相应 ZIP。若仍出现复现错误，将新输出目录内的 `reproduction_failure.json` 返回本地项目根目录，再据此排查；不要手动调大阈值。

本地两个合成数据测试通过，包括原始批次顺序和尾批、两种划分/三种子/两种模型的完整导出、坐标与注意力归一化以及故意错误的预测拒绝。本地没有 CUDA，仅完成 CPU 测试，服务器真实检查点复现尚待运行。
