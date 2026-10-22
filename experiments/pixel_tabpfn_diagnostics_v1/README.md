# 84维TabPFN：因子解释、空间诊断与初版制图

**制图分级更新（2026-10-20）：现行五级分级已改为自然断点法，使用experiments/pixel_tabpfn_mapping_v2/mapping.py及该目录README；下文原分位数描述仅留作首版历史记录。服务器解释代码保持冻结，原任务无需重启。**

本轮冻结现有84维、4k上下文与抽样，新增解释与交付流程。原训练代码与历史结果不修改。

## 已能在本地完成的部分

run.py diagnostics使用可信回传的40/48/80/84维三种子OOF预测，不训练模型。产出：

- feature_dictionary.csv：完整84维，包括原表的降水、植被、土地利用及变化、道路、水位及变化、地质土壤、地形几何和像元统计；记录单位、时间口径与解释边界。
- permutation_groups.json：11个因子/控制组和5个表示组，每个视角内部完整覆盖84列且不重复，两个视角之间重叠，贡献不能相加。
- spatial/admin/oof_units.csv：单元OOF均值、种子标准差、40/48/80维对照、前10%/20%面积筛查标记、低排名已知滑坡和高排名未标注单元。
- regional_metrics.csv：折、县、固定5km网格指标。无正样本或只有一个类别时AUC/AP留空，不伪造为0。
- context_feature_shift.csv：每折测试人口分别相对固定seed42的4k上下文、全部外折训练人口的标准化均值差与1%–99%范围外比例。用于区分候选地域差异和抽样覆盖问题，不证明退化成因。
- subgroup_metrics.csv：空像元袋、低像元数、历史淹没比例分组结果。
- observed_binned_associations.csv及图：全84列的观测分箱关系，图选12个代表变量。该图不是PDP/ALE，也不是模型干预响应或校准曲线。

县名称原文件存在乱码，显示时仅在可逆latin1→GBK解码成功时恢复，不用county_code=30000作为县分组。ID与训练分折不改变。

excluded_factor_scope.csv另列未输入84维模型的岩性硬度、坡向结构和力学参数。本轮置换不能解释这些因子的模型作用，不能据此声称无物理意义；其讨论须使用此前消融、条件关联和来源核对。

## 初版地图的口径

mapping.py用unit_id显式连接完整斜坡单元，生成两套空间/行政OOF图，每套包括Shapefile、完整CSV、90m GeoTIFF分数与等级、PNG/PDF预览和分级统计。25,509个建模单元取seed42/43/44的OOF分数均值；430个河道排除单元score=0、level=1，assignment明确注明规则赋值。25,939个单元全部有分数与等级。

五级分级在25,509个建模单元的分数上取20/40/60/80%分位点，是按单元数近似等频，不是等面积或发生概率阈值。各协议分别定阈，不能据相同等级宣称相同绝对风险。分级不使用标签寻优。矢量保留每个单元，90m栅格采用像元中心规则，极小多边形可能不落入任何栅格像元，栅格不替代矢量完整性。

原全域清单有662个正标注，其中unit_id=16490位于430个排除河道单元中。地图保留raw_label=1并特殊标记，同时按既定河道规则赋最低等级；模型评价仍只计661个正样本，不能把该单元悄悄改为负样本。

原特征包含正样本事件前、负样本匹配伪事件前的气候/用地窗口以及固定全期水位统计。图是历史回溯OOF验证图，不能称未来某时点预报，也不是重新使用全人口拟合后的最终成果图。分数不是经校准的实际发生概率。

## 一个已核实的字段含义问题

max_drawdown_rate沿用原模型数值，但原生成脚本tills/v2_build_inundation_variants.py使用abs(diff(月均水位)).max()，实际包含上涨和下降，并不是严格的“最大下降速率”。本轮仅修正解释字典，不改历史特征或训练口径。ant_drawdown_3m是前三个月序列首值减末值，允许负值表示净上涨。相对水位像元量=(高程−站点历史水位分位数)/30m，无量纲。

## 服务器需要运行的部分：分组置换解释

只重建84维模型，两套协议各5折、seed42，共10次TabPFN拟合。复用已审计的每折4000个原上下文ID和checkpoint，保持2估计器、fit_preprocessors、预测批32。每折先预测完整测试集，并与历史seed42的84维分数比较，最大绝对偏差超过2e-5立即停止，不继续解释。

基线复现后，每折无放回均匀抽最多1024个测试单元，抽样不使用标签。对每个因子组的全部列使用同一个行排列，重复2次，再预测；不改变训练上下文、不重新学习特征选择。共有16组，每折32次置换预测；两套协议合计最多327,680条置换样本预测，另加51,018条完整基线预测。耗时取决于服务器，不保证固定小时数。test sample的AP和面积覆盖仅是抽样诊断，不替代完整OOF报告。

主要输出baseline AUC/AP减去置换后AUC/AP，负值也保留。相关因子间可能共享信息，组外相关关系会被破坏，不能解释为因果贡献或信息占比，不能把两个视角的贡献求和。保留分折、原始重复与带内结果。只有一个训练种子，2次置换不是独立地理重复或置信区间。需要更稳定解释时可增加--repeats，不能覆盖不同配置的旧目录；--max-per-fold 0表示全测试集，成本明显增加。

服务器不读取大像元数组、不重建GIS缓存，也不需要新装GIS包。仅需原服务器manifest.json、units.npz、模型checkpoint和既有环境。补丁含可信紧凑预测与上下文包，不依赖服务器旧ordinary/bands结果的存放位置。

### 上传（本地PowerShell）

```powershell
scp -P 30017 "C:\Users\dollars\code\subjects\dist\pixel_tabpfn_diagnostics_v1_patch.zip" root@10.252.18.70:/3250707052/landslide/
```

### 检查与串行运行（服务器SSH）

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
unzip -o pixel_tabpfn_diagnostics_v1_patch.zip
python experiments/pixel_tabpfn_diagnostics_v1/run.py check
mkdir -p logs
nohup python -u experiments/pixel_tabpfn_diagnostics_v1/run.py explain \
  --out results/pixel_tabpfn_diagnostics_v1/explain_s42 \
  > logs/pixel_tabpfn_diagnostics_v1_explain_s42.log 2>&1 < /dev/null &
echo "PID=$!"
```

如check仅提示缺少matplotlib，执行下列命令后重新check；不重装torch/TabPFN：

```bash
python -m pip install matplotlib==3.10.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
tail -n 40 logs/pixel_tabpfn_diagnostics_v1_explain_s42.log
```

空间五折后行政五折单进程串行。出现ALL DONE才表示两套全部完成。nohup启动成功后可断开SSH，保持服务器开机，不并行跑别的GPU任务。同配置重试原命令：逐项校验已完成折后跳过，未完成折重新拟合；输出目录存在但配置、代码或依赖不一致时停止，换新--out。没有训练中状态续传。

原始server units.npz必须与历史指纹一致；本地重建缓存仅允许做CPU描述，不允许拿它绕过模型基线核验。若原服务器缓存重建过且指纹不同，应保留错误与日志，先核实数据，不手动放宽检查。

### 导出（服务器SSH）

```bash
cd /3250707052/landslide
source /3250707052/.venv-tabpfn-cu124/bin/activate
python experiments/pixel_tabpfn_diagnostics_v1/run.py export \
  --out results/pixel_tabpfn_diagnostics_v1/explain_s42
```

导出会复核全部完成折和指标，不要求CUDA；同名ZIP存在时停止而不覆盖。

### 下载（本地PowerShell）

```powershell
scp -P 30017 root@10.252.18.70:/3250707052/landslide/results/pixel_tabpfn_diagnostics_v1/explain_s42_return.zip "C:\Users\dollars\code\subjects\"
```

下载到根目录后用可信本地代码复核（不执行ZIP内代码）：

```powershell
python experiments/pixel_tabpfn_diagnostics_v1/run.py review --archive explain_s42_return.zip --out results/pixel_tabpfn_diagnostics_v1/received_explain_s42
```

## 本地复现CPU诊断与制图

已有缓存实际在下列本地恢复目录。输出必须选不存在的新目录。

```powershell
python experiments/pixel_tabpfn_diagnostics_v1/run.py diagnostics --prepared results/pixel_tabpfn_restart_check/results/pixel_bag_repr_v1/prepared --reconstructed-local --out results/pixel_tabpfn_diagnostics_v1/cpu_new
python experiments/pixel_tabpfn_mapping_v2/mapping.py --diagnostics results/pixel_tabpfn_diagnostics_v1/cpu_new --out results/pixel_tabpfn_diagnostics_v1/cpu_new/maps
```

制图环境另需rasterio、pyproj、shapely、pyshp和matplotlib。本机使用既有.geo_deps，服务器仅跑check/explain/export时不需要这些GIS包。

## 验证范围

本地已验证真实ID/标签/上下文、两套CPU诊断、完整矢量与栅格输出，并检查地图预览。单元测试覆盖分组全列覆盖、正样本保留与测试隔离、缺单元拒绝、河道回填且保留原标注、历史基线不一致停止、显式预测桩下完整五折编排/跳过/导出/可信回传复算与篡改拒绝。预测桩不是模型实验；本地未运行真实TabPFN/CUDA解释，需服务器完成。
