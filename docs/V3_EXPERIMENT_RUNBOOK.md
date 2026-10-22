# V3完整实验包：84维像元统计量＋TabPFN

这是V3基础训练指南。已有本地返回保留于`results/v3_received_stats84/`；不因文档整理重跑训练。当前解释入口见[V3 canonical模型](V3_EXPLAIN84_CANONICAL_MODEL.md)，正式出图见[V3图集](V3_BASEMAP_ATLAS.md)。

本包在独立目录运行：25,509个建模单元、661个正样本，空间/行政各5折、seed42/43/44，共30次正式拟合。另有2次32单元预测的真实GPU冒烟检查，只检查能否运行，不报告其性能。五折是四折训练、一折测试；每折对全部测试单元预测，4,000仅限制训练上下文。

## 数据组织与本轮审计

```text
data/slope_units/        全量25,939与训练25,509单元几何、计数和事件日期
data/v3/groups/          未进行全人口均值填充的18维静态因子
data/v3/sources/         按当前unit_id对齐的2000—2021年/月矩阵、权威水位工作簿
data/v3/stations/        V3站点分配、原始逐日水位与来源记录
data/v3/fold_water/      十份固定参照日期、事件前水位、像元统计及质量记录
data/v3/audit/           661个正样本定义复核、160行独立标量核对、范围与缺失审计
data/v3/context_ids.json 原30组4,000上下文ID，保留训练正样本与原抽样规则
features/v3/model_inputs/spatial/fold0 ... fold4/
features/v3/model_inputs/admin/fold0 ... fold4/
                        每折features84.csv/npz、日期、缺失统计
splits/v3/              冻结空间与行政五折
src/v3/                 独立训练、组装、时间窗口与核验代码
models/tabpfn/          已验证的本地模型权重，不自动下载
results/                在该V3目录内生成，续跑按每折每种子核验DONE
```

最终84维=38表格＋2像元数量/可用性＋44像元统计。CSV中的unit_id、label、role、reference_month均为元数据，不进入模型；精确模型列序见schema.json。NPZ与CSV已交叉核验，全部661个正样本的84维在十份视图中一致。

|因子|实际时间尺度与算子|
|---|---|
|NDVI均值|T−2、T−1两个完整日历年均值|
|NDVI变化|上述均值减去2000至T−1的长期均值，是差值，不是百分比变化率|
|年最大日雨/最大30日雨|T−2、T−1两年的年度极值取最大|
|暴雨日数|T−2、T−1的日数相加|
|年累计降水|T−2、T−1两年年累计量取均值，不是两年之和|
|前1/3/6月降水|事件/参照月之前1/3/6个完整自然月累计，不含当月，不是近30/90/180天|
|汛期比例|T−1年5—9月累计雨量÷该年全年累计|
|用地现状|T−1年的耕地/建设用地占比，不是两年均值|
|用地变化|T−1占比减T−2占比；变化频次是这一次相邻年比较是否超过0.02，取0或1|
|水位累计背景|2001-01-01至事件/参照月月初，不含当月|
|前1月淹没、前3月水位净降幅|各自保留原短时窗口，不改成2001起累计|

原始矩阵仍保留2000年，供NDVI长期背景与原两年年度尺度使用；只有水位背景统一2001年起。所有动态组共用该协议/折的参照日期，非滑坡年月联合分布仅取外折训练正样本。日期种子与模型种子分离。

正样本15个气候/用地变量可比较值与历史定义最大差小于1e-7。1个正样本(unit_id=21996)的前两年NDVI包含缺测年，V3不再以剩余一年代表完整两年窗口，保留NDVI均值和差值缺失。原始降水和用地矩阵无缺失；NDVI存在缺失，窗口未完整观测则保留NaN。18维静态组也从未全人口填充的源表重建。

清溪场长期缺测影响17个正样本的背景水位；短缺口最多沿用之前水位3天，长缺口保留NaN。原完整DEM像元对有458个非滑坡单元无覆盖，原有效地形袋为空10个；不删单元、不以0冒充观测。每次拟合只用该4,000条上下文计算填充值和标准化参数，整列无观测时使用明确的0中心/1尺度并记入fit.json。没有用测试折或全人口均值填补。

本地验证了数据、30个上下文、时间边界、真实数据的预测桩编排/续跑/导出拒绝；真实TabPFN/CUDA拟合由服务器执行。训练只用当前84维方案，暂不增加采样优化、XGBoost对照或新的MIL训练，以便先获得V3基线。结果属于历史回溯的研究区内部地域迁移评价，不是未来某时期风险概率。

## 1. Git部署

按[Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)克隆到`/3250707052/subjects-git`，复用服务器已有权重和结果。无需打包或解压；不要在旧实验目录强行覆盖。

## 2. Python环境

新项目目录可复用已验证的虚拟环境；环境与结果目录互相独立，复用环境不会混入V2结果：

```bash
source /3250707052/.venv-tabpfn-cu124/bin/activate
python -c "import sys,torch,tabpfn; print(sys.executable); print(torch.__version__,torch.version.cuda,tabpfn.__version__,torch.cuda.is_available())"
python -m src.v3.train check
```

若现有环境check通过，不要重新装包。若新服务器没有该环境，可创建：

```bash
python3.10 -m venv /3250707052/.venv-v3-cu124
source /3250707052/.venv-v3-cu124/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-v3-train.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m src.v3.train check
```

训练使用预计算统计量，无需安装rasterio/GDAL等GIS包。check检查CUDA、TabPFN8.5.0、torch2.6.0、权重指纹、十份数据和全部上下文；不会自动下载其他模型。

## 3. 运行时启动串行实验

```bash
cd /3250707052/subjects-git
source /3250707052/.venv-tabpfn-cu124/bin/activate
mkdir -p logs
nohup bash scripts/run_v3.sh train > logs/v3_suite.log 2>&1 < /dev/null &
echo "PID=$!"
```

脚本依次检查环境→两次真实GPU冒烟→空间五折三种子→行政五折三种子→导出。参数固定为4,000上下文、2估计器、fit_preprocessors、预测批32。每次拟合结束释放模型，单进程串行。可关闭SSH，保持服务器运行；30次拟合耗时取决于服务器，不保证一夜完成。

```bash
tail -n 40 logs/v3_suite.log
nvidia-smi
```

末尾出现V3 SUITE COMPLETE才表示两套正式实验及导出全部完成。中断后在相同环境、相同代码和数据下重跑同一nohup命令，校验已完成任务后跳过；未完成的任务重新拟合，不能恢复任务内训练状态。配置或版本改变时会拒绝复用输出。

## 4. 检查、导出与下载

```bash
cd /3250707052/subjects-git
source /3250707052/.venv-tabpfn-cu124/bin/activate
tail -n 50 logs/v3_suite.log
cat results/v3_stats84_4k/summary.json
```

Git入口只运行实验，不自动生成ZIP。训练完成后需要结果包时手动导出：

```bash
python -m src.v3.train export --out results/v3_stats84_4k
```

同名ZIP已存在时不覆盖。下载在本地PowerShell执行：

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

## 输出和复核

逐折/种子保存测试预测、上下文及训练/测试ID、填充/缩放参数、输入指纹。汇总含每折、每种子拼接OOF、三种子平均评分，报告AUC、AP、前20%面积捕获率。全体25,509人口为主要评价；preevent_inundation_subgroup是按各折事件前淹没比例定义的辅助组，缺失背景不进入该组，不与旧版全期组直接等同。

导出核验完整五折与种子覆盖，排除冒烟及预测桩结果。返回本地后：

```bash
python -m src.v3.train review --archive v3_stats84_4k_return.zip --out results/v3_received_stats84
```

仅需CPU。它核对冻结上下文、特征/源码/权重指纹、ID/标签与训练侧缩放参数，并从逐单元预测重算指标。

要重新组装特征，在保持data/v3原始矩阵、日期和水位包不变时执行：

```bash
python -m src.v3.assemble build --out features/v3/model_inputs_rebuilt
python -m src.v3.assemble verify --out features/v3/model_inputs_rebuilt
```

运行时直接使用包中已经核验的默认features/v3/model_inputs即可。
