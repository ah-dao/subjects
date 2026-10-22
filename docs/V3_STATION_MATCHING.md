# V3站点匹配修正

## 坐标与证据

旧`tills/v2_build_inundation_variants.py`将清溪场经度设为110.75°E并解释为秭归；清溪场实际属于涪陵干流站，在寸滩与万县之间。旧脚本仅比较经度差，清溪场没有分配到研究区单元。

V3统一采用何文社、陈骥、戴会超2008年《水力发电学报》论文[Table 1](https://hub.hku.hk/bitstream/10722/71058/1/Content.pdf)的四站代表坐标，并用[涪陵区政府公开环评报告](https://www.fl.gov.cn/bm/sthjj/zwgk_46444/fdzdgknr_46446/zdhjyxpj/jsxmhpspqk/202506/W020250630578309655902.pdf)第3.1.5节交叉核对清溪场的区域和上下游关系。

|站点|经度|纬度|
|---|---|---|
|寸滩|106°36′|29°37′|
|清溪场|107°24′|29°45′|
|万县|108°25′|30°45′|
|奉节|109°31′|31°02′|

坐标仅精确到角分，原文未明确坐标基准；代码按WGS84解释并记录假定。这是有文献来源的历史代表点，不是覆盖整个观测期的逐次迁站测量坐标。后续取得水文部门坐标与迁站记录时，需版本化更新配置并重建匹配。

## 方法与结果

单元转UTM49N计算面中心，再转回经纬度；计算WGS84椭球上中心到四站的距离并取最小值。保留次近站和距离差，小于5km标记人工复核。5km不是经过校准的位置误差界限。此为空间就近代理，未解决沿河距离、支流连通性或水面插值。

另用旧几何中心和旧坐标重建V2分配，25,939条完全一致；输出“只改坐标仍按经度”的分配，拆分坐标与方法的影响。

|训练单元|V2|V3|
|---|---:|---:|
|寸滩|7,298|3,793|
|清溪场|0|6,909|
|万县|12,043|8,551|
|奉节|6,168|6,256|
|总计|25,509|25,509|

训练单元更换站点7,073个，含正样本215个；全量更换7,137个。仅改坐标导致训练单元变化6,604个；再改距离方法变化543个，这两项可能重叠，不能相加作为最终变化数。837个训练单元的最近/次近距离差小于5km；最远匹配约75.09km，站点仍然稀疏。

## 输出

默认`features/v3/station_matching/`：

- `station_assign_v3_full.csv` / `station_assign_v3_train.csv`：全量25,939/训练25,509条，含站点、中心、距离和边界提示。
- `comparison_v2_v3.csv` / `transition_train.csv`：逐单元对照与站点迁移表。
- `daily_station_observations.csv`：权威工作簿的原始有效观测，保留全部年份，**不是可直接输入模型的全期特征**。
- `station_registry.csv` / `water_stations.json`：使用坐标及证据；`summary.json` / `manifest.json`：人口、变更、缺测、输入/源码/输出指纹。

各站观测期和缺测不同；清溪场有2条无效日期/水位行。本轮未插值、未前后填充、未实施事件截断，未读取弃用的`水位.xlsx`。下一阶段按窗口截取并审计缺测后再生成模型特征。

## 服务器命令（仅CPU）

本地已完成计算。无需立即上服务器，也无需停V2归因任务；如需复算，可上传补丁。

本地PowerShell：

代码同步统一使用Git，见[V3 Git与SSH流程](V3_GIT_SSH_WORKFLOW.md)；结果按该指南通过SSH自行导出。

服务器：

```bash
cd /3250707052/landslide
python3 -m venv .venv-v3-prep
source .venv-v3-prep/bin/activate
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-v3-prep.txt
python -m src.v3.station_matching build
python -m src.v3.station_matching verify
python -m unittest discover -s tests/v3 -p 'test_*.py'
```

已存在默认输出时只执行`verify`。重算须指定新目录，例如`build --out features/v3/station_matching_check`，再以相同`--out`校验。补丁不含本地输出，使用服务器现有原始数据与V2身份对照。不同GIS版本可能产生极小距离差异，以人口、分配及输入指纹为主核对。

不要在服务器任务运行期间照搬本地归档清理脚本。准备环境与`.venv-tabpfn-cu124`独立，不更改torch或TabPFN；补丁仅新增V3文件，不覆盖冻结V2实验代码。
