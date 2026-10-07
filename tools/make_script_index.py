# -*- coding: utf-8 -*-
"""从各脚本的模块 docstring 自动生成脚本索引（docs/脚本索引.md）。

为什么要自动生成：脚本会不断增删，手写索引必然过期。本脚本按"目录 + 文件名前缀"归类，
抽取每个 .py 的 docstring 首行作为一句话说明，并标注是否属于现行主线流程。

用法：python tools/make_script_index.py
输出：docs/脚本索引.md
"""
import ast
import os

ROOT = r'C:\Users\dollars\code\subjects'
OUT = os.path.join(ROOT, 'docs', '脚本索引.md')

STAGES = [
    ('src/', 'src', '核心库（配置 / 数据加载 / 指标）：被所有脚本共用，一般不直接运行'),
    ('根目录', '', '训练与基线入口（XGBoost 等），直接运行'),
    ('tills/ 特征提取', 'tills', '数据准备与特征提取（GEE / 地形 / 水系 / 土地利用 / 道路 / 地质四组 / 淹没）'),
    ('tills/ 消融实验', 'tills', '消融与筛选（含第 3~10 轮），输出 results/v2_ablation_*'),
    ('tills/ 出图与机理', 'tills', '出图、双口径归因、机理证据链、地质图层'),
    ('tills/ 交付与校验', 'tills', '交付表构建、ID/人口/口径校验'),
    ('tools/', 'tools', '一次性校验与制表工具（可重复运行的存档型脚本）'),
]
FILE_STAGE = {
    'v2_extract_': 'tills/ 特征提取', 'v2_build_': 'tills/ 特征提取', 'v2_assemble_': 'tills/ 特征提取',
    'litho_pipeline': 'tills/ 特征提取', 'merge_': 'tills/ 特征提取', 'build_final_population': 'tills/ 特征提取',
    'v2_fix_lithology': 'tills/ 特征提取', 'v2_dump_elev_pairs': 'tills/ 特征提取',
    'v2_ablation_': 'tills/ 消融实验', 'v2_finalize_selection': 'tills/ 消融实验',
    'v2_interaction_features': 'tills/ 消融实验', 'v2_univariate_scan': 'tills/ 消融实验',
    'v2_predict_map': 'tills/ 出图与机理', 'v2_mechanism_analysis': 'tills/ 出图与机理',
    'v2_build_geology_layers': 'tills/ 出图与机理', 'v2_run_baseline': 'tills/ 出图与机理',
    'v2_build_main38': 'tills/ 交付与校验', 'build_delivery_table': 'tills/ 交付与校验',
    'make_': 'tills/ 交付与校验', 'shp_io': 'tills/ 交付与校验', 'extract_landslide_points': 'tills/ 交付与校验',
}
STAGE_OVERRIDE = {'v2_build_geology_layers.py': 'tills/ 出图与机理'}
# 现行主线的关键脚本（在索引中加粗）
CORE = {'shp_io.py', 'v2_extract_geometry_terrain.py', 'v2_extract_network_features.py',
        'v2_extract_landuse.py', 'v2_extract_soil.py', 'v2_extract_lithology_era.py',
        'v2_extract_litho_hardness_38m.py', 'v2_extract_slopetype.py', 'v2_assemble_features.py',
        'v2_build_main38.py', 'baseline_xgb.py', 'v2_ablation_r10_geology.py',
        'v2_predict_map.py', 'v2_mechanism_analysis.py', 'v2_build_geology_layers.py',
        'make_column_status.py'}


def first_line(path):
    try:
        doc = ast.get_docstring(ast.parse(open(path, encoding='utf-8-sig').read()))
    except Exception as e:
        return '（无法解析：%s）' % type(e).__name__
    if not doc:
        return '（无 docstring）'
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line.rstrip('。')
    return '（无 docstring）'


def stage_of(rel):
    base = os.path.basename(rel)
    if base in STAGE_OVERRIDE:
        return STAGE_OVERRIDE[base]
    if rel.startswith('src'):
        return 'src/'
    if os.sep not in rel:
        return '根目录'
    if rel.startswith('tools'):
        return 'tools/'
    for k, v in FILE_STAGE.items():
        if base.startswith(k):
            return v
    return 'tills/ 交付与校验'


def main():
    files = []
    for d in ('src', 'tills', 'tools', '', 'archive'):
        base = os.path.join(ROOT, d) if d else ROOT
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            if f.endswith('.py'):
                rel = os.path.join(d, f) if d else f
                if rel.startswith('archive'):
                    continue
                files.append(rel)
    rows = {}
    for rel in sorted(set(files)):
        st = stage_of(rel)
        rows.setdefault(st, []).append((rel.replace('\\', '/'), first_line(os.path.join(ROOT, rel))))
    lines = ['# 脚本索引（自动生成）', '',
             '> 生成脚本：`tools/make_script_index.py`　｜　**请勿手改**，改脚本 docstring 后重跑即可。',
             '> 加粗 = 现行主线的关键脚本。', '']
    for st, _prefix, desc in STAGES:
        if st not in rows:
            continue
        lines += ['## %s' % st, '', desc, '', '| 脚本 | 说明 |', '|---|---|']
        for rel, d in rows[st]:
            mark = '**%s**' % rel if os.path.basename(rel) in CORE else rel
            lines.append('| `%s` | %s |' % (mark, d))
        lines.append('')
    lines += ['## 归档脚本', '',
              '`tills/_archive/` 与 `archive/` 下为已完成使命的一次性脚本（v1 时期或依赖已清理中间件），',
              '保留以溯源，不再随主线维护。见 `archive/v2_process/README.md`。', '']
    open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
    print('written:', OUT)
    print('stages:', {k: len(v) for k, v in rows.items()})


if __name__ == '__main__':
    main()
