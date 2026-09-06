"""
修复斜坡单元 shapefile 中的无效几何（PROJECT_OVERVIEW.md）。

- 用 shapely.make_valid 修复自相交、空环等问题
- 将 GeometryCollection 拆解为 Polygon/MultiPolygon（保证后续 zonal stats 可用）
- 若 shp 无 ID 列，自动补 unit_id（=行号），保证各脚本可稳定关联

用法：
    python tills/fix_slope_units.py
输出：
    data/slope_units/slope_units_fixed.shp
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SLOPE_UNITS_SHP

# 原始（未修复）斜坡单元 shp，改为你的实际路径
INPUT_SHP = r"C:\Users\dollars\研究生课程\课题\斜坡单元\three_gorges_slope_units_wgs84\slope_units_wgs84.shp"


def to_polygonish(geom):
    """把 GeometryCollection 拆成 Polygon/MultiPolygon；其它类型返回 None。"""
    import shapely.geometry as sg
    if geom is None:
        return None
    t = geom.geom_type
    if t in ('Polygon', 'MultiPolygon'):
        return geom
    if t == 'GeometryCollection':
        parts = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return sg.MultiPolygon([p if p.geom_type == 'Polygon' else list(p.geoms)
                                for p in parts])
    return None


def main():
    import geopandas as gpd
    from shapely import make_valid

    if not Path(INPUT_SHP).exists():
        raise FileNotFoundError(f'未找到原始 shp: {INPUT_SHP}（修改脚本顶部 INPUT_SHP）')

    print(f'读取: {INPUT_SHP}')
    gdf = gpd.read_file(INPUT_SHP)
    print(f'原始要素数: {len(gdf)}')

    # 补 unit_id（若已有 ID 列则保留）
    has_id = any(c in gdf.columns for c in ('unit_id', 'fid', 'FID', 'OBJECTID', 'id', 'ID'))
    if not has_id:
        gdf.insert(0, 'unit_id', gdf.index)
        print('已自动添加 unit_id 列（=行号）')

    # 修复无效几何
    invalid_count = int((~gdf.geometry.is_valid).sum())
    print(f'无效几何数: {invalid_count}')

    def fix(geom):
        if geom is None or geom.is_valid:
            return to_polygonish(geom)
        try:
            return to_polygonish(make_valid(geom))
        except Exception as e:
            print(f'  修复失败: {e}，该要素将被删除')
            return None

    gdf['geometry'] = gdf['geometry'].apply(fix)

    before = len(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()
    removed = before - len(gdf)
    if removed:
        print(f'已删除 {removed} 个无法修复的要素')

    still_invalid = int((~gdf.geometry.is_valid).sum())
    print(f'修复后仍无效: {still_invalid}')

    SLOPE_UNITS_SHP.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(SLOPE_UNITS_SHP)
    print(f'已导出: {SLOPE_UNITS_SHP}')
    print(f'最终要素数: {len(gdf)}（预期 26040 左右）')


if __name__ == '__main__':
    main()
