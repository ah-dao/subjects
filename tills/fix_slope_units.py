"""
修复斜坡单元 shapefile 中的无效几何图形
使用 shapely.make_valid 自动修复自相交、空环等问题
"""

import geopandas as gpd
from shapely import make_valid

# ============================================================
# 修改这里：你的斜坡单元文件路径
# ============================================================
INPUT_SHP = r"C:\Users\dollars\研究生课程\课题\斜坡单元\three_gorges_slope_units_wgs84\slope_units_wgs84.shp"    # 改为你的实际路径
OUTPUT_SHP = r"C:\Users\dollars\研究生课程\课题\斜坡单元\three_gorges_slope_units_fixed\slope_units_fixed.shp"    # 输出路径


# ============================================================
# 读取并修复
# ============================================================
print(f"读取: {INPUT_SHP}")
gdf = gpd.read_file(INPUT_SHP)
print(f"原始要素数: {len(gdf)}")

# 统计无效几何
invalid_count = (~gdf.geometry.is_valid).sum()
print(f"无效几何数: {invalid_count}")

if invalid_count == 0:
    print("没有无效几何，无需修复。")
else:
    # 修复
    def safe_make_valid(geom):
        if geom is None:
            return None
        if geom.is_valid:
            return geom
        try:
            return make_valid(geom)
        except Exception as e:
            print(f"  修复失败: {e}, 该要素将被删除")
            return None

    gdf["geometry"] = gdf["geometry"].apply(safe_make_valid)

    # 删除修复后仍为 None 的要素
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna()].copy()
    after = len(gdf)
    removed = before - after
    if removed > 0:
        print(f"已删除 {removed} 个无法修复的要素")

    # 验证修复结果
    still_invalid = (~gdf.geometry.is_valid).sum()
    print(f"修复后仍无效: {still_invalid}")

# 导出
gdf.to_file(OUTPUT_SHP)
print(f"已导出: {OUTPUT_SHP}")
print(f"最终要素数: {len(gdf)}")
print("完成。")
