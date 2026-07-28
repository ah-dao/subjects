"""
从Excel文件中提取2000-2021年有具体滑坡时间的滑坡点。
输出CSV文件，可直接拖入QGIS使用。
"""

import pandas as pd

# === 配置：根据你的Excel文件修改 ===
INPUT_FILE = "消落带隐患点.xls"          # 你的Excel文件路径
SHEET_NAME = 0                          # 工作表名或索引（0=第一个sheet）
DATE_COLUMN = "滑坡时间"                    # 滑坡日期列名（可能是"日期"/"发生日期"等，改成你文件中的实际列名）
LON_COLUMN = "经度"                # 经度列名
LAT_COLUMN = "纬度"                 # 纬度列名
# 如果列名是中文，改成如 "经度"、"纬度"、"发生日期"

OUTPUT_FILE = "landslide_points_2000_2021.csv"   # 输出CSV文件名

START_YEAR = 2000
END_YEAR = 2021

# === 1. 读取Excel ===
print(f"读取 {INPUT_FILE}...")
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
print(f"总记录数: {len(df)}")
print(f"列名: {list(df.columns)}")

# === 2. 解析日期列 ===
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

# 统计无法解析日期（NaT）的记录
nat_count = df[DATE_COLUMN].isna().sum()
print(f"无法解析日期的记录: {nat_count}")

# === 3. 过滤2000-2021年 ===
mask = (df[DATE_COLUMN].dt.year >= START_YEAR) & (df[DATE_COLUMN].dt.year <= END_YEAR)
filtered = df[mask].copy()
filtered = filtered.dropna(subset=[DATE_COLUMN])  # 去掉无日期的行
print(f"{START_YEAR}-{END_YEAR}年有日期的滑坡点: {len(filtered)}")

# === 4. 统计各年份滑坡数量 ===
filtered["year"] = filtered[DATE_COLUMN].dt.year
year_counts = filtered["year"].value_counts().sort_index()
print(f"\n各年份滑坡数量:")
for year, count in year_counts.items():
    print(f"  {year}: {count} 个")

# === 5. 查看复发（同一坐标多次出现）情况 ===
duplicate_mask = filtered.duplicated(subset=[LON_COLUMN, LAT_COLUMN], keep=False)
repeat_count = filtered[duplicate_mask].groupby([LON_COLUMN, LAT_COLUMN])[DATE_COLUMN].count()
print(f"\n同一位置多次滑坡: {len(repeat_count)} 个位置")
if len(repeat_count) > 0:
    print(f"  最多复发次数: {repeat_count.max()}")
    print(f"  复发≥2次的位置数: {(repeat_count >= 2).sum()}")
    print(f"  详情:")
    for (lon, lat), count in repeat_count.items():
        print(f"    ({lon}, {lat}): {count} 次")

# === 6. 导出CSV（可直接拖入QGIS） ===
# 保留关键列
output_cols = [DATE_COLUMN, LON_COLUMN, LAT_COLUMN, "year"]
for col in filtered.columns:
    if col not in output_cols and filtered[col].dtype in ["float64", "int64", "object"]:
        if col not in [DATE_COLUMN]:
            output_cols.append(col)

output_df = filtered[[c for c in output_cols if c in filtered.columns]]
output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n已导出: {OUTPUT_FILE} ({len(output_df)} 条记录)")
print("可直接拖入 QGIS 使用（图层 → 添加图层 → 添加文本数据图层，X=经度 Y=纬度）")
