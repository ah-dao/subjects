"""
从水位Excel表格中提取全局水位时序特征
结合滑坡日期，为每个斜坡单元生成水位相关特征
"""

import pandas as pd
import numpy as np

# ============================================================
# 配置（根据你的实际文件修改）
# ============================================================
WATER_FILE = "水位数据.xlsx"              # 水位Excel文件
WATER_DATE_COL = "date"                   # 日期列名
WATER_LEVEL_COL = "water_level"           # 水位列名

LANDSLIDE_FILE = "slope_units_count.csv"  # QGIS导出的统计结果
LANDSLIDE_ID_COL = "unit_id"              # 斜坡单元唯一ID
LANDSLIDE_DATE_COL = "landslide_date"     # 滑坡日期列（无滑坡则为空）
LANDSLIDE_COUNT_COL = "landslide_count"   # 滑坡次数字段

OUTPUT_FILE = "water_features.csv"

# ============================================================
# 1. 读取水位数据 → 计算月水位
# ============================================================
print("读取水位数据...")
water = pd.read_excel(WATER_FILE)
water[WATER_DATE_COL] = pd.to_datetime(water[WATER_DATE_COL])
water = water.sort_values(WATER_DATE_COL)

# 转为月尺度
water["year_month"] = water[WATER_DATE_COL].dt.to_period("M")
monthly = water.groupby("year_month").agg(
    water_level=(WATER_LEVEL_COL, "mean"),
).reset_index()
monthly["year_month"] = monthly["year_month"].astype(str)

# 计算月降幅（正值=上涨，负值=下降）
monthly["drawdown"] = monthly["water_level"].diff().clip(upper=0)

print(f"  水位数据: {len(water)} 条逐日记录 → {len(monthly)} 个月")

# ============================================================
# 2. 读取斜坡单元滑坡统计
# ============================================================
print("读取滑坡单元数据...")
su = pd.read_csv(LANDSLIDE_FILE)
print(f"  斜坡单元总数: {len(su)}")
print(f"  有滑坡的单元: {(su[LANDSLIDE_COUNT_COL] > 0).sum()}")

# ============================================================
# 3. 为每个单元提取水位特征
# ============================================================
def extract_water_features(landslide_date, monthly_df):
    """
    根据滑坡日期截断月水位序列，提取特征

    参数:
        landslide_date: pd.Timestamp 或 None (未滑坡)
        monthly_df: 月水位 DataFrame

    返回: dict of features
    """
    if pd.isna(landslide_date) or landslide_date is None:
        cutoff = pd.Timestamp("2021-12-31")  # 未滑坡→全窗口
    else:
        cutoff = pd.Timestamp(landslide_date)

    # 截断：只用截止日期之前的月度数据
    monthly_df["ym_dt"] = pd.to_datetime(monthly_df["year_month"].astype(str) + "-01")
    mask = monthly_df["ym_dt"] <= cutoff
    pre = monthly_df[mask].copy()

    n = len(pre)
    if n < 2:
        # 太短，返回默认值
        return {
            "exposure_months": n,
            "max_monthly_drawdown": 0,
            "drawdown_3m_cumulative": 0,
            "drawdown_1m_prior": 0,
            "rapid_drawdown_events": 0,
            "mean_drawdown_rate": 0,
            "last_month_level": pre["water_level"].iloc[-1] if n > 0 else 0,
            "water_range_total": 0,
        }

    features = {}
    features["exposure_months"] = n
    features["max_monthly_drawdown"] = abs(pre["drawdown"].min())
    features["mean_drawdown_rate"] = abs(pre["drawdown"].mean())

    # 前3月累计降幅（消落期核心指标）
    features["drawdown_3m_cumulative"] = abs(pre["drawdown"].iloc[-3:].sum())

    # 前1月降幅
    features["drawdown_1m_prior"] = abs(pre["drawdown"].iloc[-1])

    # 骤降事件（月降幅 > 3m）
    features["rapid_drawdown_events"] = (pre["drawdown"] < -3).sum()

    # 最后一个月水位
    features["last_month_level"] = pre["water_level"].iloc[-1]

    # 总水位波动幅度
    features["water_range_total"] = pre["water_level"].max() - pre["water_level"].min()

    return features


print("提取水位特征...")
features_list = []

for _, row in su.iterrows():
    unit_id = row[LANDSLIDE_ID_COL]
    landslide_date = row.get(LANDSLIDE_DATE_COL, None)

    feats = extract_water_features(landslide_date, monthly.copy())
    feats["unit_id"] = unit_id
    features_list.append(feats)

water_features = pd.DataFrame(features_list)

# ============================================================
# 4. 输出
# ============================================================
# 按 unit_id 关联回原表
result = su.merge(water_features, on="unit_id", how="left")

# 选列输出
output_cols = [
    LANDSLIDE_ID_COL, LANDSLIDE_COUNT_COL,
    "exposure_months", "max_monthly_drawdown", "drawdown_3m_cumulative",
    "drawdown_1m_prior", "rapid_drawdown_events",
    "mean_drawdown_rate", "last_month_level", "water_range_total",
]
result[output_cols].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"已导出: {OUTPUT_FILE}")
print(f"共 {len(result)} 个单元的水位特征")

# 打印摘要
has_landslide = result[result[LANDSLIDE_COUNT_COL] > 0]
print(f"\n滑坡单元水位特征摘要:")
print(has_landslide[["exposure_months", "max_monthly_drawdown",
                      "drawdown_3m_cumulative", "rapid_drawdown_events"]].describe())
