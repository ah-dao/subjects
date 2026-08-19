"""
NDVI 30m vs 90m 分辨率验证（OPTIMIZATION_PATHS.md 8.2 / QUICKSTART 1.2 节）。

对同一年的 30m 与 90m NDVI，用斜坡单元分别做 zonal mean（单元内均值），
计算逐单元均值的 Pearson 相关系数 / R² / MAE / RMSE。

结论判断：r > 0.99 → 分辨率降到 90m 对单元级特征几乎无损，放心用 90m 全量导出。

用法：
    python tills/validate_ndvi_resolution.py \
        --tif30 data/gee/validation/ndvi_30m_2015.tif \
        --tif90 data/gee/validation/ndvi_90m_2015.tif

输出（默认 predictions/resolution_validation/ 下）：
    resolution_validation.csv   （unit_id, ndvi30_mean, ndvi90_mean, diff）
    resolution_scatter.png      （30m vs 90m 散点图）
    resolution_report.txt       （r / R² / MAE / RMSE 与结论）
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SLOPE_UNITS_SHP, PRED_DIR


def zonal_means(gdf, tif_path, nodata=-9999):
    """对单个栅格做单元 zonal mean，返回与 gdf 等长的数组。

    nodata 哨兵值（如 GEE 掩膜导出的 -9999）在统计时排除。
    """
    from rasterstats import zonal_stats
    result = zonal_stats(gdf, str(tif_path), stats=['mean'], all_touched=True,
                         nodata=nodata)
    return np.array([r['mean'] if r else np.nan for r in result], dtype=np.float64)


def raster_diagnostics(tif_path):
    """打印栅格基本统计，用于快速诊断导出异常（如整图为 0 / 掩膜被写成 0）。"""
    import rasterio
    import numpy as np
    with rasterio.open(tif_path) as src:
        a = src.read(1)
        info = {
            'bands': src.count, 'dtype': src.dtypes[0],
            'nodata': src.nodata, 'size': f'{src.width}x{src.height}',
            'bounds': [round(v, 3) for v in src.bounds],
            'min': float(np.nanmin(a)), 'max': float(np.nanmax(a)),
            'mean': float(np.nanmean(a)),
            'zero%': float((a == 0).mean() * 100),
            'sentinel%': float((a == -9999).mean() * 100),
        }
    print(f'  [{Path(tif_path).name}] bands={info["bands"]} dtype={info["dtype"]} '
          f'nodata={info["nodata"]} size={info["size"]}')
    print(f'    extent={info["bounds"]}')
    print(f'    min={info["min"]:.4f} max={info["max"]:.4f} mean={info["mean"]:.4f} '
          f'| 0值占比={info["zero%"]:.1f}% | -9999占比={info["sentinel%"]:.1f}%')
    if info['max'] == 0.0 or info['zero%'] > 90:
        print('    ⚠ 疑似导出异常：像元几乎全为 0（GEE 掩膜像素被导出为 0）。'
              '请用修复后的 gee_export_ndvi_validation.js 重新导出（掩膜像素改为 -9999），'
              '或在 QGIS 中打开 tif 确认是否有植被区域的值。')
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tif30', required=True, help='30m NDVI GeoTIFF 路径')
    parser.add_argument('--tif90', required=True, help='90m NDVI GeoTIFF 路径')
    parser.add_argument('--shp', default=str(SLOPE_UNITS_SHP), help='斜坡单元 shp')
    parser.add_argument('--out', default=str(PRED_DIR / 'resolution_validation'))
    parser.add_argument('--nodata', type=float, default=-9999,
                        help='栅格无数据哨兵值（-9999=GEE unmask 后的掩膜值），'
                             '统计时排除；传 0 表示无哨兵')
    args = parser.parse_args()

    import geopandas as gpd
    import pandas as pd

    if not Path(args.tif30).exists():
        raise FileNotFoundError(f'未找到 30m 栅格: {args.tif30}')
    if not Path(args.tif90).exists():
        raise FileNotFoundError(f'未找到 90m 栅格: {args.tif90}')

    print('=== 栅格自诊断 ===')
    raster_diagnostics(args.tif30)
    raster_diagnostics(args.tif90)

    # rasterstats 不会自动重投影矢量：shp(4326) 需先转到栅格坐标系(UTM)
    import rasterio
    with rasterio.open(args.tif30) as src:
        raster_crs = src.crs
    from src.dataset import load_units_reprojected
    gdf = load_units_reprojected(args.shp, raster_crs)
    print(f'斜坡单元数: {len(gdf)}')

    print('提取 30m zonal mean ...')
    s30 = zonal_means(gdf, args.tif30, nodata=args.nodata)
    print('提取 90m zonal mean ...')
    s90 = zonal_means(gdf, args.tif90, nodata=args.nodata)

    # 去掉任一侧缺失的单元
    ok = ~(np.isnan(s30) | np.isnan(s90))
    a, b = s30[ok], s90[ok]
    n_valid = ok.sum()
    print(f'有效单元（两栅格均有覆盖）: {n_valid} / {len(gdf)}')

    if n_valid < 10:
        raise RuntimeError('有效单元太少，请检查栅格与 shp 的范围是否对齐')

    # 指标（均只用逐元素运算，不依赖 lstsq）
    r = float(np.corrcoef(a, b)[0, 1])
    ss_res = float(((a - b) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    mae = float(np.abs(a - b).mean())
    rmse = float(np.sqrt(((a - b) ** 2).mean()))
    mean30, mean90 = float(a.mean()), float(b.mean())

    lines = [
        'NDVI 30m vs 90m 分辨率验证报告',
        '=' * 46,
        f'  有效单元数      : {n_valid}',
        f'  30m 单元均值范围: {a.min():.4f} ~ {a.max():.4f}（均值 {mean30:.4f}）',
        f'  90m 单元均值范围: {b.min():.4f} ~ {b.max():.4f}（均值 {mean90:.4f}）',
        f'  Pearson r       : {r:.4f}',
        f'  R²              : {r2:.4f}',
        f'  MAE             : {mae:.4f}',
        f'  RMSE            : {rmse:.4f}',
        '',
        '结论：',
        f'  r > 0.99  → 90m 对单元级特征几乎无损，全量导出用 90m（速度约快 9 倍）',
        f'  0.95<r<0.99 → 谨慎，检查单元面积是否过小或栅格对齐问题',
        f'  r < 0.95  → 不要降分辨率，30m 全量（按年×分段导出避免超时）',
        f'  本次 r = {r:.4f} → ' + ('放心用 90m' if r > 0.99 else '请按上面分级判断'),
    ]
    print('\n'.join(lines))

    # 输出
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'unit_id': gdf.index.astype(str), 'ndvi30_mean': s30,
                       'ndvi90_mean': s90, 'diff': s30 - s90})
    df.to_csv(out / 'resolution_validation.csv', index=False, encoding='utf-8-sig')
    (out / 'resolution_report.txt').write_text('\n'.join(lines), encoding='utf-8')

    # 散点图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(a, b, s=4, alpha=0.5)
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='1:1')
        ax.set_xlabel('NDVI 30m 单元均值')
        ax.set_ylabel('NDVI 90m 单元均值')
        ax.set_title(f'30m vs 90m（r={r:.4f}）')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'resolution_scatter.png', dpi=150)
        plt.close(fig)
        print(f'\n已导出: {out / "resolution_scatter.png"}')
    except Exception as e:
        print(f'[警告] 散点图生成失败: {e}')
    print(f'已导出: {out / "resolution_validation.csv"} / {out / "resolution_report.txt"}')


if __name__ == '__main__':
    main()
