// ================================================================
// NDVI 分辨率验证导出：30m vs 90m（单年）
// 目的：验证"降分辨率到 90m 对斜坡单元 zonal mean 精度的影响"
// 说明：使用斜坡单元 asset 作为导出区域（需先上传 shp，见 tills/ 说明）
// 用法：改配置区 → 粘贴到 GEE Code Editor → Run
//       → 右侧 Tasks 面板里两个任务分别点 RUN（可并行提交）
// 输出：Google Drive 的 ndvi_res_validation 文件夹，两个 GeoTIFF
// ================================================================

// ---------------- 配置区（按需修改） ----------------
var ASSET      = 'users/你的用户名/slope_units_fixed'; // 斜坡单元 asset 路径（替换成你的）
var YEAR       = 2015;                                // 验证年份（选一个云相对少的年份）
var FOLDER     = 'ndvi_res_validation';               // Drive 输出文件夹（不存在会自动建）
var CRS        = 'EPSG:32649';                        // 库区中段 UTM 49N；若提示投影范围错误可换 EPSG:32648
var BUFFER_M   = 500;                                 // 区域外扩缓冲（米），保证覆盖单元边缘
var USE_L7     = true;                                // 是否纳入 Landsat-7（SLC-off 有条纹，可关掉）
var MAX_PIXELS = 1e10;                                // 导出像素上限（单年单波段足够小）

// ---------------- 研究区（斜坡单元外包络 + 缓冲） ----------------
var units  = ee.FeatureCollection(ASSET);
// 关键：导出区域用"外包络矩形"，不要用 units.geometry().buffer(...)。
// 后者会把 2.6 万个多边形合并、缓冲成巨型几何，客户端 XHR 请求体过大，
// 就会报 "Failed to execute 'send' on 'XMLHttpRequest' ... value:compute"。
// 矩形外包络能覆盖所有单元，导出范围与合并后几何的包围盒一致。
var region = units.geometry().bounds().buffer(BUFFER_M);
print('斜坡单元数:', units.size());
print('区域面积(km²):', region.area().divide(1e6));

// ---------------- 云掩膜 + NDVI 函数 ----------------
// Landsat 8/9 (C02 SR): NIR=SR_B5, Red=SR_B4
function addNdviL89(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)   // 云
              .and(qa.bitwiseAnd(1 << 4).eq(0))  // 云影
              .and(qa.bitwiseAnd(1 << 5).eq(0)); // 雪
  return img.normalizedDifference(['SR_B5', 'SR_B4'])
            .rename('NDVI').updateMask(mask);
}

// Landsat 5/7 (C02 SR): NIR=SR_B4, Red=SR_B3
function addNdviL57(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
              .and(qa.bitwiseAnd(1 << 4).eq(0))
              .and(qa.bitwiseAnd(1 << 5).eq(0));
  return img.normalizedDifference(['SR_B4', 'SR_B3'])
            .rename('NDVI').updateMask(mask);
}

// ---------------- 单年生长季 NDVI median ----------------
var start = ee.Date.fromYMD(YEAR, 6, 1);
var end   = ee.Date.fromYMD(YEAR, 9, 30);

var ndviCol = ee.ImageCollection([])
  .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(start, end).filterBounds(region).map(addNdviL89))
  .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate(start, end).filterBounds(region).map(addNdviL89));

if (USE_L7) {
  ndviCol = ndviCol
    .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
      .filterDate(start, end).filterBounds(region).map(addNdviL57));
}

// L5 只到 2012 年，2015+ 年份为空集合，merge 无害；若验证年份 ≤ 2012 可自行打开：
// ndviCol = ndviCol.merge(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
//   .filterDate(start, end).filterBounds(region).map(addNdviL57));

print('参与中值的影像数:', ndviCol.size());
var ndvi30 = ndviCol.median().rename('NDVI_' + YEAR);

// 90m：真均值聚合（30m 像元在 90m 格网内求均值），再固定到 UTM 投影
// 注意：merge 了 L5/7/8/9 的集合取 median 后，影像没有有效默认投影，
// reduceResolution 会报 "does not have a valid default projection"，
// 必须先 setDefaultProjection 指定为 30m 的 UTM 投影。
var ndvi90 = ndvi30
  .setDefaultProjection(CRS, null, 30)
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs: CRS, scale: 90});

// 关键：掩膜（无有效观测）像素导出为 -9999 哨兵值，而不是 GEE 默认的 0。
// 否则整张 tif 大片为 0，zonal mean 会被污染成 0（当前遇到的验证失败即因此）。
var NODATA = -9999;
ndvi30 = ndvi30.unmask(NODATA);
ndvi90 = ndvi90.unmask(NODATA);

// ---------------- 地图预览 ----------------
// 注意：不要对整条库区做 30m 全区域预览，会让浏览器渲染卡死。
// 这里只预览研究区中心 5km 小块；如仍卡顿或想彻底避开，把 PREVIEW 设为 false。
var PREVIEW = true;
if (PREVIEW) {
  // 只预览外包络中心 5km 小块，避免整条库区 30m 渲染卡死
  var previewGeom = region.centroid().buffer(5000);
  Map.centerObject(previewGeom, 12);
  Map.addLayer(ndvi30.clip(previewGeom).updateMask(ndvi30.neq(NODATA)),
    {min: -0.2, max: 0.9,
      palette: ['#d73027','#f46d43','#fdae61','#fee08b','#d9ef8b','#a6d96a','#66bd63','#1a9850']},
    'NDVI ' + YEAR + ' (30m, 预览5km)');
}

// ---------------- 导出 ----------------
Export.image.toDrive({
  image: ndvi30,
  description: 'ndvi_30m_' + YEAR,
  folder: FOLDER,
  region: region,
  scale: 30,
  crs: CRS,
  maxPixels: MAX_PIXELS
});

Export.image.toDrive({
  image: ndvi90,
  description: 'ndvi_90m_' + YEAR,
  folder: FOLDER,
  region: region,
  scale: 90,
  crs: CRS,
  maxPixels: MAX_PIXELS
});
