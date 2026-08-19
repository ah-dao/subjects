// ================================================================
// 方案 C v3：GEE 端直接计算斜坡单元逐年统计 → 一年一个 CSV
//
// v3 相对 v2 的改动：
//   1. 修复诊断逻辑（v2 的 diagDone 标记被第一个诊断消耗，列名永远打不出）；
//   2. 所有诊断都包 try/catch：网络抖动时诊断失败不影响导出任务提交；
//   3. 去掉大范围 reduceRegion 探测（同步请求重，容易触发连接失败）；
//   4. DIAG=false 时完全不打印诊断，最小化同步请求。
//
// v4 追加：只导出有用列（ID + 5 个统计），剔除 shp 无关属性与几何列。
//
// 用法：
//   1. 改 UNITS_ASSET（如需）
//   2. 先做单年测试：START_YEAR = END_YEAR = 2015 → Run
//      → 看 Console 诊断 → Tasks 面板点 RUN → 下载 CSV 检查列名
//   3. 无误后改回 2000/2021 全量提交（22 个任务）
//   4. CSV 放入 data/gee/unit_stats/ 后本地导入：
//      python tills/import_gee_unit_stats.py --year 2015
// ================================================================

// ---------------- 配置区 ----------------
var UNITS_ASSET  = 'projects/even-trainer-485802-e7/assets/three_gorges_slope_units_fixed'; // 斜坡单元 asset
var ID_COL       = 'Id';         // shp 中单元 ID 列名（实测为 'Id'；本地 merge 需一致）
var START_YEAR   = 2000;
var END_YEAR     = 2021;
var SCALE        = 90;           // 分辨率验证通过，用 90m
var CHUNK        = 0;            // 0 = 一年一个任务；超时改 5000
var FOLDER       = 'unit_stats'; // Drive 输出文件夹
var DIAG         = true;         // 打印诊断（不影响导出；网络不稳可设 false）

// ---------------- Landsat 云掩膜 + NDVI ----------------
function addNdviL89(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0)).and(qa.bitwiseAnd(1 << 5).eq(0));
  return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI').updateMask(mask);
}
function addNdviL57(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0)).and(qa.bitwiseAnd(1 << 5).eq(0));
  return img.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI').updateMask(mask);
}

// 单年 5 波段影像：ndvi + maxdaily + cumulative + max30d + heavydays
function yearImage(year, region) {
  var start = ee.Date.fromYMD(year, 6, 1);
  var end   = ee.Date.fromYMD(year, 9, 30);

  var ndvi = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(start, end).filterBounds(region).map(addNdviL89)
    .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
      .filterDate(start, end).filterBounds(region).map(addNdviL89))
    .merge(ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
      .filterDate(start, end).filterBounds(region).map(addNdviL57))
    .merge(ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
      .filterDate(start, end).filterBounds(region).map(addNdviL57))
    .median().rename('ndvi');

  var daily = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31))
    .filterBounds(region)
    .map(function(d) { return d.select('precipitation'); });

  var maxdaily = daily.max().rename('maxdaily');
  var cumulative = daily.sum().rename('cumulative');
  var heavy = daily.map(function(d) { return d.gt(50); }).sum().rename('heavydays');

  var list = daily.toList(daily.size());
  var lastStart = daily.size().subtract(30);
  var offsets = ee.List.sequence(0, lastStart, 30);
  var windows = offsets.map(function(off) {
    var imgs = ee.List.sequence(0, 29).map(function(k) {
      return ee.Image(list.get(ee.Number(off).add(k)));
    });
    return ee.ImageCollection(imgs).sum();
  });
  var max30d = ee.ImageCollection(windows).max().rename('max30d');

  return ndvi.addBands(maxdaily).addBands(cumulative).addBands(max30d).addBands(heavy);
}

// ---------------- 逐年导出 ----------------
var units = ee.FeatureCollection(UNITS_ASSET);
print('单元总数:', units.size().getInfo());

if (DIAG) {
  // 元数据级诊断（不涉及栅格计算，请求轻量）
  try { print('单元属性列:', units.first().propertyNames()); }
  catch (e) { print('诊断-属性列失败（忽略）:', e.message); }
}

// 只保留有用列（在第一次 reduceRegions 后取实际列名做交集）。
// 注意：统计列（ndvi/maxdaily/...）是 reduceRegions 之后才生成的，
// 不能从 units.first()（原始要素）取；也不能对 ee.List 用 JS filter
// （会被当作 ee.Filter 报 "Invalid argument specified for ee.Filter()"）。
// 失败时 PRESENT 为 null，导出全部列（不影响使用）。
var WANT_ARR = [ID_COL, 'ndvi', 'maxdaily', 'cumulative', 'max30d', 'heavydays'];
var PRESENT = null;
var presentDone = false;

function exportYear(year, feats, suffix) {
  var img = yearImage(year, feats.geometry().bounds().buffer(500));

  var stats = img.reduceRegions({
    collection: feats,
    reducer: ee.Reducer.mean(),
    scale: SCALE,
    tileScale: 4
  });

  if (!presentDone) {
    try {
      var allNames = stats.first().propertyNames().getInfo();
      PRESENT = WANT_ARR.filter(function(n) { return allNames.indexOf(n) >= 0; });
      if (DIAG) { print('诊断-最终导出列名:', PRESENT); }
    } catch (e) {
      print('诊断-列名获取失败（将导出全部列，不影响使用）:', e.message);
    }
    presentDone = true;
  }

  if (PRESENT) {
    stats = stats.select(PRESENT);
  }

  Export.table.toDrive({
    collection: stats,
    description: 'unit_stats_' + year + suffix,
    folder: FOLDER,
    fileFormat: 'CSV'
  });
}

if (CHUNK <= 0) {
  for (var y = START_YEAR; y <= END_YEAR; y++) {
    exportYear(y, units, '');
  }
} else {
  var nChunks = units.size().divide(CHUNK).ceil().getInfo();
  print('分块数:', nChunks, '| 任务数:', nChunks * (END_YEAR - START_YEAR + 1));
  for (var c = 0; c < nChunks; c++) {
    var chunk = ee.FeatureCollection(units.toList(CHUNK, c * CHUNK));
    for (var y = START_YEAR; y <= END_YEAR; y++) {
      exportYear(y, chunk, '_c' + c);
    }
  }
}

print('任务已提交。点 RUN 后下载 unit_stats_2015.csv，把列名和内容发我确认。');