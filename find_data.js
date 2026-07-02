// 导入 SRTM 高程数据 (30米分辨率)
var srtm = ee.Image("USGS/SRTMGL1_003").clip(roi);
// 高程 (Elevation)
var elevation = srtm.select('elevation');

// 坡度 (Slope) 和 坡向 (Aspect)
var terrain = ee.Terrain.products(srtm);
var slope = terrain.select('slope');
var aspect = terrain.select('aspect');
// 总曲率 (Curvature)
var curvature = srtm.convolve(ee.Kernel.laplacian8()).rename('curvature');
// TRI
var tri = srtm.convolve(ee.Kernel.fixed(3, 3, [
  [-1, -1, -1],
  [-1,  8, -1],
  [-1, -1, -1]
])).abs().rename('TRI');

// 方法A: 在研究区内随机生成非滑坡点（避开滑坡点）
var nonLandslideCount = landslidePoints.size().getInfo();  // 与滑坡点数量一致

var nonLandslidePoints = ee.FeatureCollection.randomPoints({
  region: roi,
  points: nonLandslideCount,
  seed: 42
});

// 为非滑坡点添加 class=0
nonLandslidePoints = nonLandslidePoints.map(function(f) {
  return f.set('class', 0);
});

// 为滑坡点添加 class=1
var landslideWithLabel = landslidePoints.map(function(f) {
  return f.set('class', 1);
});

var allPoints = landslideWithLabel.merge(nonLandslidePoints);
print('总样本点数量:', allPoints.size());
// 将点转为栅格标签
var labels = allPoints.reduceToImage(['class'], ee.Reducer.first()).unmask(0);

// 时序数据之后使用
// 导入 Landsat 8 表面反射率数据
// var l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
//   .filterBounds(roi)
//   .filterDate('2025-01-01', '2025-12-31')
//   .median() // 取年度中值，去除云干扰
//   .clip(roi);

// NDVI (归一化植被指数): (NIR - Red) / (NIR + Red)
// var ndvi = l8.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');

// NDWI (归一化水体指数): (Green - NIR) / (Green + NIR)
// var ndwi = l8.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
// 年均降水量 (Annual Precipitation)
// var rain = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
//   .filterDate('2024-01-01', '2024-12-31')
//   .sum() // 将每日降水累加得到年降水
//   .clip(roi)
//   .rename('Precipitation');

// 将所有因子“打包”
var finalStack = ee.Image.cat([
  elevation,
  slope,
  aspect,
  tri,
  curvature,
]).float(); // 统一转为浮点数格式以便计算
var combined = finalStack.addBands(labels.rename('label'));

var exportImage = combined.clip(roi);
print('导出影像波段:', exportImage.bandNames());
// Export.image.toDrive({
//   image: finalStack,
//   description: 'Terrain_Factors_MultiBand',
//   folder: 'GEE_Export',
//   fileNamePrefix: 'Terrain_MultiBand',
//   region: roi,
//   scale: 30,
//   fileFormat: 'GeoTIFF',
//   maxPixels: 1e13
// });
// 将地图中心定位到研究区
Map.centerObject(roi, 10); // 10 代表缩放级别
// --- 1. 显示高程 (DEM) ---
Map.addLayer(elevation, {min: 0, max: 4000, palette: ['blue', 'green', 'yellow', 'red']}, '高程图层');

// --- 2. 显示坡度 (Slope) ---
// 坡度通常在 0-90 度之间，滑坡多发于 20-50 度
Map.addLayer(slope, {min: 0, max: 60, palette: ['white', 'black']}, '坡度图层');

// --- 3. 显示坡向 (Aspect) ---
Map.addLayer(aspect, {min: 0, max: 360, palette: ['blue', 'cyan', 'green', 'yellow', 'red']}, '坡向图层');

// --- 4. 显示曲率 ---
var curvVis = {
  min: -0.01,
  max: 0.01,
  palette: [
    '#0000ff', // 蓝色：凹面 (Concave / Valley)
    '#ffffff', // 白色：平坦
    '#ff0000'  // 红色：凸面 (Convex / Ridge)
  ]
};

Map.addLayer(curvature, curvVis, 'Profile Curvature');

// --- 5. 显示TRI ---
var triVis = {
  min: 0,
  max: 10, 
  palette: ['#f7fcf5', '#e5f5e0', '#a1d99b', '#31a354', '#00441b'] // 由浅入深的绿色
};

Map.addLayer(tri, triVis, 'TRI (Terrain Ruggedness Index)');

Map.addLayer(exportImage, {bands: ['label'], min: 0, max: 1}, 'Label Map');
// --- 4. 显示 NDVI (植被) ---
// 颜色习惯：红色代表无植被，绿色代表茂密植被
// Map.addLayer(ndvi, {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, '植被指数');

// --- 5. 显示 NDWI (水体) ---
// Map.addLayer(ndwi, {min: -0.5, max: 0.5, palette: ['white', 'blue']}, '水体指数');

// --- 6. 显示年均降水量 ---
// Map.addLayer(rain, {min: 500, max: 2000, palette: ['white', 'blue', 'purple']}, '降水量图层');


