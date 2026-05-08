/**
 * GEE 滑坡易发性因子导出脚本
 * 
 * 功能：
 * - 将所有滑坡影响因子合并为一个多波段影像
 * - 每个因子作为一个波段
 * - 直接导出为一张 GeoTIFF 文件
 * 
 * 使用方法：
 * 1. 打开 https://code.earthengine.google.com
 * 2. 创建新脚本，粘贴此代码
 * 3. 修改 studyRegion 定义你的研究区
 * 4. 在 Task 面板点击 RUN
 */

// ========================================
// 1. 初始化
// ========================================
ee.Initialize();

// ========================================
// 2. 定义研究区域（修改为你自己的区域）
// ========================================
var studyRegion = ee.Geometry.Rectangle([104.0, 30.0, 105.0, 31.0]);

// ========================================
// 3. 计算滑坡影响因子
// ========================================
function calculateFactors() {
  // DEM 高程数据
  var dem = ee.Image('USGS/SRTMGL1_003');
  
  // 1. 高程
  var elevation = dem.rename('elevation');
  
  // 2. 坡度
  var slope = ee.Terrain.slope(dem).rename('slope');
  
  // 3. 坡向
  var aspect = ee.Terrain.aspect(dem).rename('aspect');
  
  // 4. 地形粗糙度指数 (TRI)
  var terrain = ee.Algorithms.Terrain(dem);
  var tri = terrain.select('TRI').rename('TRI');
  
  // 5. 总曲率
  var curvature = terrain.select('convex').rename('curvature');
  
  // 合并所有波段为一个影像
  var combined = elevation
    .addBands(slope)
    .addBands(aspect)
    .addBands(tri)
    .addBands(curvature);
  
  return combined;
}

// ========================================
// 4. 添加滑坡标签（可选）
// ========================================
function addLabels(image) {
  // 如果你有滑坡点数据，取消下面注释并修改
  // var landsidePoints = ee.FeatureCollection('YOUR_ASSET_ID');
  // var labels = landsidePoints.reduceToImage(['class'], ee.Reducer.first());
  // return image.addBands(labels.rename('label'));
  
  return image;
}

// ========================================
// 5. 执行导出
// ========================================
var factorsImage = calculateFactors();
factorsImage = addLabels(factorsImage);

// 裁剪到研究区
var exportImage = factorsImage.clip(studyRegion);

// 打印波段信息
print('波段列表:', exportImage.bandNames());
print('分辨率:', exportImage.projection().nominalScale());
print('影像尺寸信息:', exportImage.getInfo());

// 创建导出任务
var task = ee.batch.Export.image.toDrive({
  image: exportImage,
  description: 'landslide_factors_multiband',
  folder: 'GEE_Exports',
  fileNamePrefix: 'landslide_factors_multiband',
  scale: 100,
  region: studyRegion,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

// 开始导出
task.start();

print('导出任务已提交!');
print('请在 Task 面板查看进度');
print('导出的文件包含以下波段: elevation, slope, aspect, TRI, curvature');
