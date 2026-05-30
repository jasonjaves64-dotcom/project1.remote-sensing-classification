// =============================================================================
// GEE Code Editor 脚本 — Sentinel-2 L2A + Sentinel-1 GRD + SRTM DEM
// 目标区域：广东省佛山市
// 用法：复制全部代码 → 粘贴到 https://code.earthengine.google.com → 点 Run
// 输出：Google Drive → CropClassification 文件夹
// =============================================================================

// =============================================================================
// 配置
// =============================================================================
var FOSHAN = ee.Geometry.Rectangle([112.6, 22.7, 113.3, 23.4]);

var YEARS = [2023];
var DRIVE_FOLDER = "CropClassification";
var S2_CRS = "EPSG:32649";  // UTM zone 49N (佛山所在)

// =============================================================================
// Sentinel-2 L2A: SCL 最严格云掩膜 + 波段准备
// =============================================================================

// SCL 最严格模式：仅保留 VEGETATION(4) + BARE_SOILS(5)
function maskS2CloudsStrict(image) {
  var scl = image.select("SCL");
  var mask = scl.eq(4).or(scl.eq(5));
  return image.updateMask(mask);
}

// SCL 标准模式：保留 VEGETATION + BARE_SOILS + WATER
function maskS2CloudsStandard(image) {
  var scl = image.select("SCL");
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6));
  return image.updateMask(mask);
}

// 10 个光谱波段
var S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"];

function prepareS2(image) {
  image = maskS2CloudsStrict(image);
  // L2A 反射率 DN/10000 → [0,1]
  var optical = image.select(S2_BANDS).divide(10000).clamp(0, 1);
  var scl = image.select("SCL").toFloat();
  return optical.addBands(scl).copyProperties(image, ["system:time_start"]);
}

// =============================================================================
// Sentinel-1 GRD: dB 转换 + 比值
// =============================================================================

function prepareS1(image) {
  var vv_db = ee.Image(10).multiply(image.select("VV").log10()).rename("VV");
  var vh_db = ee.Image(10).multiply(image.select("VH").log10()).rename("VH");
  var ratio = image.select("VV").divide(image.select("VH")).rename("VV_VH_Ratio");
  return vv_db.addBands(vh_db).addBands(ratio)
    .copyProperties(image, ["system:time_start"]);
}

// =============================================================================
// 主流程
// =============================================================================

// 1. 可视化研究区域确认
Map.centerObject(FOSHAN, 10);
Map.addLayer(FOSHAN, {color: "red"}, "佛山市研究区");
print("研究区域: 广东省佛山市");
print("  经度: 112.6°E ~ 113.3°E");
print("  纬度: 22.7°N ~ 23.4°N");
print("  CRS: " + S2_CRS);
print("  SCL 模式: 最严格 (仅 VEGETATION + BARE_SOILS)");
print("");
print("⚠ 检查地图上的红框是否覆盖了佛山市");
print("  确认无误后，取消注释底部的导出代码，再次点 Run");
print("");

// 2. 预览一景 Sentinel-2 中值合成（确认波段和数据范围正确）
var s2_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(FOSHAN)
  .filterDate("2023-06-01", "2023-08-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
  .map(prepareS2);

var s2_preview = s2_collection.median().clip(FOSHAN);

print("Sentinel-2 可用景数 (2023夏):", s2_collection.size());
print("预览波段:", S2_BANDS.join(", ") + ", SCL");
Map.addLayer(s2_preview.select(["B4", "B3", "B2"]), {min: 0, max: 0.3, gamma: 1.4}, "S2 真彩色预览");

// 3. 预览 Sentinel-1
var s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(FOSHAN)
  .filterDate("2023-06-01", "2023-08-31")
  .filter(ee.Filter.eq("instrumentMode", "IW"))
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
  .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
  .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
  .map(prepareS1);

var s1_preview = s1_collection.mean().clip(FOSHAN);
print("Sentinel-1 可用景数 (2023夏, DESC):", s1_collection.size());
Map.addLayer(s1_preview.select("VV"), {min: -25, max: -5}, "S1 VV (dB)");

print("");
print("========================================");
print("预览通过后，取消注释下方导出代码重新 Run");
print("预计 12 月 S2 + 12 月 S1 + 1 DEM = 25 个任务");
print("任务在 https://code.earthengine.google.com/tasks 查看");
print("========================================");

// =============================================================================
// 导出代码 — 预览确认无误后取消注释
// =============================================================================

// // --- Sentinel-2 月度中值合成 ---
// YEARS.forEach(function(year) {
//   var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
//     .filterBounds(FOSHAN)
//     .filterDate(year + "-01-01", year + "-12-31")
//     .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
//     .map(prepareS2);
//
//   for (var m = 1; m <= 12; m++) {
//     var start = year + "-" + ("0" + m).slice(-2) + "-01";
//     var endDate = new Date(year, m, 1); // next month
//     var end = endDate.toISOString().slice(0, 7) + "-01";
//
//     var monthly = s2.filterDate(start, end).median().clip(FOSHAN);
//     var fname = "sentinel2_" + year + "_" + ("0" + m).slice(-2) + "_composite";
//
//     Export.image.toDrive({
//       image: monthly,
//       description: fname,
//       folder: DRIVE_FOLDER,
//       fileNamePrefix: fname,
//       region: FOSHAN,
//       scale: 10,
//       crs: S2_CRS,
//       maxPixels: 1e10,
//       fileFormat: "GeoTIFF"
//     });
//     print("✓ S2 " + m + "月: " + fname);
//   }
// });
//
// // --- Sentinel-1 月度均值合成 (DESCENDING) ---
// YEARS.forEach(function(year) {
//   var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
//     .filterBounds(FOSHAN)
//     .filterDate(year + "-01-01", year + "-12-31")
//     .filter(ee.Filter.eq("instrumentMode", "IW"))
//     .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
//     .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
//     .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
//     .map(prepareS1);
//
//   for (var m = 1; m <= 12; m++) {
//     var start = year + "-" + ("0" + m).slice(-2) + "-01";
//     var endDate = new Date(year, m, 1);
//     var end = endDate.toISOString().slice(0, 7) + "-01";
//
//     var monthly = s1.filterDate(start, end).mean().clip(FOSHAN);
//     var fname = "sentinel1_" + year + "_" + ("0" + m).slice(-2) + "_desc";
//
//     Export.image.toDrive({
//       image: monthly,
//       description: fname,
//       folder: DRIVE_FOLDER,
//       fileNamePrefix: fname,
//       region: FOSHAN,
//       scale: 10,
//       crs: S2_CRS,
//       maxPixels: 1e10,
//       fileFormat: "GeoTIFF"
//     });
//     print("✓ S1 " + m + "月: " + fname);
//   }
// });
//
// // --- SRTM DEM ---
// var srtm = ee.Image("USGS/SRTMGL1_003");
// var elevation = srtm.select("elevation").clip(FOSHAN);
// var slope = ee.Terrain.slope(elevation);
// var aspect = ee.Terrain.aspect(elevation);
// var demCombined = elevation.rename("elevation")
//   .addBands(slope.rename("slope"))
//   .addBands(aspect.rename("aspect"));
//
// Export.image.toDrive({
//   image: demCombined,
//   description: "srtm_foshan",
//   folder: DRIVE_FOLDER,
//   fileNamePrefix: "srtm_foshan",
//   region: FOSHAN,
//   scale: 30,
//   crs: S2_CRS,
//   maxPixels: 1e10,
//   fileFormat: "GeoTIFF"
// });
// print("✓ DEM: srtm_foshan");
//
// print("");
// print("全部任务已提交到 Google Drive → " + DRIVE_FOLDER);
// print("在 https://code.earthengine.google.com/tasks 查看进度");

// =============================================================================
// 辅助：查看典型物候曲线（可选）
// 在预览区点一个农田位置，看该点全年 NDVI 变化
// =============================================================================
// var ndviChart = ui.Chart.image.series({
//   imageCollection: s2_collection.select("B8"),
//   region: FOSHAN.centroid(),
//   reducer: ee.Reducer.mean(),
//   scale: 10
// });
// print(ndviChart);
