"""
GEE 数据下载脚本 — Sentinel-2 L2A + Sentinel-1 GRD + SRTM DEM
目标区域：广东省佛山市
输出：Google Drive → 本地 raw_data/

用法：
  python scripts/gee_download_sentinel2.py

数据产出（每月一个 GeoTIFF）：
  sentinel2/sentinel2_2023_01_composite.tif  — 10波段光学 + SCL
  sentinel1/sentinel1_2023_01_desc.tif       — VV + VH + ratio
  dem/srtm_foshan.tif                        — elevation + slope + aspect
"""

import ee
import os
from datetime import datetime, timedelta

# 先初始化 GEE
try:
    ee.Initialize()
    print("GEE 初始化成功\n")
except Exception as e:
    print(f"GEE 初始化失败: {e}")
    print("请运行: earthengine authenticate")
    exit(1)

# =============================================================================
# 配置 — 修改这里
# =============================================================================
# 佛山市 (覆盖三水、高明、南海主要农业区)
FOSHAN = ee.Geometry.Rectangle([112.6, 22.7, 113.3, 23.4])

YEARS = [2023]
OUTPUT_DIR = "./raw_data"
DRIVE_FOLDER = "CropClassification"

# Sentinel-2 参数
S2_SCALE = 10  # 10m 分辨率
S2_CRS = "EPSG:32649"  # UTM zone 49N (佛山所在)

# Sentinel-1 参数
S1_SCALE = 10
S1_CRS = "EPSG:32649"

# DEM 参数
DEM_SCALE = 30  # SRTM 原生 30m

# =============================================================================
# Sentinel-2 L2A 处理
# =============================================================================

# SCL 最严格模式：仅保留 VEGETATION(4) + BARE_SOILS(5)
SCL_STRICT_KEEP = [4, 5]

# Sentinel-2 L2A 波段映射 (Harmonized 集合)
S2_BANDS_10M = ["B2", "B3", "B4", "B8"]  # Blue, Green, Red, NIR
S2_BANDS_20M = ["B5", "B6", "B7", "B8A", "B11", "B12"]  # RedEdge1-3, NIRn, SWIR1-2
S2_ALL_BANDS = S2_BANDS_10M + S2_BANDS_20M  # 10 bands total


def mask_s2_clouds_strict(image):
    """
    SCL 最严格模式：仅保留 VEGETATION(4) + BARE_SOILS(5)
    排除：NO_DATA(0), SATURATED(1), DARK(2), CLOUD_SHADOW(3),
          WATER(6), UNCLASSIFIED(7), CLOUD_MED(8), CLOUD_HIGH(9),
          THIN_CIRRUS(10), SNOW(11)
    """
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5))
    return image.updateMask(mask)


def mask_s2_clouds_standard(image):
    """
    SCL 标准模式：保留 VEGETATION(4) + BARE_SOILS(5) + WATER(6)
    排除：各种云、云影、雪、无效像素
    """
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    return image.updateMask(mask)


def prepare_s2_image(image):
    """处理单景 Sentinel-2 L2A：掩膜 + 缩放 + 选波段"""
    image = mask_s2_clouds_strict(image)

    # L2A 反射率缩放：DN / 10000 → [0, 1]
    optical = image.select(S2_ALL_BANDS).divide(10000).clamp(0, 1)

    # 保留 SCL 波段用于后续质量控制
    scl = image.select("SCL")

    return optical.addBands(scl).copyProperties(image, ["system:time_start"])


def download_sentinel2_timeseries(study_area, year, output_dir, monthly_composite=True):
    """下载 Sentinel-2 L2A 时序数据（月度中值合成）"""
    os.makedirs(output_dir, exist_ok=True)

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(study_area) \
        .filterDate(f"{year}-01-01", f"{year}-12-31") \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)) \
        .map(prepare_s2_image)

    count = s2.size().getInfo()
    print(f"Sentinel-2 {year}年 可用景数（云量<70%）: {count}")

    if monthly_composite:
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end_dt = datetime(year, month, 1) + timedelta(days=32)
            end = end_dt.replace(day=1).strftime("%Y-%m-%d")

            monthly = s2.filterDate(start, end).median().clip(study_area)

            filename = f"sentinel2_{year}_{month:02d}_composite"
            task = ee.batch.Export.image.toDrive(
                image=monthly,
                description=filename,
                folder=DRIVE_FOLDER,
                fileNamePrefix=filename,
                region=study_area,
                scale=S2_SCALE,
                crs=S2_CRS,
                maxPixels=1e10,
                fileFormat="GeoTIFF"
            )
            task.start()
            print(f" ✓ S2 {month:02d}月合成: {filename}")
    else:
        img_list = s2.toList(count)
        n = img_list.size().getInfo()
        for i in range(n):
            img = ee.Image(img_list.get(i))
            date_str = ee.Date(img.get("system:time_start")).format("YYYYMMdd").getInfo()
            filename = f"sentinel2_{date_str}"

            task = ee.batch.Export.image.toDrive(
                image=img.clip(study_area),
                description=filename,
                folder=DRIVE_FOLDER,
                fileNamePrefix=filename,
                region=study_area,
                scale=S2_SCALE,
                crs=S2_CRS,
                maxPixels=1e10,
                fileFormat="GeoTIFF"
            )
            task.start()
            print(f" ✓ [{i+1}/{n}] S2 单景: {filename}")

    print(f"Sentinel-2 全部任务已提交 ({year}年)")


# =============================================================================
# Sentinel-1 GRD 处理
# =============================================================================

def prepare_s1_image(image):
    """处理 Sentinel-1 GRD：dB 转换 + 计算比值"""
    vv = image.select("VV")
    vh = image.select("VH")

    # 线性 → dB
    vv_db = ee.Image(10).multiply(vv.divide(ee.Image(10).pow(10)).log10()).rename("VV")
    vh_db = ee.Image(10).multiply(vh.divide(ee.Image(10).pow(10)).log10()).rename("VH")

    # VV/VH ratio (in linear domain)
    ratio = vv.divide(vh).rename("VV_VH_Ratio")

    return vv_db.addBands(vh_db).addBands(ratio) \
        .copyProperties(image, ["system:time_start"])


def download_sentinel1_timeseries(study_area, year, output_dir, orbit_pass="DESCENDING"):
    """下载 Sentinel-1 GRD 时序数据（月度均值合成）"""
    os.makedirs(output_dir, exist_ok=True)

    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(study_area) \
        .filterDate(f"{year}-01-01", f"{year}-12-31") \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass)) \
        .map(prepare_s1_image)

    count = s1.size().getInfo()
    print(f"Sentinel-1 {year}年 可用景数 ({orbit_pass}): {count}")

    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end_dt = datetime(year, month, 1) + timedelta(days=32)
        end = end_dt.replace(day=1).strftime("%Y-%m-%d")

        monthly = s1.filterDate(start, end).mean().clip(study_area)
        filename = f"sentinel1_{year}_{month:02d}_{orbit_pass[:3].lower()}"

        task = ee.batch.Export.image.toDrive(
            image=monthly,
            description=filename,
            folder=DRIVE_FOLDER,
            fileNamePrefix=filename,
            region=study_area,
            scale=S1_SCALE,
            crs=S1_CRS,
            maxPixels=1e10,
            fileFormat="GeoTIFF"
        )
        task.start()
        print(f" ✓ S1 {month:02d}月合成: {filename}")

    print(f"Sentinel-1 全部任务已提交 ({year}年, {orbit_pass})")


# =============================================================================
# SRTM DEM 下载
# =============================================================================

def download_srtm_dem(study_area, output_dir):
    """下载 SRTM DEM 并计算坡度、坡向"""
    os.makedirs(output_dir, exist_ok=True)

    srtm = ee.Image("USGS/SRTMGL1_003")

    elevation = srtm.select("elevation").clip(study_area)
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)

    # 合并为 3 波段
    dem_combined = elevation.rename("elevation") \
        .addBands(slope.rename("slope")) \
        .addBands(aspect.rename("aspect"))

    filename = "srtm_foshan"

    task = ee.batch.Export.image.toDrive(
        image=dem_combined,
        description=filename,
        folder=DRIVE_FOLDER,
        fileNamePrefix=filename,
        region=study_area,
        scale=DEM_SCALE,
        crs=S1_CRS,
        maxPixels=1e10,
        fileFormat="GeoTIFF"
    )
    task.start()
    print(f" ✓ DEM: {filename}")
    print("SRTM DEM 任务已提交")


# =============================================================================
# 主流程
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("研究区域: 广东省佛山市")
    print(f"  经度: 112.6°E ~ 113.3°E")
    print(f"  纬度: 22.7°N ~ 23.4°N")
    print(f"  CRS: {S2_CRS}")
    print(f"  SCL 模式: 最严格 (仅 VEGETATION + BARE_SOILS)")
    print("=" * 60)

    for year in YEARS:
        print(f"\n{'='*40}")
        print(f"处理年份: {year}")
        print(f"{'='*40}")

        print(f"\n[1/3] Sentinel-2 L2A ({year})...")
        download_sentinel2_timeseries(
            study_area=FOSHAN,
            year=year,
            output_dir=f"{OUTPUT_DIR}/sentinel2",
            monthly_composite=True
        )

        print(f"\n[2/3] Sentinel-1 GRD ({year})...")
        for orbit in ["DESCENDING", "ASCENDING"]:
            try:
                download_sentinel1_timeseries(
                    study_area=FOSHAN,
                    year=year,
                    output_dir=f"{OUTPUT_DIR}/sentinel1",
                    orbit_pass=orbit
                )
            except Exception as e:
                print(f" ⚠ {orbit} 轨道失败: {e}")

    print(f"\n[3/3] SRTM DEM...")
    download_srtm_dem(
        study_area=FOSHAN,
        output_dir=f"{OUTPUT_DIR}/dem"
    )

    print("\n" + "=" * 60)
    print("全部任务已提交！")
    print(f"请访问 https://code.earthengine.google.com/tasks 查看进度")
    print(f"完成后从 Google Drive 的 '{DRIVE_FOLDER}' 文件夹下载到本地")
    print(f"  → {OUTPUT_DIR}/")
    print("=" * 60)
