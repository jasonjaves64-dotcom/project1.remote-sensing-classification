import ee
import geemap
import os
from datetime import datetime, timedelta

def initialize_gee(project_id="your-gee-project-id"):
    try:
        ee.Initialize(project=project_id)
        print(f"✓ GEE 初始化成功 (项目: {project_id})")
    except Exception as e:
        print(f"⚠ GEE 初始化失败: {e}")
        print("  请确保已安装 GEE 认证: earthengine authenticate")

def get_study_area(lon_min, lat_min, lon_max, lat_max):
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

def mask_landsat_clouds(image):
    qa = image.select("QA_PIXEL")
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0) \
        .And(qa.bitwiseAnd(1 << 4).eq(0)) \
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    return image.updateMask(cloud_mask) \
        .select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]) \
        .rename(["Blue","Green","Red","NIR","SWIR1","SWIR2"]) \
        .multiply(0.0000275).add(-0.2)

def add_vegetation_indices(image):
    ndvi = image.normalizedDifference(["NIR","Red"]).rename("NDVI")
    evi = image.expression(
        "2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)",
        {"NIR": image.select("NIR"),
        "Red": image.select("Red"),
        "Blue": image.select("Blue")}
    ).rename("EVI")
    lswi = image.normalizedDifference(["NIR","SWIR1"]).rename("LSWI")
    ndwi = image.normalizedDifference(["Green","NIR"]).rename("NDWI")
    return image.addBands([ndvi, evi, lswi, ndwi])

def preprocess_sentinel1(image):
    vv = image.select("VV")
    vh = image.select("VH")
    ratio = vv.subtract(vh).rename("VV_VH_ratio")
    return image.select(["VV","VH"]).addBands(ratio) \
        .set("system:time_start", image.get("system:time_start"))

def download_landsat_timeseries(
    study_area,
    year: int,
    output_dir: str,
    scale: int = 30,
    max_cloud_pct: int = 20,
    monthly_composite: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    collection = l8.merge(l9) \
        .filterBounds(study_area) \
        .filterDate(f"{year}-01-01", f"{year}-12-31") \
        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_pct)) \
        .map(mask_landsat_clouds) \
        .map(add_vegetation_indices)
    
    print(f"Landsat 可用景数: {collection.size().getInfo()}")
    
    if monthly_composite:
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end_dt = datetime(year, month, 1) + timedelta(days=32)
            end = end_dt.replace(day=1).strftime("%Y-%m-%d")
            
            monthly = collection.filterDate(start, end) \
                .median() \
                .clip(study_area)
            
            band_count = monthly.bandNames().size().getInfo()
            if band_count == 0:
                print(f" {month}月：无有效数据，跳过")
                continue
            
            filename = f"landsat_{year}_{month:02d}_composite"
            task = ee.batch.Export.image.toDrive(
                image = monthly,
                description = filename,
                folder = "CropClassification",
                fileNamePrefix = filename,
                region = study_area,
                scale = scale,
                crs = "EPSG:32650",
                maxPixels = 1e10,
                fileFormat = "GeoTIFF"
            )
            task.start()
            print(f" ✓ 已提交 {month}月合成任务: {filename}")
    else:
        img_list = collection.toList(collection.size())
        n = img_list.size().getInfo()
        for i in range(n):
            img = ee.Image(img_list.get(i))
            date = ee.Date(img.get("system:time_start")).format("YYYYMMdd").getInfo()
            filename = f"landsat_{date}"
            
            task = ee.batch.Export.image.toDrive(
                image = img.clip(study_area),
                description = filename,
                folder = "CropClassification",
                fileNamePrefix = filename,
                region = study_area,
                scale = scale,
                crs = "EPSG:32650",
                maxPixels = 1e10,
                fileFormat = "GeoTIFF"
            )
            task.start()
            print(f" ✓ [{i+1}/{n}] 提交: {filename}")
    
    print("\n所有Landsat任务已提交，请在GEE任务面板查看进度。")

def download_sentinel1_timeseries(
    study_area,
    year: int,
    output_dir: str,
    scale: int = 10,
    orbit_pass: str = "DESCENDING"
):
    os.makedirs(output_dir, exist_ok=True)
    
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(study_area) \
        .filterDate(f"{year}-01-01", f"{year}-12-31") \
        .filter(ee.Filter.eq("instrumentMode", "IW")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass)) \
        .map(preprocess_sentinel1)
    
    print(f"Sentinel-1 可用景数: {s1.size().getInfo()}")
    
    for month in range(1, 13):
        start = f"{year}-{month:02d}-01"
        end_dt = datetime(year, month, 1) + timedelta(days=32)
        end = end_dt.replace(day=1).strftime("%Y-%m-%d")
        
        monthly_sar = s1.filterDate(start, end).mean().clip(study_area)
        filename = f"sentinel1_{year}_{month:02d}_{orbit_pass[:3].lower()}"
        
        task = ee.batch.Export.image.toDrive(
            image = monthly_sar,
            description = filename,
            folder = "CropClassification",
            fileNamePrefix = filename,
            region = study_area,
            scale = scale,
            crs = "EPSG:32650",
            maxPixels = 1e10,
            fileFormat = "GeoTIFF"
        )
        task.start()
        print(f" ✓ 已提交SAR {month}月合成任务: {filename}")
    
    print("\n所有Sentinel-1任务已提交。")

def download_crop_mask(study_area, year: int):
    lc = ee.ImageCollection("MODIS/061/MCD12Q1") \
        .filterDate(f"{year}-01-01", f"{year}-12-31") \
        .first() \
        .select("LC_Type1") \
        .clip(study_area)
    
    crop_mask = lc.eq(12).Or(lc.eq(14)).rename("crop_mask")
    
    task = ee.batch.Export.image.toDrive(
        image = crop_mask,
        description = f"crop_mask_{year}_modis",
        folder = "CropClassification",
        region = study_area,
        scale = 500,
        crs = "EPSG:32650",
        maxPixels = 1e10,
        fileFormat = "GeoTIFF"
    )
    task.start()
    print("✓ 已提交MODIS作物掩膜下载任务")

if __name__ == "__main__":
    initialize_gee()
    
    STUDY_AREA = get_study_area(
        lon_min=115.0, lat_min=36.0,
        lon_max=117.0, lat_max=38.0
    )
    
    YEAR = 2023
    OUTPUT_DIR = "./raw_data"
    
    print("=" * 60)
    print(f"研究区域: {STUDY_AREA.getInfo()['coordinates']}")
    print(f"目标年份: {YEAR}")
    print("=" * 60)
    
    print("\n[1/3] 开始提交 Landsat 8/9 下载任务...")
    download_landsat_timeseries(
        study_area = STUDY_AREA,
        year = YEAR,
        output_dir = f"{OUTPUT_DIR}/landsat",
        monthly_composite = True
    )
    
    print("\n[2/3] 开始提交 Sentinel-1 SAR 下载任务...")
    download_sentinel1_timeseries(
        study_area = STUDY_AREA,
        year = YEAR,
        output_dir = f"{OUTPUT_DIR}/sentinel1"
    )
    
    print("\n[3/3] 开始提交作物掩膜下载任务...")
    download_crop_mask(STUDY_AREA, YEAR)
    
    print("\n 全部任务已提交！")
    print("请访问 `https://code.earthengine.google.com/tasks` 查看进度")
    print("完成后从 Google Drive 下载文件到 ./raw_data 目录")