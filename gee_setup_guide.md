# GEE 环境配置指南

## 1. 前置要求

- Python 版本 >= 3.8
- Google 账号（需要注册 GEE 账户）
- Google Cloud Project（用于批量导出）

## 2. 安装依赖

```bash
pip install earthengine-api geemap numpy pandas
```

## 3. 认证方式选择

### 方式一：本地认证（推荐用于开发）

```bash
earthengine authenticate
```

运行后会打开浏览器，登录你的 Google 账号并授权，然后复制授权码到终端。

### 方式二：服务账号认证（用于服务器/云端）

1. 在 [Google Cloud Console](https://console.cloud.google.com/) 创建服务账号
2. 为服务账号分配 Earth Engine 权限
3. 下载密钥文件（JSON格式）
4. 设置环境变量：

```bash
# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your\service-account-key.json"
```

### 方式三：使用 geemap 交互式认证

```python
import geemap
Map = geemap.Map()  # 会自动弹出认证窗口
```

## 4. 配置项目

修改 `config.yaml` 中的 GEE 配置：

```yaml
gee:
  project_id: your-gee-project-id
  study_area:
    lon_min: 115.0
    lat_min: 36.0
    lon_max: 117.0
    lat_max: 38.0
  year: 2023
  crs: EPSG:32650
```

**参数说明**：
- `project_id`: 你的 Google Cloud 项目 ID
- `study_area`: 研究区域的经纬度范围（左下角和右上角）
- `year`: 目标年份
- `crs`: 输出投影（建议使用 UTM 投影）

## 5. 测试连接

```python
import ee

# 方式1：使用默认认证
ee.Initialize()

# 方式2：指定项目（推荐）
ee.Initialize(project="your-project-id")

# 方式3：使用服务账号
# service_account = 'your-service-account@your-project.iam.gserviceaccount.com'
# credentials = ee.ServiceAccountCredentials(service_account, 'path/to/key.json')
# ee.Initialize(credentials, project="your-project-id")

print("✅ GEE 连接成功！")
print(f"当前用户: {ee.ServiceAccounts().list().getInfo()}")
```

## 6. 使用数据下载脚本

项目提供了完整的数据下载脚本 `scripts/gee_download.py`：

```bash
python scripts/gee_download.py
```

### 脚本功能

| 功能 | 说明 |
|------|------|
| `download_landsat_timeseries` | 下载 Landsat 8/9 时序数据（含云掩膜和植被指数） |
| `download_sentinel1_timeseries` | 下载 Sentinel-1 SAR 时序数据 |
| `download_crop_mask` | 下载 MODIS 作物掩膜 |

### 自定义参数

```python
# 修改下载区域和年份
STUDY_AREA = get_study_area(
    lon_min=115.0, lat_min=36.0,
    lon_max=117.0, lat_max=38.0
)
YEAR = 2023
```

## 7. 数据预处理流程

下载完成后，数据需要进行预处理：

```bash
# 1. 从 Google Drive 下载文件到本地
# 2. 运行预处理脚本
python scripts/preprocess_data.py
```

预处理步骤包括：
- 几何校正和配准
- 辐射校正
- 时间序列对齐
- 归一化处理
- 生成训练样本

## 8. 常见问题

### 认证问题

**Q: `earthengine authenticate` 命令找不到？**
```bash
# 确保 earthengine-api 已正确安装
pip install --upgrade earthengine-api
```

**Q: 认证超时或失败？**
- 确保网络可以访问 Google 服务
- 尝试使用 VPN 或代理
- 重新运行 `earthengine authenticate`

**Q: 服务账号权限不足？**
- 在 [GEE Console](https://code.earthengine.google.com/) 添加服务账号为项目成员
- 确保服务账号拥有 `Earth Engine Editor` 权限

### 导出问题

**Q: 任务卡在 `READY` 状态？**
- 检查 Google Cloud 项目是否启用了 Earth Engine API
- 确保账户有足够的配额

**Q: 导出文件过大导致失败？**
- 减小导出区域
- 使用更高的分辨率（scale 参数）
- 分块导出

### 配额问题

**Q: 提示配额不足？**
- 等待配额重置（通常每天重置）
- 在 Google Cloud Console 申请提高配额
- 优化代码减少 API 调用次数

## 9. 资源链接

- [GEE Python API 文档](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api)
- [geemap 文档](https://geemap.org/)
- [GEE 数据集目录](https://developers.google.com/earth-engine/datasets/catalog)
- [Google Cloud Console](https://console.cloud.google.com/)
- [GEE 代码编辑器](https://code.earthengine.google.com/)

## 10. 注意事项

1. **数据使用合规**：确保遵守各数据集的使用许可
2. **配额管理**：合理安排批量任务，避免超出配额
3. **数据存储**：定期清理不必要的临时文件
4. **版本控制**：不要将密钥文件提交到版本控制系统