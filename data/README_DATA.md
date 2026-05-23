# 数据存放说明

## 目录结构

```
data/
├── landsat_images/          # Landsat影像数据
│   ├── 2023/                # 2023年影像
│   ├── 2022/                # 2022年影像（可选）
│   └── ...
├── sentinel1_images/        # Sentinel-1 SAR数据（新增）
│   ├── 2023/
│   └── ...
├── dem/                     # DEM数字高程模型（新增）
├── labels/                  # 标签数据
├── stats/                   # 全局统计量（新增）
│   └── global_stats.json
├── preprocess.py            # 预处理脚本（保留）
├── preprocess_pipeline.py   # 完整预处理管道（新增）
└── dataset.py               # 数据集定义
```

## 影像数据格式

### Landsat 8/9 影像要求

- **格式**：GeoTIFF (.tif)
- **波段顺序**（共7个波段）：
  1. Blue (波段2)
  2. Green (波段3)
  3. Red (波段4)
  4. NIR (波段5)
  5. SWIR1 (波段6)
  6. SWIR2 (波段7)
  7. QA_PIXEL (质量评估波段，用于云检测)

- **命名规范**：文件名需包含 acquisition date
  示例：`LC08_L2SP_123032_20230515_20230523_02_T1.tif`
  其中 `20230515` 表示影像获取日期（2023年5月15日）

- **存储位置**：按年份存放，如 `data/landsat_images/2023/`

### Sentinel-1 SAR 影像要求（新增）

- **格式**：GeoTIFF (.tif)
- **极化方式**：VV + VH 双极化
- **预处理级别**：GRD（Ground Range Detected）
- **命名规范**：`S1A_IW_GRDH_1SDV_20230515T053000_20230515T053025_048544_05D7B8_XXXX.tif`
- **存储位置**：`data/sentinel1_images/2023/`

### DEM 数据要求（新增）

- **格式**：GeoTIFF (.tif)
- **分辨率**：10m（与目标分辨率一致）
- **存储位置**：`data/dem/dem_10m.tif`

## 标签数据格式

- **格式**：NumPy数组 (.npy)
- **尺寸**：与影像一致 (H, W)
- **像素值**：
  - 0：背景
  - 1：冬小麦
  - 2：夏玉米
  - 3：水稻
  - 4：大豆
  - 5：棉花
  - 6：其他作物
  - 255：忽略区域（如云、水体等）
- **存储位置**：`data/labels/crop_label_2023.npy`

## 数据预处理流程（更新版）

### 完整处理流程

```
原始数据输入
    ↓
[1] 空间配准
    └─ 统一坐标系（UTM/WGS84）
    └─ 双线性插值到目标分辨率（10m）
    ↓
[2] 云检测
    └─ NDVI + NIR 光学云检测
    └─ SAR低后向散射辅助检测
    └─ 形态学后处理优化
    ↓
[3] SAR对数变换（仅SAR数据）
    └─ log(1 + x) 变换
    ↓
[4] 时序插值（带掩码更新）
    └─ 线性插值填充云遮挡区域
    └─ 标记所有插值位置为无效
    └─ 屏蔽长时间空缺（>60天）
    ↓
[5] 数据增强（空间变换，在归一化之前）
    └─ 水平/垂直翻转
    └─ 同步所有模态和标签
    ↓
[6] 光谱归一化（使用全局统计量）
    └─ Z-score 标准化
    └─ 统计量从训练集预计算并冻结
    ↓
[7] 质量检查
    └─ NaN/Inf 值检查
    └─ 值范围验证
    └─ 有效像素比例检查（>30%）
    ↓
输出样本
```

### 关键设计决策

1. **增强在归一化之前**：避免空间变换破坏标准化统计量
2. **云掩码跟随插值**：所有插值位置标记为无效，防止污染证据估计
3. **全局统计量**：Z-score参数从整个训练集预计算，推理时固化使用
4. **SAR对数变换**：先做 log(1+x) 变换再标准化，避免动态范围扭曲
5. **长时间空缺屏蔽**：超过阈值（默认60天）的空缺完全标记为无效

## 预处理配置参数

```yaml
preprocess:
  # 空间配准
  target_resolution: 10.0      # 目标分辨率（米）
  
  # 云检测
  cloud_threshold: 0.3         # 云检测阈值
  use_sar_for_cloud_mask: true # 使用SAR辅助检测
  
  # 时序插值
  max_gap: 30                  # 最大插值间隔（天）
  mask_long_gaps: true         # 屏蔽长时间空缺
  long_gap_threshold: 60       # 长时间空缺阈值（天）
  
  # 归一化
  normalize: true              # 是否归一化
  global_stats_path: ./stats/global_stats.json  # 全局统计量路径
  freeze_stats: true           # 推理时冻结统计量
  
  # SAR处理
  sar_log_transform: true      # SAR先做对数变换
  
  # 数据增强
  augment: true                # 是否增强
  augment_prob: 0.5            # 增强概率
```

## 全局统计量生成

### 训练阶段（生成统计量）

```python
from data.preprocess_pipeline import compute_global_stats

# 计算全局统计量
stats = compute_global_stats(
    data_paths=['./data/train_data_01.npy', './data/train_data_02.npy'],
    output_path='./data/stats/global_stats.json'
)
```

### 推理阶段（加载预计算统计量）

```python
from data.preprocess_pipeline import PreprocessPipeline, PreprocessConfig

config = PreprocessConfig(
    global_stats_path='./data/stats/global_stats.json',
    freeze_stats=True  # 冻结统计量，不再更新
)
pipeline = PreprocessPipeline(config)
```

## 快速开始

### 使用完整预处理管道

```python
from data.preprocess_pipeline import PreprocessPipeline, PreprocessConfig

# 创建配置
config = PreprocessConfig(
    target_resolution=10.0,
    cloud_threshold=0.3,
    mask_long_gaps=True,
    long_gap_threshold=60,
    normalize=True,
    global_stats_path='./data/stats/global_stats.json',
    freeze_stats=False,  # 训练阶段设为False，推理阶段设为True
    sar_log_transform=True,
    augment=True
)

# 创建管道
pipeline = PreprocessPipeline(config)

# 准备输入数据
raw_data = {
    'opt': opt_sequence,      # [T, 10, H, W] - 光学时序
    'sar': sar_sequence,      # [T, 5, H, W]  - SAR时序
    'dem': dem_data,          # [5, H, W]     - DEM数据
    'doy': doy_norm           # [T]           - 归一化日序
}

transforms = {
    'opt': {'target_size': (H, W)},
    'sar': {'target_size': (H, W)},
    'dem': {'target_size': (H, W)}
}

# 处理数据
sample = pipeline.process(
    raw_data=raw_data,
    transforms=transforms,
    label=label_data,
    is_training=True
)

# 输出样本结构
# sample.opt_seq: [T, 10, H, W]    - 预处理后的光学时序
# sample.sar_seq: [T, 5, H, W]     - 预处理后的SAR时序
# sample.dem: [5, H, W]            - 预处理后的DEM
# sample.cloud_mask: [T, H, W]     - 云掩膜（包含插值位置）
# sample.is_interpolated: [T, H, W] - 插值位置标记
# sample.valid_count: [H, W]       - 有效观测计数
```

### 使用传统预处理脚本（保留）

```python
from data.preprocess import build_time_sequence

# 构建时间序列
sequence, doy_norm = build_time_sequence(
    img_dir="./data/landsat_images/2023",
    year=2023,
    cloud_threshold=0.2
)

# sequence shape: (T, 10, H, W)
#   T: 时间步数
#   10: 6个光学波段 + 4个植被指数
# doy_norm shape: (T,)
```

## 输出样本字段说明

| 字段 | 形状 | 说明 |
|------|------|------|
| opt_seq | [T, 10, H, W] | 预处理后的光学时序（6个波段+4个植被指数） |
| sar_seq | [T, 5, H, W] | 预处理后的SAR时序（对数变换后） |
| dem | [5, H, W] | 预处理后的DEM数据 |
| doy | [T] | 归一化日序信息 |
| label | [H, W]（可选） | 标签数据 |
| cloud_mask | [T, H, W] | 云掩膜（True=无效） |
| is_interpolated | [T, H, W] | 插值位置标记（True=插值） |
| valid_count | [H, W] | 有效观测计数 |

## 注意事项

1. **数据对齐**：确保光学、SAR、DEM数据空间对齐到同一坐标系
2. **统计量一致性**：训练和推理必须使用相同的全局统计量
3. **SAR预处理**：SAR数据必须先做对数变换再归一化
4. **长时间空缺**：冬季等长时间缺失应设置合理的阈值进行屏蔽
5. **增强同步**：空间变换必须同步应用于所有模态和标签

---

*文档版本: v2.0（更新于2026年5月）*
