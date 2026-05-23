# FusionCropNetV5 API Documentation

## Overview

FusionCropNetV5 is a multi-modal temporal fusion network designed for crop classification using remote sensing imagery. This API provides endpoints for model inference, health checking, and performance monitoring.

## API Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check if the API is running

**Response**:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "model_version": "v5.0",
    "uptime_seconds": 7200
}
```

### 2. Model Inference

**Endpoint**: `POST /predict`

**Description**: Perform crop classification on remote sensing data

**Request Body**:
```json
{
    "opt_seq": [[[[...]]]],
    "sar_seq": [[[[...]]]],
    "dem": [[[...]]],
    "doy": [...],
    "cloud_mask": [[[...]]],
    "valid_count": [[...]]
}
```

**Parameters**:
| Parameter | Type | Shape | Description | Required |
|-----------|------|-------|-------------|----------|
| `opt_seq` | float32 | [T, opt_ch, H, W] | Optical image sequence | Yes |
| `sar_seq` | float32 | [T, sar_ch, H, W] | SAR image sequence | Yes |
| `dem` | float32 | [dem_ch, H, W] | DEM features | Yes |
| `doy` | float32 | [T] | Day of year normalized [0,1] | Yes |
| `cloud_mask` | float32 | [T, H, W] | Cloud mask (0=clear, 1=cloudy) | No |
| `valid_count` | int32 | [H, W] | Valid observations per pixel | No |

**Response**:
```json
{
    "success": true,
    "predictions": [[...]],
    "class_probs": [[[[...]]]],
    "inference_time_ms": 45.5,
    "request_id": "abc123"
}
```

**Example cURL**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "opt_seq": [...],
    "sar_seq": [...],
    "dem": [...],
    "doy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
  }'
```

### 3. Batch Inference

**Endpoint**: `POST /predict_batch`

**Description**: Perform batch crop classification

**Request Body**:
```json
{
    "batch": [
        {
            "opt_seq": [[[[...]]]],
            "sar_seq": [[[[...]]]],
            "dem": [[[...]]],
            "doy": [...]
        }
    ]
}
```

**Response**:
```json
{
    "success": true,
    "results": [
        {
            "predictions": [[...]],
            "inference_time_ms": 42.3
        }
    ],
    "total_inference_time_ms": 156.8
}
```

### 4. Performance Metrics

**Endpoint**: `GET /metrics`

**Description**: Get real-time performance metrics

**Response**:
```json
{
    "total_requests": 1500,
    "success_rate": 99.2,
    "avg_inference_time_ms": 45.5,
    "min_inference_time_ms": 23.1,
    "max_inference_time_ms": 120.8,
    "request_counts": {"/predict": 1400, "/health": 100},
    "error_counts": {400: 5, 500: 3},
    "uptime_seconds": 7200,
    "uptime_formatted": "2h 0m 0s",
    "active_requests": 2,
    "max_concurrent_requests": 15
}
```

### 5. Model Info

**Endpoint**: `GET /model_info`

**Description**: Get model configuration and statistics

**Response**:
```json
{
    "model_name": "FusionCropNetV5",
    "version": "v5.0",
    "num_classes": 7,
    "input_specs": {
        "opt_ch": 10,
        "sar_ch": 5,
        "dem_ch": 5,
        "num_timesteps": 12
    },
    "parameters": 74243228,
    "backbone": "resnet18",
    "features": ["multi-modal", "temporal", "cross-modal-attention", "film-modulation"]
}
```

## Classification Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | Background | Non-crop areas |
| 1 | Winter Wheat | 冬小麦 |
| 2 | Summer Corn | 夏玉米 |
| 3 | Rice | 水稻 |
| 4 | Soybean | 大豆 |
| 5 | Cotton | 棉花 |
| 6 | Other | Other crops |

## Data Preprocessing Guidelines

### Optical Images
- **Channels**: 10 (B, G, R, NIR, SWIR1, SWIR2, NDVI, EVI, SAVI, LAI)
- **Normalization**: Divide by 10000
- **Shape**: [T, 10, H, W]

### SAR Images
- **Channels**: 5 (VV, VH, HV, HH, Ratio)
- **Processing**: Convert to dB scale
- **Normalization**: Scale to [-25, 5]
- **Shape**: [T, 5, H, W]

### DEM Features
- **Channels**: 5 (elevation, slope, aspect_cos, aspect_sin, TWI)
- **Normalization**: 
  - Elevation: [0, 2000]
  - Slope: [0, 90]
  - Aspect: [-1, 1]
  - TWI: [0, 20]
- **Shape**: [5, H, W]

### DOY (Day of Year)
- **Normalization**: DOY / 365
- **Range**: [0, 1]
- **Shape**: [T]

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input format |
| 401 | Unauthorized - Authentication required |
| 404 | Not Found - Endpoint does not exist |
| 500 | Internal Server Error - Model inference failed |
| 503 | Service Unavailable - Model not loaded |

## Rate Limiting

- **Requests per minute**: 60
- **Burst limit**: 10 requests

## Swagger UI

Interactive API documentation is available at:
- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/redoc` - ReDoc

## Python Client Example

```python
import requests
import numpy as np

# Prepare data
opt_seq = np.random.randn(12, 10, 64, 64).astype(np.float32)
sar_seq = np.random.randn(12, 5, 64, 64).astype(np.float32)
dem = np.random.randn(5, 64, 64).astype(np.float32)
doy = np.linspace(0.1, 0.99, 12).astype(np.float32)

# Create request
data = {
    "opt_seq": opt_seq.tolist(),
    "sar_seq": sar_seq.tolist(),
    "dem": dem.tolist(),
    "doy": doy.tolist()
}

# Send request
response = requests.post(
    "http://localhost:8000/predict",
    json=data
)

# Process response
result = response.json()
predictions = np.array(result["predictions"])
print(f"Predicted class map shape: {predictions.shape}")
```

## Deployment

### Requirements
- Python 3.8+
- PyTorch 2.0+
- FastAPI
- uvicorn

### Running the API

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
python api/main.py

# Or with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model checkpoint | `checkpoints/best_phase2.pth` |
| `DEVICE` | Device to use (cpu/cuda) | `cuda` |
| `MAX_BATCH_SIZE` | Maximum batch size | `8` |
| `LOG_LEVEL` | Log level | `INFO` |

## License

This project is licensed under the MIT License.
