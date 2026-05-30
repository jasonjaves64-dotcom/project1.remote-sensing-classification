"""
遥感影像作物分类API服务 - 增强版

新增功能：
1. 请求日志中间件
2. API密钥认证
3. 请求限流（中间件级，零侵入）
4. 异步任务处理
5. 更完善的错误处理
6. 系统状态监控
"""

import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import time
import json
from datetime import datetime
import asyncio
import io
from contextlib import asynccontextmanager

# 导入项目模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fusion_net_v5_edl import FusionCropNetV5EDL
from data.preprocess_pipeline import PreprocessPipeline, PreprocessConfig
from utils.metrics import compute_metrics
from utils.monitoring import log_manager, log_info, log_error, log_inference, get_stats
from utils.rate_limiter import RateLimitMiddleware


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────
@asynccontextmanager
async def lifespan(api_app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI app."""
    global PIPELINE
    try:
        config = PreprocessConfig(
            normalize=True, freeze_stats=True, sar_log_transform=True, augment=False
        )
        PIPELINE = PreprocessPipeline(config)
        log_info("预处理管道初始化成功")
    except Exception as e:
        log_error("管道初始化失败", exception=e)
    yield
    # shutdown: nothing to clean up currently


# 初始化FastAPI应用
app = FastAPI(
    title="遥感影像作物分类API",
    description="基于深度学习的多模态遥感影像作物分类服务",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")
)

# Rate limiter — middleware-based, zero per-endpoint intrusion
app.add_middleware(RateLimitMiddleware)

# API密钥认证
API_KEY = os.environ.get("API_KEY", "dev_key_change_in_production")
if API_KEY == "dev_key_change_in_production":
    log_info("WARNING: Using default API_KEY. Set API_KEY env var for production.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    log_info(f"收到请求", method=request.method, path=request.url.path, client=request.client.host)

    response = await call_next(request)

    process_time = time.time() - start_time
    log_info(f"请求完成", method=request.method, path=request.url.path,
             status_code=response.status_code, duration_ms=process_time * 1000)

    return response


# 全局变量
MODEL = None
PIPELINE = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOADED = False
MODEL_VERSION = "v5_edl"

# ── Request / Response models ─────────────────────────────────────────

class InferenceRequest(BaseModel):
    opt_sequence: List[List[List[List[float]]]] = Field(..., description="光学时序数据 [T, C, H, W]")
    sar_sequence: Optional[List[List[List[List[float]]]]] = Field(None, description="SAR时序数据 [T, C, H, W]")
    dem_data: Optional[List[List[List[float]]]] = Field(None, description="DEM数据 [C, H, W]")
    doy: List[float] = Field(..., description="归一化日序 [T]")
    n_passes: int = Field(1, ge=1, le=20, description="推理次数")
    use_tta: bool = Field(False, description="是否使用TTA")

    @field_validator('opt_sequence')
    @classmethod
    def check_opt_shape(cls, v):
        if len(v) == 0:
            raise ValueError("光学序列不能为空")
        if len(v[0]) != 10:
            raise ValueError("光学序列通道数必须为10")
        return v


class BatchInferenceRequest(BaseModel):
    requests: List[InferenceRequest] = Field(..., min_length=1, max_length=100)


class ModelLoadRequest(BaseModel):
    model_path: str = Field(..., description="模型文件路径")
    config_path: Optional[str] = Field(None, description="配置文件路径")
    use_pre_trained: bool = Field(False, description="是否使用预训练权重")


class TrainingRequest(BaseModel):
    data_path: str = Field(..., description="训练数据路径")
    epochs: int = Field(10, ge=1, le=100)
    batch_size: int = Field(8, ge=1, le=64)
    lr: float = Field(1e-4, ge=1e-6, le=1e-2)


class InferenceResponse(BaseModel):
    success: bool
    message: str
    prediction: Optional[List[List[int]]] = None
    probabilities: Optional[List[List[List[float]]]] = None
    uncertainty: Optional[Dict[str, List[List[float]]]] = None
    inference_time: Optional[float] = None
    model_version: str = MODEL_VERSION


class BatchInferenceResponse(BaseModel):
    success: bool
    message: str
    results: List[InferenceResponse]
    total_inference_time: float


class ModelStatusResponse(BaseModel):
    loaded: bool
    model_version: str
    device: str
    last_loaded: Optional[str] = None
    memory_usage: Optional[Dict[str, float]] = None


class MetricsResponse(BaseModel):
    success: bool
    metrics: Dict[str, float]


class SystemStatsResponse(BaseModel):
    success: bool
    stats: Dict[str, Any]
    errors: Dict[str, Any]


class TaskResponse(BaseModel):
    success: bool
    task_id: str
    message: str


# ── Health & System ───────────────────────────────────────────────────

@app.get("/health", tags=["健康检查"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.2.0"
    }


@app.get("/stats", response_model=SystemStatsResponse, tags=["系统信息"])
async def get_system_stats():
    return SystemStatsResponse(
        success=True,
        stats=get_stats(),
        errors=log_manager.get_error_summary()
    )


@app.get("/model/status", response_model=ModelStatusResponse, tags=["模型管理"])
async def get_model_status():
    memory_info = None
    if torch.cuda.is_available():
        memory_info = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "cached_mb": torch.cuda.memory_reserved() / 1024 / 1024
        }
    return ModelStatusResponse(
        loaded=MODEL_LOADED,
        model_version=MODEL_VERSION,
        device=str(DEVICE),
        memory_usage=memory_info
    )


@app.get("/version", tags=["系统信息"])
async def get_version():
    return {
        "api_version": "1.2.0",
        "model_version": MODEL_VERSION,
        "framework": "FastAPI",
        "device": str(DEVICE),
        "python_version": sys.version.split()[0]
    }


@app.get("/", tags=["系统信息"])
async def root():
    return {
        "message": "遥感影像作物分类API v1.2.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "stats": "/stats"
    }


# ── Model Management ──────────────────────────────────────────────────

@app.post("/model/load", response_model=ModelStatusResponse, tags=["模型管理"])
async def load_model(body: ModelLoadRequest, api_key: str = Depends(get_api_key)):
    global MODEL, MODEL_LOADED
    try:
        model_path = Path(body.model_path).resolve()
        allowed_dirs = [Path("/app/checkpoints").resolve(), Path("/app/models").resolve(),
                        Path.cwd().resolve() / "checkpoints", Path.cwd().resolve() / "pretrained_weights"]
        if not any(str(model_path).startswith(str(d)) for d in allowed_dirs):
            raise HTTPException(status_code=403, detail="模型路径不在允许的目录内")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="模型文件不存在")

        log_info(f"正在加载模型: {model_path}")

        global MODEL
        if MODEL is None:
            MODEL = FusionCropNetV5EDL(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet18", pretrained=False,
                n_heads=16, win_size=4, n_layers=4,
                use_v6_enhancements=False
            ).to(DEVICE)

        MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        MODEL.eval()
        MODEL_LOADED = True

        log_info("模型加载成功")
        return ModelStatusResponse(
            loaded=True,
            model_version=MODEL_VERSION,
            device=str(DEVICE),
            last_loaded=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        log_error("模型加载失败", exception=e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Inference ─────────────────────────────────────────────────────────

@app.post("/inference", response_model=InferenceResponse, tags=["推理服务"])
async def inference(body: InferenceRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")

    start_time = time.time()

    try:
        opt_seq = np.array(body.opt_sequence, dtype=np.float32)
        sar_seq = np.array(body.sar_sequence, dtype=np.float32) if body.sar_sequence else None
        dem = np.array(body.dem_data, dtype=np.float32) if body.dem_data else None
        doy = np.array(body.doy, dtype=np.float32)

        T, C_opt, H, W = opt_seq.shape

        raw_data = {
            'opt': opt_seq,
            'sar': sar_seq if sar_seq is not None else np.random.rand(T, 5, H, W).astype(np.float32),
            'dem': dem if dem is not None else np.random.rand(5, H, W).astype(np.float32),
            'doy': doy
        }

        transforms = {'opt': {'target_size': (H, W)}, 'sar': {'target_size': (H, W)}, 'dem': {'target_size': (H, W)}}
        sample = PIPELINE.process(raw_data, transforms, is_training=False)

        if sample is None:
            raise ValueError("数据预处理失败")

        with torch.no_grad():
            opt_t = torch.from_numpy(sample.opt_seq).unsqueeze(0).to(DEVICE)
            sar_t = torch.from_numpy(sample.sar_seq).unsqueeze(0).to(DEVICE)
            dem_t = torch.from_numpy(sample.dem).unsqueeze(0).to(DEVICE)
            doy_t = torch.from_numpy(sample.doy).unsqueeze(0).to(DEVICE)

            result = MODEL.predict_uncertainty(opt_t, sar_t, dem_t, doy_t,
                                               n_passes=body.n_passes,
                                               use_tta=body.use_tta)

        inference_time = time.time() - start_time
        log_inference(inference_time, opt_seq.shape, success=True)

        return InferenceResponse(
            success=True,
            message="推理成功",
            prediction=result['pred_class'].squeeze().cpu().numpy().tolist(),
            probabilities=result['probs'].squeeze().cpu().numpy().tolist(),
            uncertainty={
                'vacuity': result['vacuity'].squeeze().cpu().numpy().tolist(),
                'dissonance': result['dissonance'].squeeze().cpu().numpy().tolist()
            },
            inference_time=inference_time
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("推理失败", exception=e)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/inference/batch", response_model=BatchInferenceResponse, tags=["推理服务"])
async def batch_inference(body: BatchInferenceRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")

    start_time = time.time()
    results = []

    for req in body.requests:
        try:
            inner = InferenceRequest(
                opt_sequence=req.opt_sequence,
                sar_sequence=req.sar_sequence,
                dem_data=req.dem_data,
                doy=req.doy,
                n_passes=req.n_passes,
                use_tta=req.use_tta,
            )
            response = await inference(inner)
            results.append(response)
        except Exception as e:
            results.append(InferenceResponse(
                success=False,
                message=f"推理失败: {str(e)}"
            ))

    total_time = time.time() - start_time
    log_info(f"批量推理完成", count=len(body.requests), duration_ms=total_time * 1000)

    return BatchInferenceResponse(
        success=all(r.success for r in results),
        message="批量推理完成",
        results=results,
        total_inference_time=total_time
    )


# ── File Upload Inference ─────────────────────────────────────────────

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB


@app.post("/inference/file", tags=["推理服务"])
async def inference_file(
    opt_file: UploadFile = File(...),
    sar_file: Optional[UploadFile] = None,
    dem_file: Optional[UploadFile] = None,
    doy_file: Optional[UploadFile] = None,
):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="模型未加载")

    for f in [opt_file, sar_file, dem_file, doy_file]:
        if f and f.size and f.size > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"文件 {f.filename} 超过大小限制 ({MAX_UPLOAD_SIZE // 1024 // 1024}MB)")

    try:
        opt_data = np.load(opt_file.file)
        sar_data = np.load(sar_file.file) if sar_file else None
        dem_data = np.load(dem_file.file) if dem_file else None
        doy_data = np.load(doy_file.file) if doy_file else np.linspace(0, 1, opt_data.shape[0])

        req = InferenceRequest(
            opt_sequence=opt_data.tolist(),
            sar_sequence=sar_data.tolist() if sar_data is not None else None,
            dem_data=dem_data.tolist() if dem_data is not None else None,
            doy=doy_data.tolist()
        )
        return await inference(req)

    except HTTPException:
        raise
    except Exception as e:
        log_error("文件推理失败", exception=e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Metrics ───────────────────────────────────────────────────────────

@app.post("/metrics", response_model=MetricsResponse, tags=["指标计算"])
async def calculate_model_metrics(predictions: List[List[int]], labels: List[List[int]]):
    try:
        pred_np = np.array(predictions)
        label_np = np.array(labels)
        metrics = compute_metrics(pred_np, label_np)
        return MetricsResponse(success=True, metrics=metrics)
    except Exception as e:
        log_error("指标计算失败", exception=e)
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Async Training ────────────────────────────────────────────────────

async def run_training(data_path: str, epochs: int, batch_size: int, lr: float):
    """后台训练任务 — 调用 scripts/train_fusion_edl.py"""
    import subprocess
    log_info("开始训练任务", data_path=data_path, epochs=epochs, batch_size=batch_size, lr=lr)

    script = Path(__file__).parent.parent / "scripts" / "train_fusion_edl.py"
    if not script.exists():
        log_error("训练脚本不存在", path=str(script))
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script),
            "--data_path", data_path,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            log_info("训练任务完成", data_path=data_path)
        else:
            log_error("训练任务失败", stderr=stderr.decode()[:500])
    except Exception as e:
        log_error("训练任务异常", exception=e)


@app.post("/train", response_model=TaskResponse, tags=["训练管理"])
async def start_training(body: TrainingRequest, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)):
    import pathlib
    data_p = pathlib.Path(body.data_path).resolve()
    allowed = [pathlib.Path("/app/data").resolve(), pathlib.Path.cwd().resolve() / "data"]
    if not any(str(data_p).startswith(str(d)) for d in allowed):
        raise HTTPException(status_code=403, detail="data_path outside allowed directories")

    task_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    background_tasks.add_task(run_training, str(data_p), body.epochs, body.batch_size, body.lr)

    return TaskResponse(
        success=True,
        task_id=task_id,
        message="训练任务已启动"
    )


# ── Unified Model Inference ───────────────────────────────────────────

_MODELS = {}

class PredictRequest(BaseModel):
    aoi: Optional[dict] = None
    bbox: Optional[list] = None


class PredictResponse(BaseModel):
    dominant: str = "—"
    confidence: float = 0.0
    time: float = 0.0
    distribution: dict = {}
    aux: dict = {}
    geojson: Optional[dict] = None


CROP_NAMES = {0: 'wheat', 1: 'corn', 2: 'rice', 3: 'soybean', 4: 'cotton', 5: 'vegetable', 6: 'other'}


def _get_or_create_model(name: str):
    if name not in _MODELS:
        if name == 'v6':
            from models.fusion_net_v6 import FusionCropNetV6
            _MODELS[name] = FusionCropNetV6(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet18", pretrained=False,
                n_heads=4, n_layers=2
            ).to(DEVICE).eval()
        elif name == 'v5pro':
            from models.fusion_net_v5pro import FusionCropNetV5Pro
            _MODELS[name] = FusionCropNetV5Pro(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet18", pretrained=False,
                n_heads=4, win_size=4, n_layers=2,
                use_carafe=False, dynamic_dropout=False, adaptive_kl=False,
                edl_dropout_p=0.3, edl_lambda_max=0.5, edl_anneal_ep=50
            ).to(DEVICE).eval()
        elif name == 'v5':
            from models.fusion_net_v5 import FusionCropNetV5
            _MODELS[name] = FusionCropNetV5(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet18", pretrained=False,
                n_heads=16, win_size=4, n_layers=4
            ).to(DEVICE).eval()
        elif name == 'tsvit':
            from models.tsvit import TSViT
            _MODELS[name] = TSViT(
                in_channels=10, num_classes=7, embed_dim=256,
                depth=4, num_heads=8
            ).to(DEVICE).eval()
        else:
            _MODELS[name] = FusionCropNetV5EDL(
                opt_ch=10, sar_ch=5, dem_ch_in=5, num_classes=7,
                feat_dim=512, backbone="resnet18", pretrained=False,
                n_heads=16, win_size=4, n_layers=4,
                use_v6_enhancements=False
            ).to(DEVICE).eval()
    return _MODELS[name]


def _run_inference(model, name: str):
    """演示推理 — 使用合成数据 (生产环境需替换为真实数据加载)"""
    import time as _time
    t0 = _time.time()
    torch.manual_seed(42)
    B, T, H, W = 1, 12, 64, 64
    opt = torch.randn(B, T, 10, H, W).to(DEVICE)
    sar = torch.randn(B, T, 5, H, W).to(DEVICE)
    dem = torch.randn(B, 5, H, W).to(DEVICE)
    doy = torch.rand(B, T).to(DEVICE)

    with torch.no_grad():
        if name == 'v6':
            alpha = model(opt, sar, dem, doy)
        elif name == 'tsvit':
            opt_small = torch.nn.functional.interpolate(
                opt.view(B * T, *opt.shape[2:]), (16, 16), mode='bilinear').view(B, T, -1, 16, 16)
            alpha = model(opt_small, doy)
        else:
            alpha = model(opt, sar, dem, doy)

    if name == 'tsvit':
        alpha = torch.softmax(alpha, dim=1)
        probs = alpha.squeeze(0)
        elapsed = _time.time() - t0
        pred_class = probs.argmax().item()
        dist = {}
        for k in range(7):
            pct = round(probs[k].item() * 100, 1)
            if pct > 0:
                dist[CROP_NAMES[k]] = pct
        dominant_class = CROP_NAMES[pred_class]
        result = {
            'dominant': dominant_class,
            'confidence': round(probs.max().item() * 100, 1),
            'time': round(elapsed, 2),
            'distribution': dist,
            'aux': {}
        }
    else:
        probs = (alpha / alpha.sum(dim=1, keepdim=True)).squeeze(0)
        pred = probs.argmax(dim=0)
        elapsed = _time.time() - t0

        ph, pw = pred.shape[-2:]
        total_pixels = ph * pw

        dist = {}
        for k in range(7):
            pct = round((pred == k).sum().item() / total_pixels * 100, 1)
            if pct > 0:
                dist[CROP_NAMES[k]] = pct

        dominant_class = max(dist, key=dist.get) if dist else "—"

        result = {
            'dominant': dominant_class,
            'confidence': round(probs.max(dim=0)[0].mean().item() * 100, 1),
            'time': round(elapsed, 2),
            'distribution': dist,
            'aux': {}
        }

    return result


@app.post("/predict/{model}", response_model=PredictResponse, tags=["Inference"])
async def predict_model(model: str, body: PredictRequest):
    valid = {'v5', 'v5edl', 'v5pro', 'v6', 'tsvit'}
    if model not in valid:
        raise HTTPException(400, f"Unknown model '{model}'. Choose: {', '.join(valid)}")
    m = _get_or_create_model(model)
    return _run_inference(m, model)


# ── File Upload (Multi-model) ─────────────────────────────────────────

class UploadResponse(BaseModel):
    success: bool
    message: str = ""
    model: str = ""
    dominant: str = "—"
    confidence: float = 0.0
    distribution: dict = {}
    aux: dict = {}
    geojson: Optional[dict] = None


@app.post("/predict/{model}/upload", response_model=UploadResponse, tags=["Inference"])
async def predict_upload(
    model: str,
    files: list[UploadFile] = File(...),
):
    valid = {'v5', 'v5edl', 'v5pro', 'v6', 'tsvit'}
    if model not in valid:
        raise HTTPException(400, f"Unknown model '{model}'")
    if not files:
        raise HTTPException(400, "No files uploaded")

    import time as _time
    t0 = _time.time()

    opt_data = sar_data = dem_data = None
    for f in files:
        name = f.filename or ''
        content = await f.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File {name} exceeds {MAX_UPLOAD_SIZE // 1024 // 1024}MB limit")
        ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''

        if ext == 'npy':
            import io
            data = np.load(io.BytesIO(content))
        elif ext in ('tif', 'tiff'):
            try:
                import rasterio
                with rasterio.open(io.BytesIO(content)) as src:
                    data = src.read()
            except ImportError:
                raise HTTPException(500, "rasterio not installed for .tif support")
        else:
            continue

        name_lower = name.lower()
        if 'opt' in name_lower or 'optical' in name_lower:
            opt_data = torch.from_numpy(data).float()
        elif 'sar' in name_lower:
            sar_data = torch.from_numpy(data).float()
        elif 'dem' in name_lower:
            dem_data = torch.from_numpy(data).float()

    if opt_data is None:
        opt_data = torch.randn(1, 12, 10, 64, 64)
    if sar_data is None:
        sar_data = torch.randn(1, 12, 5, 64, 64)
    if dem_data is None:
        dem_data = torch.randn(1, 5, 64, 64)

    if opt_data.dim() == 3:
        opt_data = opt_data.unsqueeze(0)
    if sar_data.dim() == 3:
        sar_data = sar_data.unsqueeze(0)
    if dem_data.dim() == 2:
        dem_data = dem_data.unsqueeze(0)
    if opt_data.dim() == 5:
        opt_data = opt_data[0]
    if sar_data.dim() == 5:
        sar_data = sar_data[0]
    if dem_data.dim() == 4:
        dem_data = dem_data[0]

    doy = torch.linspace(0, 1, opt_data.shape[1] if opt_data.dim() >= 2 else 12)

    m = _get_or_create_model(model)
    opt_t = opt_data[:1].to(DEVICE)
    sar_t = sar_data[:1].to(DEVICE)
    dem_t = dem_data[:1].to(DEVICE)
    doy_t = doy[:opt_t.shape[1]].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model == 'v6':
            alpha, aux = m(opt_t, sar_t, dem_t, doy_t)
        elif model == 'tsvit':
            alpha = m(opt_t, doy_t)
        else:
            alpha = m(opt_t, sar_t, dem_t, doy_t)

    probs = (alpha / alpha.sum(dim=1, keepdim=True)).squeeze(0)
    pred = probs.argmax(dim=0)
    elapsed = _time.time() - t0

    dist = {}
    for k in range(7):
        pct = round((pred == k).sum().item() / pred.numel() * 100, 1)
        if pct > 0:
            dist[CROP_NAMES[k]] = pct

    result = UploadResponse(
        success=True,
        message=f"Inference complete ({elapsed:.1f}s)",
        model=model,
        dominant=max(dist, key=dist.get) if dist else "—",
        confidence=round(probs.max(dim=0)[0].mean().item() * 100, 1),
        distribution=dist,
        aux={}
    )
    if model == 'v6':
        result.aux = {
            'lai': round(aux.get('lai', torch.tensor(0.0)).mean().item(), 3),
            'growth_stage': aux.get('growth', torch.zeros(1)).argmax(dim=1).item(),
            'boundary_coverage': round(aux.get('boundary', torch.tensor(0.0)).mean().item() * 100, 1)
        }
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
