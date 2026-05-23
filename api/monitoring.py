"""
API 监控和日志系统

提供结构化日志、性能监控、错误追踪等功能。
"""

import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str = "api"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        file_handler = logging.FileHandler(
            LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """记录信息级别日志"""
        log_entry = self._build_log_entry(message, **kwargs)
        self.logger.info(log_entry)
    
    def warning(self, message: str, **kwargs):
        """记录警告级别日志"""
        log_entry = self._build_log_entry(message, **kwargs)
        self.logger.warning(log_entry)
    
    def error(self, message: str, **kwargs):
        """记录错误级别日志"""
        log_entry = self._build_log_entry(message, **kwargs)
        self.logger.error(log_entry)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误级别日志"""
        log_entry = self._build_log_entry(message, **kwargs)
        self.logger.critical(log_entry)
    
    def _build_log_entry(self, message: str, **kwargs) -> str:
        """构建结构化日志条目"""
        entry = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs
        }
        return json.dumps(entry, ensure_ascii=False)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "success_requests": 0,
            "failed_requests": 0,
            "total_inference_time": 0.0,
            "min_inference_time": float("inf"),
            "max_inference_time": 0.0,
            "request_counts": {},
            "error_counts": {},
            "start_time": datetime.utcnow()
        }
    
    def record_request(self, endpoint: str, status_code: int, inference_time: float = 0.0):
        """记录请求信息"""
        self.metrics["total_requests"] += 1
        
        if endpoint not in self.metrics["request_counts"]:
            self.metrics["request_counts"][endpoint] = 0
        self.metrics["request_counts"][endpoint] += 1
        
        if status_code >= 200 and status_code < 400:
            self.metrics["success_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
            if status_code not in self.metrics["error_counts"]:
                self.metrics["error_counts"][status_code] = 0
            self.metrics["error_counts"][status_code] += 1
        
        if inference_time > 0:
            self.metrics["total_inference_time"] += inference_time
            self.metrics["min_inference_time"] = min(self.metrics["min_inference_time"], inference_time)
            self.metrics["max_inference_time"] = max(self.metrics["max_inference_time"], inference_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_inference_time = 0.0
        if self.metrics["success_requests"] > 0:
            avg_inference_time = self.metrics["total_inference_time"] / self.metrics["success_requests"]
        
        uptime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
        
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": self._calculate_success_rate(),
            "avg_inference_time_ms": avg_inference_time * 1000,
            "min_inference_time_ms": self.metrics["min_inference_time"] * 1000,
            "max_inference_time_ms": self.metrics["max_inference_time"] * 1000,
            "request_counts": self.metrics["request_counts"],
            "error_counts": self.metrics["error_counts"],
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime)
        }
    
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if self.metrics["total_requests"] == 0:
            return 0.0
        return (self.metrics["success_requests"] / self.metrics["total_requests"]) * 100
    
    def _format_uptime(self, seconds: float) -> str:
        """格式化运行时间"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {secs}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class MonitoringMiddleware(BaseHTTPMiddleware):
    """监控中间件"""
    
    def __init__(self, app, logger: StructuredLogger, monitor: PerformanceMonitor):
        super().__init__(app)
        self.logger = logger
        self.monitor = monitor
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """处理请求并记录监控数据"""
        start_time = time.time()
        endpoint = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            
            self.monitor.record_request(endpoint, response.status_code, process_time)
            self.logger.info(
                "Request processed",
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                process_time_ms=round(process_time * 1000, 2)
            )
            
            return response
        
        except Exception as e:
            process_time = time.time() - start_time
            
            self.monitor.record_request(endpoint, 500, process_time)
            self.logger.error(
                "Request failed",
                endpoint=endpoint,
                method=method,
                error=str(e),
                process_time_ms=round(process_time * 1000, 2)
            )
            
            raise


class InferenceTracker:
    """推理追踪器"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.active_requests = 0
        self.max_concurrent_requests = 0
    
    def start_inference(self, request_id: str, data_shape: tuple):
        """开始推理"""
        self.active_requests += 1
        self.max_concurrent_requests = max(self.max_concurrent_requests, self.active_requests)
        
        self.logger.info(
            "Inference started",
            request_id=request_id,
            data_shape=data_shape,
            active_requests=self.active_requests
        )
    
    def end_inference(self, request_id: str, success: bool, inference_time: float, error: str = None):
        """结束推理"""
        self.active_requests -= 1
        
        if success:
            self.logger.info(
                "Inference completed",
                request_id=request_id,
                inference_time_ms=round(inference_time * 1000, 2),
                active_requests=self.active_requests
            )
        else:
            self.logger.error(
                "Inference failed",
                request_id=request_id,
                error=error,
                inference_time_ms=round(inference_time * 1000, 2),
                active_requests=self.active_requests
            )
    
    def get_status(self) -> Dict[str, Any]:
        """获取追踪状态"""
        return {
            "active_requests": self.active_requests,
            "max_concurrent_requests": self.max_concurrent_requests
        }
