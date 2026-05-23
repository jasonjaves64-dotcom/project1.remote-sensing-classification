"""
监控与日志系统 - 增强版

新增功能：
1. 分布式日志支持
2. 告警系统（邮件/钉钉）
3. 性能仪表盘数据收集
4. 自定义日志级别
5. 日志过滤和搜索
"""

import logging
import logging.handlers
import os
import time
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from collections import defaultdict, deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from email.mime.text import MIMEText
    import smtplib
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False

class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def __init__(self, fmt: str = None, datefmt: str = "%Y-%m-%d %H:%M:%S"):
        super().__init__(fmt, datefmt)
        self.extra_fields = ['request_id', 'user_id', 'trace_id']
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
            "pid": os.getpid()
        }
        
        for field in self.extra_fields:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.email_enabled = self.config.get('email', {}).get('enabled', False)
        self.dingtalk_enabled = self.config.get('dingtalk', {}).get('enabled', False)
        
        if self.email_enabled:
            self.email_config = self.config['email']
        
        self.alert_history = deque(maxlen=100)
    
    def send_email_alert(self, subject: str, message: str):
        """发送邮件告警"""
        if not self.email_enabled:
            return
        
        try:
            msg = MIMEText(message, 'plain', 'utf-8')
            msg['Subject'] = subject
            msg['From'] = self.email_config['sender']
            msg['To'] = ','.join(self.email_config['recipients'])
            
            with smtplib.SMTP(self.email_config['host'], self.email_config['port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.sendmail(self.email_config['sender'], self.email_config['recipients'], msg.as_string())
            
            self.alert_history.append({
                'type': 'email',
                'timestamp': time.time(),
                'subject': subject,
                'success': True
            })
        except Exception as e:
            self.alert_history.append({
                'type': 'email',
                'timestamp': time.time(),
                'subject': subject,
                'success': False,
                'error': str(e)
            })
    
    def send_dingtalk_alert(self, title: str, message: str):
        """发送钉钉告警"""
        if not self.dingtalk_enabled:
            return
        
        try:
            import requests
            
            webhook_url = self.config['dingtalk']['webhook_url']
            data = {
                "msgtype": "text",
                "text": {
                    "content": f"【{title}】\n{message}"
                }
            }
            
            response = requests.post(webhook_url, json=data)
            response.raise_for_status()
            
            self.alert_history.append({
                'type': 'dingtalk',
                'timestamp': time.time(),
                'title': title,
                'success': True
            })
        except Exception as e:
            self.alert_history.append({
                'type': 'dingtalk',
                'timestamp': time.time(),
                'title': title,
                'success': False,
                'error': str(e)
            })
    
    def trigger_alert(self, level: str, message: str, details: Dict[str, Any] = None):
        """触发告警"""
        title = f"【{level}】遥感影像分类系统告警"
        full_message = f"{message}\n\n详细信息:\n{json.dumps(details or {}, indent=2, ensure_ascii=False)}"
        
        if level in ['ERROR', 'CRITICAL']:
            self.send_email_alert(title, full_message)
            self.send_dingtalk_alert(title, message)
        elif level == 'WARNING':
            self.send_dingtalk_alert(title, message)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, alert_manager: Optional[AlertManager] = None):
        self.alert_manager = alert_manager
        self.metrics = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'min_inference_time': float('inf'),
            'max_inference_time': 0.0,
            'error_count': 0,
            'success_count': 0,
            'memory_usage_history': deque(maxlen=100),
            'gpu_memory_usage': deque(maxlen=100),
            'cpu_usage_history': deque(maxlen=100),
            'response_time_history': deque(maxlen=100),
            'throughput_history': deque(maxlen=100)
        }
        self.start_time = time.time()
        self.request_timestamps = deque(maxlen=1000)
        self.last_alert_time = 0
    
    def record_inference(self, duration: float, success: bool = True):
        """记录推理性能"""
        self.metrics['inference_count'] += 1
        self.metrics['total_inference_time'] += duration
        self.metrics['min_inference_time'] = min(self.metrics['min_inference_time'], duration)
        self.metrics['max_inference_time'] = max(self.metrics['max_inference_time'], duration)
        
        if success:
            self.metrics['success_count'] += 1
        else:
            self.metrics['error_count'] += 1
        
        self.request_timestamps.append(time.time())
        self.metrics['response_time_history'].append({
            'timestamp': time.time(),
            'duration_ms': duration * 1000
        })
        
        self._check_alerts()
    
    def record_resource_usage(self):
        """Record resource usage (requires psutil)."""
        if not HAS_PSUTIL:
            return
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        self.metrics['memory_usage_history'].append({
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent()
        })
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
            self.metrics['gpu_memory_usage'].append({
                'timestamp': time.time(),
                'gpu_memory_mb': gpu_memory,
                'gpu_cached_mb': gpu_cached
            })
    
    def _check_alerts(self):
        """检查告警条件"""
        current_time = time.time()
        if current_time - self.last_alert_time < 300:
            return
        
        stats = self.get_stats()
        
        if stats['error_rate'] > 5:
            self.alert_manager.trigger_alert(
                'ERROR',
                f"错误率过高: {stats['error_rate']:.2f}%",
                stats
            )
            self.last_alert_time = current_time
        
        if stats['avg_inference_time_ms'] > 500:
            self.alert_manager.trigger_alert(
                'WARNING',
                f"推理延迟过高: {stats['avg_inference_time_ms']:.2f}ms",
                stats
            )
            self.last_alert_time = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self.metrics['success_count'] + self.metrics['error_count']
        avg_latency = self.metrics['total_inference_time'] / max(self.metrics['inference_count'], 1)
        
        recent_requests = [t for t in self.request_timestamps if t > time.time() - 60]
        qps = len(recent_requests) / 60
        
        return {
            'uptime': time.time() - self.start_time,
            'total_requests': total_requests,
            'success_rate': self.metrics['success_count'] / max(total_requests, 1) * 100,
            'error_rate': self.metrics['error_count'] / max(total_requests, 1) * 100,
            'inference_count': self.metrics['inference_count'],
            'avg_inference_time_ms': avg_latency * 1000,
            'min_inference_time_ms': self.metrics['min_inference_time'] * 1000,
            'max_inference_time_ms': self.metrics['max_inference_time'] * 1000,
            'qps': qps,
            'error_count': self.metrics['error_count'],
            'memory_usage': self.metrics['memory_usage_history'][-1] if self.metrics['memory_usage_history'] else None,
            'gpu_memory_usage': self.metrics['gpu_memory_usage'][-1] if self.metrics['gpu_memory_usage'] else None,
            'throughput': qps
        }
    
    def reset(self):
        """重置统计信息"""
        self.__init__(self.alert_manager)

class ErrorTracker:
    """错误追踪器"""
    
    def __init__(self, max_errors: int = 100):
        self.errors = deque(maxlen=max_errors)
        self.error_counts = defaultdict(int)
        self.error_trend = deque(maxlen=60)
    
    def record_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None):
        """记录错误"""
        error_info = {
            'timestamp': time.time(),
            'error_type': type(exception).__name__,
            'message': str(exception),
            'context': context or {}
        }
        self.errors.append(error_info)
        self.error_counts[type(exception).__name__] += 1
        
        self.error_trend.append({
            'timestamp': time.time(),
            'count': 1
        })
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的错误"""
        return list(self.errors)[-limit:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        recent_errors = [e for e in self.error_trend if e['timestamp'] > time.time() - 300]
        recent_count = len(recent_errors)
        
        return {
            'total_errors': len(self.errors),
            'error_distribution': dict(self.error_counts),
            'recent_errors': self.get_recent_errors(5),
            'last_5min_errors': recent_count,
            'is_healthy': recent_count < 10
        }

class LogManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO", alert_config: Dict[str, Any] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = getattr(logging, log_level.upper())
        
        self.alert_manager = AlertManager(alert_config)
        self.monitor = PerformanceMonitor(self.alert_manager)
        self.error_tracker = ErrorTracker()
        
        self.logger = logging.getLogger("crop_classification")
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False
        self.logger.handlers.clear()
        
        self._add_handlers()
        
        self.request_id = None
    
    def _add_handlers(self):
        """添加日志处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(console_handler)
        
        # 文件处理器（JSON格式）
        file_path = self.log_dir / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=10,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)
        
        # 错误日志处理器
        error_file_path = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)
        
        # 性能日志处理器
        perf_file_path = self.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_file_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(perf_handler)
    
    def set_request_context(self, request_id: str = None, user_id: str = None):
        """设置请求上下文"""
        self.request_id = request_id
    
    def _create_log_record(self, level: int, message: str, **kwargs):
        """创建日志记录"""
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        record.extra = kwargs
        if self.request_id:
            record.request_id = self.request_id
        
        return record
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        if exception:
            self.error_tracker.record_error(exception, kwargs)
            message = f"{message}: {str(exception)}"
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        if exception:
            self.error_tracker.record_error(exception, kwargs)
            message = f"{message}: {str(exception)}"
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """通用日志记录方法"""
        if kwargs:
            record = self._create_log_record(level, message, **kwargs)
            self.logger.handle(record)
        else:
            self.logger.log(level, message)
    
    def log_inference(self, duration: float, input_shape: tuple, success: bool = True):
        """记录推理日志"""
        self.monitor.record_inference(duration, success)
        self.monitor.record_resource_usage()
        
        log_func = self.info if success else self.error
        log_func(
            "推理完成" if success else "推理失败",
            duration_ms=duration * 1000,
            input_shape=input_shape,
            status="success" if success else "error"
        )
    
    def log_performance(self, metrics: Dict[str, Any]):
        """记录性能指标"""
        self.info("性能指标", **metrics)
    
    def get_monitor_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        return self.monitor.get_stats()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        return self.error_tracker.get_error_summary()
    
    def export_logs(self, output_path: Optional[str] = None) -> str:
        """导出日志统计"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'monitor_stats': self.get_monitor_stats(),
            'error_summary': self.get_error_summary(),
            'alert_history': list(self.alert_manager.alert_history)
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            return f"日志已导出到: {output_path}"
        else:
            return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def search_logs(self, keyword: str, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """搜索日志"""
        results = []
        log_file = self.log_dir / "app.log"
        
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if keyword.lower() in entry.get('message', '').lower():
                            if level is None or entry.get('level') == level.upper():
                                results.append(entry)
                                if len(results) >= limit:
                                    break
                    except json.JSONDecodeError:
                        continue
        
        return results

# Lazy-initialized global log manager
_log_manager = None

def _get_log_manager():
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager

# Module-level property for backward compatibility
class _LogManagerProxy:
    def __getattr__(self, name):
        return getattr(_get_log_manager(), name)

log_manager = _LogManagerProxy()

# Convenience functions
def get_logger(name: str = "crop_classification") -> LogManager:
    return log_manager

def log_debug(message: str, **kwargs):
    log_manager.debug(message, **kwargs)

def log_info(message: str, **kwargs):
    log_manager.info(message, **kwargs)

def log_warning(message: str, **kwargs):
    log_manager.warning(message, **kwargs)

def log_error(message: str, exception: Optional[Exception] = None, **kwargs):
    log_manager.error(message, exception, **kwargs)

def log_critical(message: str, exception: Optional[Exception] = None, **kwargs):
    log_manager.critical(message, exception, **kwargs)

def log_inference(duration: float, input_shape: tuple, success: bool = True):
    log_manager.log_inference(duration, input_shape, success)

def get_stats() -> Dict[str, Any]:
    return log_manager.get_monitor_stats()

def get_errors() -> Dict[str, Any]:
    return log_manager.get_error_summary()

def set_request_context(request_id: str):
    log_manager.set_request_context(request_id)

def search_logs(keyword: str, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    return log_manager.search_logs(keyword, level, limit)

if __name__ == "__main__":
    log_manager.info("系统启动", service="crop_classification", version="1.0.0")
    log_manager.debug("调试信息", module="test", value=42)
    log_manager.warning("警告信息", level="medium")
    
    try:
        raise ValueError("测试异常")
    except ValueError as e:
        log_manager.error("发生错误", exception=e, context={"test": "value"})
    
    log_manager.log_inference(0.123, (1, 12, 10, 256, 256), success=True)
    log_manager.log_inference(0.234, (1, 12, 10, 256, 256), success=True)
    
    print("\n监控统计:")
    print(json.dumps(log_manager.get_monitor_stats(), indent=2))
    
    print("\n错误摘要:")
    print(json.dumps(log_manager.get_error_summary(), indent=2))
    
    export_path = "logs/export_stats.json"
    log_manager.export_logs(export_path)
    print(f"\n日志已导出到: {export_path}")
