"""
Middleware-based rate limiter — zero endpoint intrusion.

Uses a sliding-window counter per client IP, with per-route limits
configured via a simple dict. No decorators, no request-signature
requirements, no per-endpoint changes needed when adding new routes.

Usage:
    from utils.rate_limiter import RateLimitMiddleware, rate_limit_config

    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)

Config keys are matched via str.startswith() against request.url.path,
so "/predict" matches "/predict/v5", "/predict/v6/upload", etc.
More specific paths should be listed first.
"""

import time
import asyncio
from collections import defaultdict
from typing import Callable, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# ---------------------------------------------------------------------------
# Default per-route rate limits (requests / minute / IP)
# ---------------------------------------------------------------------------
rate_limit_config: dict[str, tuple[int, int]] = {
    # path prefix  -> (max_requests, window_seconds)
    "/predict":       (30, 60),   # 30 req/min for model inference
    "/inference":     (50, 60),   # 50 req/min for legacy inference
    "/train":         (5,  60),   # 5 req/min for training
    "/model/load":    (5,  60),   # 5 req/min for model loading
    "/health":        (120, 60),  # health check — generous
    "/stats":         (60,  60),
    "/version":       (120, 60),
    "/docs":          (120, 60),
    "/redoc":         (120, 60),
    "/openapi.json":  (120, 60),
    "/":              (120, 60),
}

# Global default when no route matches
DEFAULT_LIMIT = (100, 60)  # 100 req/min


def _find_limit(path: str) -> tuple[int, int]:
    """Find the rate limit for a given path.  Checks config keys in order."""
    for prefix, limit in rate_limit_config.items():
        if path.startswith(prefix):
            return limit
    return DEFAULT_LIMIT


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter middleware.

    State is kept in-memory (per-process).  For multi-worker deployments
    behind a load balancer, use IP-hash sticky sessions or swap the store
    for Redis.
    """

    def __init__(self, app, config: Optional[dict] = None):
        super().__init__(app)
        if config is not None:
            self._config = config
        else:
            self._config = rate_limit_config
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = getattr(request, "client", None)
        if client is not None:
            return client.host if hasattr(client, "host") else str(client)
        return "unknown"

    def _find_limit(self, path: str) -> tuple[int, int]:
        for prefix, limit in self._config.items():
            if path.startswith(prefix):
                return limit
        return DEFAULT_LIMIT

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        max_req, window = self._find_limit(path)
        now = time.monotonic()
        key = f"{self._get_client_ip(request)}:{path}"

        async with self._lock:
            timestamps = self._windows[key]
            # Evict expired entries
            cutoff = now - window
            while timestamps and timestamps[0] < cutoff:
                timestamps.pop(0)
            if len(timestamps) >= max_req:
                retry_after = int(timestamps[0] + window - now + 1)
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded. Try again in {retry_after}s",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            timestamps.append(now)

        # Periodic cleanup of stale keys (every ~100 requests, best-effort)
        if hash(key) % 100 == 0:
            async with self._lock:
                stale = [k for k, v in self._windows.items() if not v]
                for k in stale:
                    del self._windows[k]

        return await call_next(request)
