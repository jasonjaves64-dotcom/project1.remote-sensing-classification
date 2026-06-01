"""
Sliding-window rate limiter middleware for Starlette / FastAPI.
In-memory, thread-safe, periodic stale-key cleanup (time-driven).

Configuration::

    RATE_LIMIT_CONFIG = {
        "/predict/*": "10/minute",  "/health": "60/minute",
        "default":    "30/minute",
    }
    app.add_middleware(RateLimiterMiddleware, rate_limit_config=RATE_LIMIT_CONFIG)
"""

from __future__ import annotations

import threading, time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import ClassVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# -- helpers ---------------------------------------------------------------

def _parse_rate(rate_str: str) -> tuple[int, float]:
    """``"10/minute"`` → ``(10, 60.0)``."""
    count_str, unit = rate_str.strip().split("/")
    count = int(count_str)
    match unit:
        case "second" | "seconds":   return count, 1.0
        case "minute" | "minutes":   return count, 60.0
        case "hour" | "hours":       return count, 3600.0
        case _:  raise ValueError(f"Unknown rate unit: {unit!r}")

def _match_route(path: str, pattern: str) -> bool:
    """Exact (``"/health"``) or prefix (``"/predict/*"``) match."""
    if pattern.endswith("/*"):
        prefix = pattern[:-2]
        return path == prefix or path.startswith(prefix + "/")
    return path == pattern
# -- sliding-window store --------------------------------------------------

class _SlidingWindowStore:
    """Per-key deque of request timestamps; protected by ``threading.Lock``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}
        self._last_cleanup = time.monotonic()

    def hit(self, key: str, limit: int, window: float) -> tuple[bool, float]:
        """Record a hit → ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        with self._lock:
            ts = self._buckets.get(key)
            if ts is None:
                ts = deque[float]()
                self._buckets[key] = ts
            # Evict timestamps outside the current window
            cutoff = now - window
            while ts and ts[0] < cutoff:
                ts.popleft()
            if len(ts) < limit:
                ts.append(now)
                return True, 0.0
            retry_after = ts[0] + window - now  # oldest expires
            return False, max(retry_after, 0.0)

    def cleanup_stale(self, max_age: float) -> int:
        """Remove keys whose newest timestamp is older than *max_age*."""
        cutoff = time.monotonic() - max_age
        with self._lock:
            stale = [k for k, ts in self._buckets.items()
                     if not ts or ts[-1] < cutoff]
            for k in stale:
                del self._buckets[k]
        return len(stale)
# -- middleware -------------------------------------------------------------

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter — one bucket per (IP, path)."""

    CLEANUP_INTERVAL: ClassVar[float] = 300.0  # sweep every 5 min

    def __init__(self, app,
                 rate_limit_config: dict[str, str] | None = None) -> None:
        super().__init__(app)
        raw = rate_limit_config or {}
        self._rules: list[tuple[str, int, float]] = []
        self._default: tuple[int, float] = (30, 60.0)

        for pat, rate_str in raw.items():
            limit, window = _parse_rate(rate_str)
            if pat == "default":
                self._default = (limit, window)
            else:
                self._rules.append((pat, limit, window))

        self._store = _SlidingWindowStore()
        self._cleanup_lock = threading.Lock()

    def _resolve(self, path: str) -> tuple[int, float]:
        for pat, limit, window in self._rules:
            if _match_route(path, pat):
                return limit, window
        return self._default

    def _maybe_cleanup(self) -> None:
        """Sweep stale keys every ``CLEANUP_INTERVAL`` (non-blocking)."""
        now = time.monotonic()
        if now - self._store._last_cleanup < self.CLEANUP_INTERVAL:
            return
        if not self._cleanup_lock.acquire(blocking=False):
            return
        try:
            if now - self._store._last_cleanup >= self.CLEANUP_INTERVAL:
                max_w = max((w for _, _, w in self._rules),
                            default=self._default[1])
                self._store.cleanup_stale(max_w * 2)
                self._store._last_cleanup = time.monotonic()
        finally:
            self._cleanup_lock.release()

    async def dispatch(self, request: Request,
                       call_next: Callable[[Request], Awaitable]):
        # Resolve client IP (honour X-Forwarded-For)
        forwarded = request.headers.get("X-Forwarded-For")
        client_ip = (
            forwarded.split(",")[0].strip() if forwarded else
            request.client.host if request.client else
            "unknown"
        )
        path = request.url.path
        limit, window = self._resolve(path)
        allowed, retry = self._store.hit(f"{client_ip}:{path}", limit, window)

        if not allowed:
            retry_s = max(1, int(retry + 0.999))  # ceil
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Retry in {retry_s} seconds."},
                headers={"Retry-After": str(retry_s)},
            )

        self._maybe_cleanup()
        return await call_next(request)
