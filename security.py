"""
Lightweight per-IP rate limiting.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi.responses import JSONResponse
from starlette.requests import Request

from config import get_rate_limit_per_minute

PUBLIC_PATHS_EXACT = frozenset(
    {"/health", "/ready", "/ping", "/openapi.json", "/docs/oauth2-redirect"}
)


def client_remote_address(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"


def is_public_path(path: str) -> bool:
    if path in PUBLIC_PATHS_EXACT:
        return True
    if path.startswith("/docs") or path.startswith("/redoc"):
        return True
    return False


class RateLimiter:
    def __init__(self, max_requests: int, window_s: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_s
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self._window:
            q.popleft()
        if len(q) >= self._max:
            return False
        q.append(now)
        return True


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(get_rate_limit_per_minute(), 60.0)
    return _rate_limiter


async def rate_limit_middleware(request: Request, call_next):
    if is_public_path(request.url.path):
        return await call_next(request)
    client = client_remote_address(request)
    if not get_rate_limiter().allow(client):
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests", "details": "Rate limit exceeded"},
        )
    return await call_next(request)
