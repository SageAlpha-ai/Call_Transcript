"""
Structured JSON logging, request IDs, request size limits, and request timeout.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Callable, FrozenSet

from fastapi.responses import JSONResponse
from pythonjsonlogger import jsonlogger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from config import get_max_request_body_bytes

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_configured = False


def configure_structured_logging() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s",
        rename_fields={"levelname": "level", "asctime": "timestamp"},
    )
    handler.setFormatter(fmt)

    class RequestIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.request_id = request_id_ctx.get("-")
            return True

    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured = True


async def request_logging_middleware(request: Request, call_next: Callable):
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    request_id_ctx.set(rid)
    t0 = time.perf_counter()
    logging.info(
        json.dumps(
            {
                "event": "request_start",
                "method": request.method,
                "path": request.url.path,
                "request_id": rid,
            }
        )
    )
    try:
        response = await call_next(request)
    except Exception:
        logging.exception(
            json.dumps(
                {
                    "event": "request_error",
                    "method": request.method,
                    "path": request.url.path,
                    "request_id": rid,
                }
            )
        )
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logging.info(
        json.dumps(
            {
                "event": "request_end",
                "method": request.method,
                "path": request.url.path,
                "request_id": rid,
                "status_code": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
            }
        )
    )
    response.headers["X-Request-ID"] = rid
    return response


async def max_body_middleware(request: Request, call_next: Callable):
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > get_max_request_body_bytes():
                    return JSONResponse(
                        status_code=413,
                        content={"error": "Payload too large", "details": None},
                    )
            except ValueError:
                pass
    return await call_next(request)


REQUEST_TIMEOUT_SECONDS = 120.0

_SKIP_REQUEST_TIMEOUT_PATHS: FrozenSet[str] = frozenset(
    {
        "/transcribe",
        "/pipeline",
        "/analyze",
        "/analyze-all",
        "/next-move",
        "/escalation",
        "/followup-email",
        "/agent-score",
    }
)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """ASGI timeout; skips long-running analysis/transcription routes."""

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in _SKIP_REQUEST_TIMEOUT_PATHS:
            return await call_next(request)
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": "Request timeout"},
            )
