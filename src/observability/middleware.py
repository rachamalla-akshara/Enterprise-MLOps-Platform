"""Prometheus instrumentation middleware for FastAPI."""

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.observability.metrics import (
    HTTP_ERRORS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
    normalize_endpoint,
)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for measuring HTTP request metrics, latency distributions, and errors."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Ignore Prometheus scrape requests to avoid feedback loop skewing request rates
        endpoint = normalize_endpoint(request.url.path)
        method = request.method

        if endpoint == "/metrics":
            return await call_next(request)

        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        t_start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - t_start
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

            status_str = str(status_code)
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_str,
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            if status_code >= 400:
                HTTP_ERRORS_TOTAL.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_str,
                ).inc()
