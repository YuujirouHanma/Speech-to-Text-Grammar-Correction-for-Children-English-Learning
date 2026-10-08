"""Prometheus metrics.

Metrics are module-level singletons registered on the default registry, so they
are created exactly once per process no matter how many app instances tests
build. Label values are low-cardinality by design (route templates, stage names)
and never contain user content.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import psutil
from prometheus_client import Counter, Gauge, Histogram, Info

_LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

BUILD_INFO = Info("speakfix_build", "Service build information")

HTTP_REQUESTS = Counter(
    "speakfix_http_requests_total",
    "HTTP requests by route template, method and status code",
    ["route", "method", "status"],
)
HTTP_LATENCY = Histogram(
    "speakfix_http_request_duration_seconds",
    "End-to-end HTTP request latency",
    ["route", "method"],
    buckets=_LATENCY_BUCKETS,
)
STAGE_LATENCY = Histogram(
    "speakfix_stage_duration_seconds",
    "Latency of pipeline stages (decode, preprocess, stt, correct)",
    ["stage"],
    buckets=_LATENCY_BUCKETS,
)
AUDIO_DURATION = Histogram(
    "speakfix_audio_duration_seconds",
    "Duration of accepted audio clips",
    buckets=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89),
)
INFERENCE_ERRORS = Counter(
    "speakfix_inference_errors_total",
    "Inference failures by component and kind (oom, timeout, error)",
    ["component", "kind"],
)
FALLBACKS = Counter(
    "speakfix_fallbacks_total",
    "Times a fallback engine/backend was used",
    ["component"],
)
RATE_LIMITED = Counter("speakfix_rate_limited_total", "Requests rejected by the rate limiter")
INFLIGHT = Gauge("speakfix_inference_inflight", "Inference calls currently running or waiting")
JOBS = Gauge("speakfix_jobs", "Async jobs by state", ["state"])
MODEL_READY = Gauge("speakfix_model_ready", "1 when a model component is loaded", ["component"])
MODEL_INFO = Gauge(
    "speakfix_model_info",
    "Loaded model description (value is always 1)",
    ["component", "name", "backend", "device", "compute_type"],
)
PROCESS_RSS = Gauge("speakfix_process_resident_memory_bytes", "Resident memory of the process")
PROCESS_RSS.set_function(lambda: float(psutil.Process().memory_info().rss))


@contextmanager
def track_stage(stage: str) -> Iterator[None]:
    """Time a pipeline stage and record it in :data:`STAGE_LATENCY`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(stage=stage).observe(time.perf_counter() - start)
