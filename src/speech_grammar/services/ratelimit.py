"""In-memory token-bucket rate limiter (per API key or client IP)."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """Allow ``rate_per_minute`` requests per key with bursts up to ``burst``.

    Suitable for a single instance. Behind several replicas, move this to the
    reverse proxy (nginx ``limit_req``) or a Redis-based limiter.
    """

    def __init__(
        self, rate_per_minute: int, *, burst: int | None = None, max_keys: int = 10_000
    ) -> None:
        self.rate_per_second = rate_per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(1, rate_per_minute))
        self.enabled = rate_per_minute > 0
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def acquire(self, key: str) -> tuple[bool, int]:
        """Consume one token for ``key``.

        Returns:
            ``(allowed, retry_after_seconds)``.
        """
        if not self.enabled:
            return True, 0
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._evict(now)
                bucket = _Bucket(tokens=self.capacity, updated=now)
                self._buckets[key] = bucket
            bucket.tokens = min(
                self.capacity, bucket.tokens + (now - bucket.updated) * self.rate_per_second
            )
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            retry_after = math.ceil((1.0 - bucket.tokens) / self.rate_per_second)
            return False, max(1, retry_after)

    def _evict(self, now: float) -> None:
        idle = [k for k, b in self._buckets.items() if now - b.updated > 60.0 * 10]
        for key in idle or list(self._buckets)[: len(self._buckets) // 2]:
            self._buckets.pop(key, None)
