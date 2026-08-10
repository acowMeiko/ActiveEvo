"""
core/endpoint_pool.py — Per-GPU smart endpoint pool with heavy/light separation.

Replaces the simple round-robin URLRouter with load-aware, per-endpoint
semaphore-gated routing that separates long-context ("heavy") calls from
short-context ("light") calls.

Capacity model (per A800 80GB):
  - heavy (answer_guided, answer_direct): 6 slots/GPU  → 48 total for 8 GPUs
  - light (extract_semantics, diff_analysis):  24 slots/GPU → 192 total
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)

# Per-GPU slot counts — calibrated for Qwen 7B on A800 80GB
DEFAULT_HEAVY_PER_GPU = 6   # long-context (~3-5K tokens prompt)
DEFAULT_LIGHT_PER_GPU = 24  # short-context (~500 tokens prompt)


class EndpointPool:
    """Load-aware endpoint pool with separate heavy/light semaphores per endpoint."""

    def __init__(
        self,
        urls: List[str],
        heavy_per_gpu: int = DEFAULT_HEAVY_PER_GPU,
        light_per_gpu: int = DEFAULT_LIGHT_PER_GPU,
    ):
        if not urls:
            raise ValueError("EndpointPool requires at least one URL")
        self.urls = list(urls)
        self.n_gpus = len(urls)
        self._heavy_sems = {url: asyncio.Semaphore(heavy_per_gpu) for url in urls}
        self._light_sems = {url: asyncio.Semaphore(light_per_gpu) for url in urls}
        self._heavy_active: Dict[str, int] = {url: 0 for url in urls}
        self._light_active: Dict[str, int] = {url: 0 for url in urls}
        self._selector_lock = asyncio.Lock()

        # Stats
        self._total_acquired = 0
        self._total_released = 0
        self._timeouts = 0

        logger.info(
            "EndpointPool: %d GPUs, heavy=%d/gpu (total=%d), light=%d/gpu (total=%d)",
            self.n_gpus, heavy_per_gpu, self.n_gpus * heavy_per_gpu,
            light_per_gpu, self.n_gpus * light_per_gpu,
        )

    # ── Properties ──

    @property
    def total_heavy_slots(self) -> int:
        return sum(s._value for s in self._heavy_sems.values())

    @property
    def total_light_slots(self) -> int:
        return sum(s._value for s in self._light_sems.values())

    def stats(self) -> Dict[str, Any]:
        return {
            "n_gpus": self.n_gpus,
            "heavy_free": self.total_heavy_slots,
            "light_free": self.total_light_slots,
            "heavy_active": dict(self._heavy_active),
            "light_active": dict(self._light_active),
            "total_acquired": self._total_acquired,
            "total_timeouts": self._timeouts,
        }

    # ── Core acquire ──

    @asynccontextmanager
    async def acquire(self, call_type: str = "heavy") -> AsyncIterator[str]:
        """Acquire a slot on the least-loaded endpoint for this call type.

        Args:
            call_type: "heavy" for answer_guided/direct, "light" for extract_semantics/diff

        Yields:
            url: The endpoint URL to use
        """
        if call_type not in ("heavy", "light"):
            call_type = "heavy"

        sems = self._heavy_sems if call_type == "heavy" else self._light_sems
        active = self._heavy_active if call_type == "heavy" else self._light_active

        async with self._selector_lock:
            url = min(active, key=active.get)
            active[url] += 1

        self._total_acquired += 1
        sem = sems[url]
        async with sem:
            try:
                yield url
            finally:
                async with self._selector_lock:
                    active[url] -= 1
                self._total_released += 1

    def report_timeout(self, url: str) -> None:
        self._timeouts += 1
