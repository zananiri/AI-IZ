"""Cooperative GPU arbitration between vLLM and the GPU-based OCR fallback.

We never hold a GPU-resident OCR model at the same time vLLM is actively
generating a batch. Since vLLM and the OCR fallback run as separate
processes, coordination is cooperative: before loading a GPU OCR model, the
fallback path asks vLLM's Prometheus metrics endpoint whether requests are
currently in flight (`vllm:num_requests_running` / `vllm:num_requests_waiting`)
and waits (polling) until vLLM is idle, up to `max_wait_s`. If vLLM never
idles within that window, the fallback proceeds anyway rather than blocking
OCR indefinitely -- correctness of the OCR result matters more than strict
mutual exclusion, but we always prefer to yield first.

An in-process asyncio lock additionally serializes GPU OCR calls against each
other, since only one fallback model should be resident at a time.
"""

from __future__ import annotations

import asyncio
import re
import time

import httpx

from docslides.config import get_config
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

_gpu_ocr_lock = asyncio.Lock()

_METRIC_RE = re.compile(r"^vllm:num_requests_(running|waiting)\{.*?\}\s+([0-9.]+)", re.MULTILINE)


async def _vllm_is_busy(metrics_url: str, client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(metrics_url, timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("vllm_metrics_probe_failed_assuming_idle", error=str(exc))
        return False

    total = 0.0
    for match in _METRIC_RE.finditer(resp.text):
        total += float(match.group(2))
    return total > 0


async def wait_for_gpu_opportunity() -> None:
    """Block (with polling) until vLLM appears idle, or `max_wait_s` elapses.

    This arbitration is vLLM-specific (it polls vLLM's Prometheus metrics
    endpoint, which Ollama does not expose in the same shape). When the
    configured backend is Ollama there is no equivalent signal to poll, so
    this is a no-op -- Ollama's own scheduler handles resource contention
    between the chat model and anything else touching the GPU it's on.
    """
    cfg = get_config()
    if cfg.llm.backend != "vllm":
        return
    metrics_url = cfg.llm.base_url.rsplit("/v1", 1)[0] + "/metrics"
    arbiter_cfg = cfg.ocr.gpu_arbiter

    deadline = time.monotonic() + arbiter_cfg.max_wait_s
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if not await _vllm_is_busy(metrics_url, client):
                return
            logger.info("gpu_ocr_waiting_for_vllm_idle")
            await asyncio.sleep(arbiter_cfg.poll_interval_s)

    logger.warning("gpu_ocr_proceeding_without_confirmed_idle_vllm")


class GPUOCRSession:
    """Async context manager: acquire the in-process GPU-OCR lock and yield
    to vLLM first if it's busy. Use around any GPU-resident OCR fallback call."""

    async def __aenter__(self) -> "GPUOCRSession":
        await wait_for_gpu_opportunity()
        await _gpu_ocr_lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _gpu_ocr_lock.release()
