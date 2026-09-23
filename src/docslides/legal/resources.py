"""RAM/VRAM check behind the Legal tab's DictaLM tier suggestion (spec 13).

Available memory is measured on the machine running the API: free VRAM
(nvidia-smi) for a vLLM-served tier, which needs the model in GPU memory,
and free VRAM + available RAM for Ollama, which can split layers across
both. A tier whose model is already loaded on its server is treated as
fitting -- its weights are what's using the memory -- so a resident Heavy
model doesn't trigger a false "insufficient RAM" banner.

This only ever *suggests*. The user can keep Heavy regardless.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass

import httpx

from docslides.config import DictaTierConfig, get_config

SUGGESTION_MESSAGE = (
    "Insufficient RAM detected for the Heavy DictaLM model. "
    "We recommend switching to the Light model for stable performance."
)
RISK_NOTE = "You can keep Heavy anyway, but expect slow generation or a model load failure."


@dataclass
class MemorySnapshot:
    ram_available_gb: float
    ram_total_gb: float
    vram_free_gb: float
    vram_total_gb: float


@dataclass
class TierStatus:
    key: str
    label: str
    model: str
    min_memory_gb: float
    available_gb: float
    loaded: bool
    fits: bool


def memory_snapshot() -> MemorySnapshot:
    import psutil

    vm = psutil.virtual_memory()
    vram_free = vram_total = 0.0
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            for line in out.strip().splitlines():
                free, total = (float(v) for v in line.split(","))
                vram_free += free / 1024
                vram_total += total / 1024
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    gb = 1024**3
    return MemorySnapshot(vm.available / gb, vm.total / gb, vram_free, vram_total)


def _model_loaded(tier: DictaTierConfig) -> bool:
    llm = tier.llm
    try:
        with httpx.Client(timeout=2) as client:
            if llm.backend == "ollama":
                models = client.get(f"{llm.base_url.rstrip('/')}/api/ps").json().get("models", [])
                return any(m.get("name", "").split(":")[0] == llm.model.split(":")[0] for m in models)
            data = client.get(f"{llm.base_url.rstrip('/')}/models").json().get("data", [])
            return any(m.get("id") == llm.model for m in data)
    except (httpx.HTTPError, ValueError):
        return False


def evaluate_tier(tier: DictaTierConfig, snapshot: MemorySnapshot, loaded: bool) -> tuple[float, bool]:
    available = snapshot.vram_free_gb if tier.llm.backend == "vllm" else snapshot.vram_free_gb + snapshot.ram_available_gb
    return available, loaded or available >= tier.min_memory_gb


def tier_report(selected: str | None) -> dict:
    legal = get_config().legal
    selected_key, _ = legal.dicta_tier(selected)
    snapshot = memory_snapshot()

    statuses: list[TierStatus] = []
    for key, tier in legal.dicta_tiers.items():
        loaded = _model_loaded(tier)
        available, fits = evaluate_tier(tier, snapshot, loaded)
        statuses.append(TierStatus(key, tier.label, tier.llm.model, tier.min_memory_gb, round(available, 1), loaded, fits))

    return {
        "selected": selected_key,
        "memory": {k: round(v, 1) for k, v in asdict(snapshot).items()},
        "tiers": [asdict(s) for s in statuses],
        **suggestion(selected_key, statuses),
    }


def suggestion(selected_key: str, statuses: list[TierStatus]) -> dict:
    """Suggest the largest fitting smaller tier when the selected one doesn't fit."""
    by_key = {s.key: s for s in statuses}
    current = by_key[selected_key]
    if current.fits:
        return {"suggest": None, "message": None}
    smaller = sorted(
        (s for s in statuses if s.min_memory_gb < current.min_memory_gb and s.fits),
        key=lambda s: s.min_memory_gb,
        reverse=True,
    )
    if not smaller:
        return {
            "suggest": None,
            "message": f"Available memory ({current.available_gb} GB) is below every configured DictaLM "
            f"tier's requirement. {RISK_NOTE}",
        }
    message = (
        SUGGESTION_MESSAGE
        if selected_key == "heavy" and smaller[0].key == "light"
        else f"Insufficient RAM detected for the {current.label} model. We recommend switching to "
        f"{smaller[0].label} for stable performance."
    )
    return {"suggest": smaller[0].key, "message": f"{message} {RISK_NOTE}"}
