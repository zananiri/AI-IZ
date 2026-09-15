#!/usr/bin/env python3
"""Sanity-check the vLLM launch configuration before deploying.

Run this ONCE, online, at build/deploy time -- not at inference time -- to
confirm the configured model repo exists and print the exact `vllm serve`
command implied by config/config.yaml, so it can be diffed against
docker-compose.yml or run manually.

    python scripts/verify_vllm_launch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from docslides.config import get_config  # noqa: E402


def build_launch_command(cfg) -> str:
    return (
        f"vllm serve {cfg.llm.model} \\\n"
        f"  --quantization {cfg.vllm_launch.quantization} \\\n"
        f"  --max-model-len {cfg.llm.max_model_len} \\\n"
        f"  --gpu-memory-utilization {cfg.vllm_launch.gpu_memory_utilization} \\\n"
        f"  --guided-decoding-backend {cfg.llm.guided_decoding_backend} \\\n"
        f"  --port {cfg.vllm_launch.port}"
    )


def check_model_repo_exists(model_repo: str) -> None:
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_repo)
        print(f"[ok] Model repo '{model_repo}' found on the Hugging Face Hub.")
    except ImportError:
        print("[skip] huggingface_hub not installed; cannot verify the repo exists. "
              "pip install huggingface_hub to enable this check.")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] Could not verify model repo '{model_repo}': {exc}\n"
            "        Confirm the repo name against the current Qwen3-32B AWQ/GPTQ "
            "checkpoint listings before deploying -- quantized repo names change "
            "as new checkpoints are published."
        )


def main() -> None:
    cfg = get_config()
    print("Configured vLLM launch command:\n")
    print(build_launch_command(cfg))
    print()
    check_model_repo_exists(cfg.llm.model)


if __name__ == "__main__":
    main()
