"""Thin client around the configured LLM backend: vLLM's OpenAI-compatible
/v1/chat/completions (NVIDIA GPU only) or Ollama's native /api/chat (CPU/
CUDA/ROCm/Metal -- the portable path). See `config.llm.backend`.

Responsibilities:
  * Toggle Qwen3's "thinking" mode per call site, however the active backend
    exposes that knob (vLLM: chat_template_kwargs; Ollama: `think`).
  * Drive structured JSON decoding for every content-generation call --
    callers never regex-parse free text (vLLM: guided_json extra_body;
    Ollama: top-level `format` JSON Schema).
  * Validate JSON responses against a pydantic schema and retry with an
    error-correction prompt on failure.
  * Stream chat turns and split reasoning from content for the UI, using
    each backend's own wire format (SSE + <think> tags for vLLM; newline-
    delimited JSON with a separate `thinking` field for Ollama).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from docslides.config import LLMConfig, get_config
from docslides.logging_setup import get_logger

logger = get_logger(__name__)


class SchemaValidationFailed(Exception):
    def __init__(self, raw: str, errors: str):
        super().__init__(f"LLM JSON output failed schema validation: {errors}")
        self.raw = raw
        self.errors = errors


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class StreamDelta:
    kind: Literal["reasoning", "content"]
    text: str


@dataclass
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 0.92
    max_tokens: int = 2048


@dataclass
class LLMCallSite:
    """Identifies which config-driven `enable_thinking` default applies."""

    name: Literal[
        "outline_generation",
        "slide_fill",
        "translation",
        "chat_general",
        "tone_rewrite",
        "chunk_summary",
        "legal_language_id",
        "legal_query_normalization",
        "legal_research_memo",
        "legal_draft",
        "legal_citation_verification",
        "legal_hebrew_polish",
        "legal_equivalence_check",
        "legal_eval_baseline",
        "legal_eval_judge",
        "canon_orchestration",
        "canon_answer",
    ]


class QwenClient:
    """Talks to either vLLM's OpenAI-compatible API or Ollama's native API,
    selected by `config.llm.backend`. vLLM is NVIDIA-GPU-only but fastest;
    Ollama runs on CPU or whatever acceleration the host exposes (CUDA/ROCm/
    Metal), which is what makes the app portable to non-NVIDIA machines. The
    two APIs differ enough (endpoint path, request shape, structured-JSON
    mechanism, streaming wire format, and how "thinking" tokens are exposed)
    that this class branches on `self._backend` rather than pretending
    they're the same protocol.
    """

    def __init__(self, llm_cfg: LLMConfig | None = None) -> None:
        """`llm_cfg` picks which model this client talks to -- defaults to
        the general-purpose `config.llm` section, but callers needing a
        different deployment (e.g. the Legal tab's orchestrator/Hebrew-
        analyst models, see llm/client.py's `get_legal_*_client`) pass their
        own `LLMConfig` instead."""
        cfg = get_config()
        self._cfg = cfg
        self._llm_cfg = llm_cfg or cfg.llm
        self._backend = self._llm_cfg.backend
        self._chat_path = "/chat/completions" if self._backend == "vllm" else "/api/chat"
        self._client = httpx.AsyncClient(
            base_url=self._llm_cfg.base_url,
            timeout=self._llm_cfg.request_timeout_s,
            headers={"Authorization": f"Bearer {self._llm_cfg.api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _enable_thinking(self, call_site: LLMCallSite, override: bool | None) -> bool:
        if override is not None:
            return override
        return getattr(self._llm_cfg.thinking_defaults, call_site.name)

    def _build_payload(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None,
        enable_thinking: bool | None,
        guided_json_schema: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        sampling = sampling or SamplingParams(**self._llm_cfg.default_sampling.model_dump())
        thinking = self._enable_thinking(call_site, enable_thinking)
        common: dict[str, Any] = {
            "model": self._llm_cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
        }

        if self._backend == "vllm":
            payload: dict[str, Any] = {
                **common,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_tokens": sampling.max_tokens,
                "chat_template_kwargs": {"enable_thinking": thinking},
            }
            if guided_json_schema is not None:
                payload["extra_body"] = {
                    "guided_json": guided_json_schema,
                    "guided_decoding_backend": self._llm_cfg.guided_decoding_backend,
                }
            return payload

        # Ollama's native /api/chat: sampling params live under "options"
        # (Modelfile PARAMETER names), "think" toggles reasoning directly
        # (no chat-template-string juggling), and structured output is a
        # top-level "format" field holding the raw JSON Schema dict.
        payload = {
            **common,
            "think": thinking,
            "options": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "num_predict": sampling.max_tokens,
                "num_ctx": self._llm_cfg.max_model_len,
            },
        }
        if guided_json_schema is not None:
            payload["format"] = guided_json_schema
        return payload

    @staticmethod
    def _split_thinking(text: str) -> tuple[str, str]:
        """Split a full (non-streamed) completion into (reasoning, content)."""
        if "<think>" in text and "</think>" in text:
            start = text.index("<think>") + len("<think>")
            end = text.index("</think>")
            reasoning = text[start:end].strip()
            content = text[end + len("</think>") :].strip()
            return reasoning, content
        return "", text.strip()

    @property
    def model(self) -> str:
        return self._llm_cfg.model

    async def _post_completion(self, payload: dict[str, Any]) -> str:
        """Non-streamed completion; returns the content with any reasoning stripped."""
        resp = await self._client.post(self._chat_path, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if self._backend == "vllm":
            _, content = self._split_thinking(data["choices"][0]["message"]["content"])
            return content
        message = data["message"]
        content = message["content"]
        if not message.get("thinking"):
            # Older Ollama builds / models that ignore "think" still
            # emit reasoning inline as <think> tags in content.
            _, content = self._split_thinking(content)
        return content

    async def complete_text(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Free-text generation, for call sites whose output is prose that
        must not be forced through a JSON schema (e.g. DictaLM's normalized
        query / polished reply in legal/pipeline.py)."""
        payload = self._build_payload(
            messages, call_site, sampling, enable_thinking, guided_json_schema=None, stream=False
        )
        return (await self._post_completion(payload)).strip()

    async def complete_json(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        schema: type[BaseModel],
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
        max_retries: int | None = None,
    ) -> BaseModel:
        """Structured-JSON generation with schema validation + error-correction retry."""
        cfg = self._cfg
        attempts = 1 + (max_retries if max_retries is not None else cfg.slides.schema_retry_attempts)
        last_error: Exception | None = None
        working_messages = list(messages)

        for attempt in range(attempts):
            payload = self._build_payload(
                working_messages,
                call_site,
                sampling,
                enable_thinking,
                guided_json_schema=schema.model_json_schema(),
                stream=False,
            )
            content = await self._post_completion(payload)

            try:
                parsed = json.loads(content)
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = SchemaValidationFailed(raw=content, errors=str(exc))
                logger.warning(
                    "llm_schema_validation_failed",
                    call_site=call_site.name,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                working_messages = [
                    *messages,
                    ChatMessage(role="assistant", content=content),
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous response did not match the required JSON schema. "
                            f"Validation error: {exc}\n"
                            f"Required schema: {schema.model_json_schema()}\n"
                            "Return ONLY corrected JSON matching the schema, no commentary."
                        ),
                    ),
                ]

        assert last_error is not None
        raise last_error

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream a chat turn, yielding separate reasoning/content deltas."""
        payload = self._build_payload(
            messages, call_site, sampling, enable_thinking, guided_json_schema=None, stream=True
        )
        async with self._client.stream("POST", self._chat_path, json=payload) as resp:
            resp.raise_for_status()
            if self._backend == "vllm":
                async for delta in self._stream_vllm(resp):
                    yield delta
            else:
                async for delta in self._stream_ollama(resp):
                    yield delta

    @staticmethod
    async def _stream_vllm(resp: httpx.Response) -> AsyncIterator[StreamDelta]:
        """vLLM's OpenAI-compatible SSE stream embeds reasoning inline as
        <think>...</think> tags in the content deltas -- split on them as
        they arrive, buffering only across a tag boundary."""
        in_thinking = False
        thinking_closed = False
        buffer = ""

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content")
            if not token:
                continue
            buffer += token

            while buffer:
                if not in_thinking and not thinking_closed and buffer.lstrip().startswith("<think>"):
                    idx = buffer.index("<think>")
                    buffer = buffer[idx + len("<think>") :]
                    in_thinking = True
                    continue
                if in_thinking and "</think>" in buffer:
                    idx = buffer.index("</think>")
                    if idx > 0:
                        yield StreamDelta(kind="reasoning", text=buffer[:idx])
                    buffer = buffer[idx + len("</think>") :]
                    in_thinking = False
                    thinking_closed = True
                    continue
                kind = "reasoning" if in_thinking else "content"
                yield StreamDelta(kind=kind, text=buffer)
                buffer = ""

        if buffer:
            kind = "reasoning" if in_thinking else "content"
            yield StreamDelta(kind=kind, text=buffer)

    @staticmethod
    async def _stream_ollama(resp: httpx.Response) -> AsyncIterator[StreamDelta]:
        """Ollama's /api/chat stream is newline-delimited JSON (not SSE); each
        line already separates `message.thinking` from `message.content`, so
        no tag-splitting is needed here -- only models that actually honor
        the "think" request field populate `thinking` distinctly."""
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            message = chunk.get("message") or {}
            thinking = message.get("thinking")
            if thinking:
                yield StreamDelta(kind="reasoning", text=thinking)
            content = message.get("content")
            if content:
                yield StreamDelta(kind="content", text=content)
            if chunk.get("done"):
                break


_client_singleton: QwenClient | None = None
_legal_orchestrator_singleton: QwenClient | None = None
_legal_dicta_clients: dict[str, QwenClient] = {}
_canon_generation_singleton: QwenClient | None = None


def get_client() -> QwenClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = QwenClient()
    return _client_singleton


def get_legal_orchestrator_client() -> QwenClient:
    """Qwen: the Legal tab's research, drafting and verification model. See
    legal/pipeline.py."""
    global _legal_orchestrator_singleton
    if _legal_orchestrator_singleton is None:
        _legal_orchestrator_singleton = QwenClient(get_config().legal.orchestrator)
    return _legal_orchestrator_singleton


def get_legal_dicta_client(tier: str) -> QwenClient:
    """DictaLM at the user-selected tier (config.legal.dicta_tiers): Hebrew
    query normalization and Hebrew polish only. See legal/pipeline.py."""
    if tier not in _legal_dicta_clients:
        _, tier_cfg = get_config().legal.dicta_tier(tier)
        _legal_dicta_clients[tier] = QwenClient(tier_cfg.llm)
    return _legal_dicta_clients[tier]


def get_canon_generation_client() -> QwenClient:
    """Answers Canon GPT questions grounded in retrieved canon-law chunks.
    See canon/pipeline.py."""
    global _canon_generation_singleton
    if _canon_generation_singleton is None:
        _canon_generation_singleton = QwenClient(get_config().canon.generation)
    return _canon_generation_singleton


async def aclose_all_clients() -> None:
    """Closes whichever of the above singletons were actually instantiated,
    without creating new ones just to close them -- called once at app
    shutdown (see api/main.py's lifespan)."""
    for client in (
        _client_singleton,
        _legal_orchestrator_singleton,
        *_legal_dicta_clients.values(),
        _canon_generation_singleton,
    ):
        if client is not None:
            await client.aclose()
