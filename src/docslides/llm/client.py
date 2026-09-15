"""Thin client around vLLM's OpenAI-compatible /v1/chat/completions endpoint.

Responsibilities:
  * Toggle Qwen3's `enable_thinking` chat-template flag per call site.
  * Drive guided/structured JSON decoding (xgrammar or outlines backend) for
    every content-generation call -- callers never regex-parse free text.
  * Validate JSON responses against a pydantic schema and retry with an
    error-correction prompt on failure.
  * Stream chat turns and split the raw token stream into `reasoning` (the
    contents of <think>...</think>) and `content` deltas for the UI.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from docslides.config import get_config
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
    ]


class QwenClient:
    def __init__(self) -> None:
        cfg = get_config()
        self._cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.llm.base_url,
            timeout=cfg.llm.request_timeout_s,
            headers={"Authorization": f"Bearer {cfg.llm.api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _enable_thinking(self, call_site: LLMCallSite, override: bool | None) -> bool:
        if override is not None:
            return override
        return getattr(self._cfg.llm.thinking_defaults, call_site.name)

    def _build_payload(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None,
        enable_thinking: bool | None,
        guided_json_schema: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        sampling = sampling or SamplingParams(**self._cfg.llm.default_sampling.model_dump())
        payload: dict[str, Any] = {
            "model": self._cfg.llm.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": self._enable_thinking(call_site, enable_thinking)},
        }
        if guided_json_schema is not None:
            payload["extra_body"] = {
                "guided_json": guided_json_schema,
                "guided_decoding_backend": self._cfg.llm.guided_decoding_backend,
            }
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
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            _, content = self._split_thinking(raw_content)

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
        """Stream a chat turn, splitting <think>...</think> from the final answer."""
        payload = self._build_payload(
            messages, call_site, sampling, enable_thinking, guided_json_schema=None, stream=True
        )

        in_thinking = False
        thinking_closed = False
        buffer = ""

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
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

                # Drain buffer, splitting on <think>/</think> boundaries.
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


_client_singleton: QwenClient | None = None


def get_client() -> QwenClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = QwenClient()
    return _client_singleton
