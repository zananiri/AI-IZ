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
    error-correction prompt on failure. An output cut off at max_tokens --
    at temperature 0, almost always a loop repeating one sentence -- is
    retried fresh with Qwen's recommended non-greedy sampling instead.
  * Keep every request inside the context window: max_tokens is capped so
    prompt + output fit max_model_len (Ollama otherwise shifts the context
    mid-answer, silently dropping the system prompt and evidence).
  * Record every call -- reasoning, output, stop reason, tokens -- in the
    trace (llm/trace.py).
  * Stream chat turns and split reasoning from content for the UI, using
    each backend's own wire format (SSE + <think> tags for vLLM; newline-
    delimited JSON with a separate `thinking` field for Ollama).
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from docslides.config import LLMConfig, get_config
from docslides.llm import trace
from docslides.logging_setup import get_logger

logger = get_logger(__name__)

# Output-budget arithmetic (see QwenClient._fit_context). Deliberately pessimistic
# about Hebrew/Arabic, which Qwen's tokenizer splits finely.
_CHARS_PER_TOKEN_NON_LATIN = 2.0
_CHARS_PER_TOKEN_LATIN = 3.5
_TOKENS_PER_MESSAGE = 8
_CONTEXT_MARGIN = 128
_MIN_OUTPUT_TOKENS = 256

# How much of a malformed reply goes back to the model with the correction request.
_MAX_ECHOED_CHARS = 8000

_RUNAWAY_NOTE = (
    "(Your previous reply to this was cut off at the length limit because it kept repeating itself. "
    "Reply again, concisely: state each point once, never repeat a sentence, and finish the JSON.)"
)


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
    max_tokens: int = 2048  # thinking tokens count against it too, on both backends
    top_k: int | None = None
    seed: int | None = None


@dataclass
class Completion:
    content: str
    reasoning: str = ""
    done_reason: str | None = None  # "stop", or "length" when max_tokens cut it off
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


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
        "legal_analysis",
        "legal_research_memo",
        "legal_draft",
        "legal_citation_verification",
        "legal_script_repair",
        "legal_eval_baseline",
        "legal_eval_judge",
        "canon_orchestration",
        "canon_answer",
    ]


def _with_note(messages: list[ChatMessage], note: str) -> list[ChatMessage]:
    """`messages` with `note` appended to the last user turn."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].role == "user":
            out[i] = ChatMessage("user", f"{out[i].content}\n\n{note}")
            return out
    return [*out, ChatMessage("user", note)]


def _loop_breaking(sampling: SamplingParams, attempt: int) -> SamplingParams:
    """Qwen3's recommended non-thinking sampling (temperature 0.7, top_p 0.8,
    top_k 20), with a new seed each retry so the next one isn't a replay."""
    return dataclasses.replace(
        sampling, temperature=max(sampling.temperature, 0.7), top_p=0.8, top_k=20,
        seed=(sampling.seed or 0) + attempt + 1,
    )


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
            payload.update({k: v for k, v in (("top_k", sampling.top_k), ("seed", sampling.seed)) if v is not None})
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
        extra = (("top_k", sampling.top_k), ("seed", sampling.seed))
        payload["options"].update({k: v for k, v in extra if v is not None})
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
        if text.lstrip().startswith("<think>"):  # cut off while still thinking: there is no answer
            return text.lstrip()[len("<think>") :].strip(), ""
        return "", text.strip()

    @property
    def model(self) -> str:
        return self._llm_cfg.model

    @staticmethod
    def _estimate_tokens(messages: list[ChatMessage]) -> int:
        chars = "".join(m.content for m in messages)
        non_latin = sum(1 for ch in chars if ord(ch) > 0x024F)
        return (
            int(non_latin / _CHARS_PER_TOKEN_NON_LATIN + (len(chars) - non_latin) / _CHARS_PER_TOKEN_LATIN)
            + _TOKENS_PER_MESSAGE * len(messages)
        )

    def _fit_context(self, messages: list[ChatMessage], sampling: SamplingParams) -> SamplingParams:
        """`sampling` with max_tokens lowered, if need be, so prompt + output fit
        max_model_len. Ollama runs with --context-shift: a generation that
        reaches num_ctx discards the oldest tokens -- the system prompt and the
        evidence -- and carries on without them. vLLM rejects the request."""
        budget = self._llm_cfg.max_model_len - self._estimate_tokens(messages) - _CONTEXT_MARGIN
        if sampling.max_tokens <= budget:
            return sampling
        capped = max(budget, _MIN_OUTPUT_TOKENS)
        logger.info("llm_max_tokens_capped", requested=sampling.max_tokens, capped=capped,
                    max_model_len=self._llm_cfg.max_model_len)
        return dataclasses.replace(sampling, max_tokens=capped)

    async def _post_completion(self, payload: dict[str, Any]) -> Completion:
        """Non-streamed completion, with the reasoning split from the content."""
        resp = await self._client.post(self._chat_path, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if self._backend == "vllm":
            choice = data["choices"][0]
            message = choice["message"]
            # A server started with --reasoning-parser returns the reasoning separately;
            # otherwise it arrives inline as <think> tags.
            inline_reasoning, content = self._split_thinking(message.get("content") or "")
            usage = data.get("usage") or {}
            return Completion(
                content=content,
                reasoning=message.get("reasoning_content") or message.get("reasoning") or inline_reasoning,
                done_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
        message = data["message"]
        content = message.get("content") or ""
        reasoning = message.get("thinking") or ""
        if not reasoning:
            # Older Ollama builds / models that ignore "think" still
            # emit reasoning inline as <think> tags in content.
            reasoning, content = self._split_thinking(content)
        return Completion(
            content=content,
            reasoning=reasoning,
            done_reason=data.get("done_reason"),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    async def _complete(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None,
        enable_thinking: bool | None,
        guided_json_schema: dict[str, Any] | None,
        attempt: int = 1,
    ) -> Completion:
        """One non-streamed call, recorded in the trace whatever its outcome."""
        sampling = self._fit_context(
            messages, sampling or SamplingParams(**self._llm_cfg.default_sampling.model_dump())
        )
        thinking = self._enable_thinking(call_site, enable_thinking)
        started = time.monotonic()
        completion: Completion | None = None
        error: str | None = None
        try:
            payload = self._build_payload(messages, call_site, sampling, thinking, guided_json_schema, stream=False)
            try:
                completion = await self._post_completion(payload)
            except httpx.HTTPStatusError as exc:
                if not thinking or exc.response.status_code != 400:
                    raise
                # A model or server that can't think ("does not support thinking"): answer without it.
                logger.warning("llm_thinking_unsupported", call_site=call_site.name, error=exc.response.text[:300])
                thinking = False
                payload = self._build_payload(messages, call_site, sampling, False, guided_json_schema, stream=False)
                completion = await self._post_completion(payload)
            return completion
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace.record(
                {
                    "call_site": call_site.name,
                    "model": self._llm_cfg.model,
                    "backend": self._backend,
                    "thinking": thinking,
                    "json": guided_json_schema is not None,
                    "attempt": attempt,
                    "sampling": dataclasses.asdict(sampling),
                    "seconds": round(time.monotonic() - started, 1),
                    "done_reason": completion.done_reason if completion else None,
                    "prompt_tokens": completion.prompt_tokens if completion else None,
                    "completion_tokens": completion.completion_tokens if completion else None,
                    "reasoning": completion.reasoning if completion else "",
                    "content": completion.content if completion else "",
                    "error": error,
                },
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )

    async def complete_text(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Free-text generation, for call sites whose output is prose that
        must not be forced through a JSON schema. With thinking on, the
        reasoning goes to the trace and only the answer is returned."""
        completion = await self._complete(messages, call_site, sampling, enable_thinking, guided_json_schema=None)
        return completion.content.strip()

    async def complete_json(
        self,
        messages: list[ChatMessage],
        call_site: LLMCallSite,
        schema: type[BaseModel],
        sampling: SamplingParams | None = None,
        enable_thinking: bool | None = None,
        max_retries: int | None = None,
        salvage: Callable[[str], BaseModel | None] | None = None,
    ) -> BaseModel:
        """Structured-JSON generation with schema validation + error-correction retry.

        `salvage` gets each failed reply, in order, once the retries are spent;
        the first model it recovers is returned instead of raising."""
        cfg = self._cfg
        attempts = 1 + (max_retries if max_retries is not None else cfg.slides.schema_retry_attempts)
        last_error: Exception | None = None
        failed_replies: list[str] = []
        working_messages = list(messages)
        working_sampling = sampling
        guided = schema.model_json_schema()

        for attempt in range(attempts):
            completion = await self._complete(
                working_messages, call_site, working_sampling, enable_thinking, guided, attempt=attempt + 1
            )
            content = completion.content

            try:
                parsed = json.loads(content)
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = SchemaValidationFailed(raw=content, errors=str(exc))
                failed_replies.append(content)
                logger.warning(
                    "llm_schema_validation_failed",
                    call_site=call_site.name,
                    attempt=attempt + 1,
                    truncated=completion.truncated,
                    error=str(exc),
                )
                if completion.truncated:
                    # Cut off at max_tokens. Greedy decoding (temperature 0) loops -- Qwen's own
                    # guidance warns of it -- and feeding the loop back would only prime it and eat
                    # the context: start over, sampling as Qwen recommends for non-thinking mode.
                    working_messages = _with_note(messages, _RUNAWAY_NOTE)
                    working_sampling = _loop_breaking(sampling or SamplingParams(), attempt)
                    continue
                working_messages = [
                    *messages,
                    ChatMessage(role="assistant", content=content[:_MAX_ECHOED_CHARS]),
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous response did not match the required JSON schema. "
                            f"Validation error: {exc}\n"
                            f"Required schema: {guided}\n"
                            "Return ONLY corrected JSON matching the schema, no commentary."
                        ),
                    ),
                ]

        if salvage is not None:
            for reply in failed_replies:
                rescued = salvage(reply)
                if rescued is not None:
                    logger.warning("llm_output_salvaged", call_site=call_site.name)
                    return rescued
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
        _canon_generation_singleton,
    ):
        if client is not None:
            await client.aclose()
