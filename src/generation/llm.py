"""LLM backends. Default: Google Gemini (google-genai SDK).

All backends expose the same tiny interface:
    chat(messages, max_tokens=None, temperature=None) -> str
    stream(messages, ...) -> Iterator[str]
where ``messages`` is a list of {"role": "system"|"user"|"assistant", "content": str}.
"""
from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional

from src.config import settings
from src.logging_setup import get_logger

log = get_logger(__name__)
Messages = List[Dict[str, str]]


class LLMUnavailable(RuntimeError):
    pass


class BaseLLM:
    name = "base"
    is_generative = True

    def chat(self, messages: Messages, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        return "".join(self.stream(messages, max_tokens=max_tokens, temperature=temperature))

    def stream(self, messages: Messages, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Iterator[str]:
        raise NotImplementedError


# ── Gemini ────────────────────────────────────────────────────────────────
class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, fallbacks: Optional[List[str]] = None):
        if not api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not set.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise LLMUnavailable("Package 'google-genai' is not installed (pip install google-genai).") from e
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.models = [model] + [m for m in (fallbacks or []) if m and m != model]
        self.name = f"Gemini · {model}"

    def _config(self, system: Optional[str], max_tokens: int, temperature: float, thinking: bool):
        kwargs = dict(
            system_instruction=system or None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if thinking and settings.GEMINI_THINKING_LEVEL:
            try:
                kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_level=settings.GEMINI_THINKING_LEVEL)
            except Exception:  # SDK/model without thinking_level support
                pass
        return self._types.GenerateContentConfig(**kwargs)

    def _contents(self, messages: Messages):
        types = self._types
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
        return contents

    def stream(self, messages: Messages, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Iterator[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = self._contents(messages)
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        temperature = settings.LLM_TEMPERATURE if temperature is None else temperature

        last_err: Optional[Exception] = None
        for model in self.models:
            use_thinking = bool(settings.GEMINI_THINKING_LEVEL)
            attempt = 0
            while attempt <= settings.LLM_MAX_RETRIES:
                emitted = False
                try:
                    cfg = self._config(system, max_tokens, temperature, use_thinking)
                    for chunk in self._client.models.generate_content_stream(model=model, contents=contents, config=cfg):
                        text = getattr(chunk, "text", None)
                        if text:
                            emitted = True
                            yield text
                    return
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if emitted:  # partial answer already shown: never restart mid-stream
                        raise
                    msg = str(e).lower()
                    if use_thinking and ("thinking" in msg or "invalid" in msg or "400" in msg):
                        use_thinking = False  # model rejects thinking_config: retry once without it
                        continue
                    if any(k in msg for k in ("429", "500", "503", "unavailable", "overloaded", "deadline")):
                        attempt += 1
                        time.sleep(1.5 * attempt)
                        continue
                    break  # not retryable on this model -> try next model
            log.warning("Gemini model %s failed: %s", model, last_err)
        raise LLMUnavailable(f"All Gemini models failed. Last error: {last_err}")


# ── Groq (optional) ───────────────────────────────────────────────────────
class GroqLLM(BaseLLM):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMUnavailable("GROQ_API_KEY is not set.")
        try:
            from groq import Groq
        except ImportError as e:
            raise LLMUnavailable("Package 'groq' is not installed.") from e
        self._client = Groq(api_key=api_key)
        self.model = model
        self.name = f"Groq · {model}"

    def stream(self, messages: Messages, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Iterator[str]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── No-LLM fallback ───────────────────────────────────────────────────────
class ExtractiveLLM(BaseLLM):
    """Marker backend: the pipeline builds an extractive answer itself (no generation)."""

    name = "Extractive (no LLM)"
    is_generative = False

    def stream(self, messages, max_tokens=None, temperature=None):
        raise LLMUnavailable("Extractive backend does not generate text.")


def build_llm() -> BaseLLM:
    """Create the configured backend; degrade to extractive mode (with a warning) rather than crash."""
    backend = settings.LLM_BACKEND
    try:
        if backend == "gemini":
            fallbacks = [m.strip() for m in settings.GEMINI_FALLBACK_MODELS.split(",") if m.strip()]
            return GeminiLLM(settings.GEMINI_API_KEY, settings.GEMINI_MODEL, fallbacks)
        if backend == "groq":
            return GroqLLM(settings.GROQ_API_KEY, settings.GROQ_MODEL)
        if backend != "extractive":
            log.warning("Unknown LLM_BACKEND=%r; using extractive mode.", backend)
    except LLMUnavailable as e:
        log.warning("LLM backend '%s' unavailable (%s). Falling back to extractive answers.", backend, e)
    return ExtractiveLLM()
