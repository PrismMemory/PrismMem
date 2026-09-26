"""Small OpenAI-compatible chat client with strict retry behavior."""

from __future__ import annotations

import logging
import random
import threading
import time

from openai import OpenAI

import config
from . import providers

logger = logging.getLogger("prism.llm")
_clients: dict[tuple[str | None, str | None], OpenAI] = {}
_pace_lock = threading.Lock()
_next_request_at = 0.0
# Gateway wordings that mean "slow down" rather than "this request is invalid". The
# Some gateways report throttling as HTTP 400 with a nested 429 code.
_THROTTLE_MARKERS = (
    "tpm/rpm",
    "429",
    "rate limit",
    "ratelimit",
    "too many requests",
    "throttl",
)


def _is_throttled(exc: Exception) -> bool:
    text = str(exc).casefold()
    return any(marker in text for marker in _THROTTLE_MARKERS)


def _pace() -> None:
    """Keep the whole process under one request per configured interval."""
    interval = float(config.LLM_MIN_REQUEST_INTERVAL_SECONDS)
    if interval <= 0:
        return
    global _next_request_at
    while True:
        with _pace_lock:
            now = time.monotonic()
            if now >= _next_request_at:
                _next_request_at = now + interval
                return
            wait = _next_request_at - now
        time.sleep(wait)


def _retry_delay(attempt: int, exc: Exception) -> float:
    """Linear backoff for ordinary faults, exponential + jitter when throttled."""
    if not _is_throttled(exc):
        return config.LLM_RETRY_BACKOFF_SECONDS * attempt
    cooldown = float(config.LLM_RATE_LIMIT_COOLDOWN_SECONDS)
    return min(cooldown * (2 ** (attempt - 1)), 120.0) + random.uniform(0, cooldown)


def _client(endpoint: providers.Endpoint) -> OpenAI:
    providers.validate_endpoint(endpoint)
    key = (endpoint.base_url, endpoint.api_key)
    if key not in _clients:
        _clients[key] = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            max_retries=0,
        )
    return _clients[key]


def _request(
    messages: list[dict],
    *,
    model: str | None,
    max_tokens: int | None,
    timeout: int | None,
    max_retries: int | None,
    temperature: float | None,
    role: str | None = None,
):
    if model is not None and not isinstance(model, str):
        raise ValueError("Model name must be a string")
    if model is None or not model.strip():
        role = role or "extraction"
        models = {
            "extraction": config.EXTRACTION_MODEL,
            "answer": config.QA_ANSWER_MODEL,
            "judge": config.EVAL_JUDGE_MODEL,
        }
        if role not in models:
            raise ValueError("Unknown chat model role")
        model = models[role]
    else:
        model = model.strip()
    max_tokens = config.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    timeout = config.LLM_TIMEOUT_SECONDS if timeout is None else timeout
    max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if type(max_retries) is not int or max_retries <= 0:
        raise ValueError("max_retries must be a positive integer")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise ValueError("timeout must be positive")
    provider = providers.provider_for(model, role=role)
    if provider.protocol != "openai":
        raise ValueError(f"model {model!r} is not a chat model")
    for attempt in range(1, max(1, max_retries) + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if provider.extra_body:
                kwargs["extra_body"] = provider.extra_body
            _pace()
            return _client(provider.endpoint).chat.completions.create(**kwargs)
        except Exception as exc:  # provider SDKs expose several exception classes.
            if attempt >= max(1, max_retries):
                break
            delay = _retry_delay(attempt, exc)
            logger.warning(
                "chat request failed (model=%s, attempt=%d/%d%s); retrying in %.1fs",
                model,
                attempt,
                max_retries,
                ", throttled" if _is_throttled(exc) else "",
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"chat request failed after {max(1, max_retries)} attempt(s)"
    ) from None


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    role: str | None = None,
) -> str:
    response = _request(
        messages,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        role=role,
    )
    return str(response.choices[0].message.content or "")


def complete_with_usage(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    role: str | None = None,
) -> tuple[str, int]:
    response = _request(
        messages,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        role=role,
    )
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    return str(response.choices[0].message.content or ""), prompt_tokens


def get_response(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    temperature: float | None = None,
    role: str | None = None,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return complete(
        messages,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        temperature=temperature,
        role=role,
    )
