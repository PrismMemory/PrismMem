"""Remote text-vector generation; all similarity calculations remain local."""

from __future__ import annotations

import logging
import time

from openai import OpenAI

import config
from llm import providers

logger = logging.getLogger("prism.embedding")
_client: OpenAI | None = None


def _endpoint() -> providers.Endpoint:
    return providers.provider_for(config.PRISM_MODEL).endpoint


def available() -> bool:
    return bool(_endpoint().api_key)


def _get_client() -> OpenAI:
    global _client
    endpoint = _endpoint()
    if not endpoint.api_key:
        raise RuntimeError("embedding API key is not configured")
    if _client is None:
        _client = OpenAI(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            max_retries=0,
        )
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), config.PRISM_BATCH_SIZE):
        batch = texts[start : start + config.PRISM_BATCH_SIZE]
        response = None
        last_error: Exception | None = None
        for attempt in range(1, max(1, config.LLM_MAX_RETRIES) + 1):
            try:
                response = _get_client().embeddings.create(
                    model=config.PRISM_MODEL,
                    input=batch,
                    timeout=config.PRISM_TIMEOUT_SECONDS,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max(1, config.LLM_MAX_RETRIES):
                    break
                time.sleep(config.LLM_RETRY_BACKOFF_SECONDS * attempt)
        if response is None:
            raise RuntimeError("embedding request failed after retries") from last_error
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise RuntimeError(
                f"expected {len(batch)} embeddings, received {len(ordered)}"
            )
        vectors.extend([list(item.embedding) for item in ordered])
    return vectors


def embed_text(text: str) -> list[float]:
    vectors = embed_texts([text])
    return vectors[0]

