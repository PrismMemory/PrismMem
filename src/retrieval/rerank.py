"""RRF ordering and strict score-bearing qwen3 reranking."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import Callable

try:
    import config
except ImportError:  # pragma: no cover - makes this module independently inspectable.
    config = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _cfg(name: str, default: object) -> object:
    return getattr(config, name, default) if config is not None else default


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = 60
) -> list[str]:
    """Fuse rankings with ``1 / (k + rank)`` where rank is explicitly 0-based."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    # Python's sort is stable. Ties therefore preserve first occurrence, which is the
    # embedding ranking because that route is inserted first by rrf90.py.
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)


def _result_index(item: object) -> int | None:
    if isinstance(item, dict):
        return item.get("index")
    return getattr(item, "index", None)


def _result_score(item: object) -> float | None:
    if isinstance(item, dict):
        value = item.get("relevance_score", item.get("score"))
    else:
        value = getattr(item, "relevance_score", getattr(item, "score", None))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _rerank_endpoint() -> tuple[str | None, str | None]:
    """Resolve credentials through the package provider registry, never hard-code them."""
    try:
        from llm import providers

        model = str(_cfg("RERANK_MODEL", "qwen3-rerank"))
        endpoint = providers.provider_for(model).endpoint
        return endpoint.base_url, endpoint.api_key
    except (ImportError, ValueError, AttributeError):
        return None, None


_NATIVE_RERANK_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"
_SESSION_LOCK = threading.Lock()
_SESSION = None


def _http_session():
    """One pooled session; a per-call session would re-handshake TLS on every batch."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            import requests
            from requests.adapters import HTTPAdapter

            pool = int(_cfg("RERANK_HTTP_POOL_MAXSIZE", 32))
            session = requests.Session()
            # max_retries stays 0: rerank_scored owns the retry policy.
            adapter = HTTPAdapter(
                pool_connections=pool, pool_maxsize=pool, max_retries=0
            )
            session.mount("https://", adapter)
            _SESSION = session
        return _SESSION


def _native_rerank_url(base_url: str) -> str:
    """Accept either the deployment root or its OpenAI-compatible base URL."""
    root = base_url.rstrip("/")
    for suffix in ("/compatible-mode/v1", "/compatible-mode", "/api/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root + _NATIVE_RERANK_PATH


def _call_rerank(query: str, documents: list[str], api_key: str, base_url: str | None):
    model = str(_cfg("RERANK_MODEL", "qwen3-rerank"))
    if not base_url:
        from dashscope import TextReRank

        return TextReRank.call(
            model=model,
            query=query,
            documents=documents,
            top_n=len(documents),
            return_documents=False,
            api_key=api_key,
        )

    # A dedicated deployment is addressed over the same native rerank route the SDK
    # uses, so only the quota and the connection pool differ from the shared endpoint.
    response = _http_session().post(
        _native_rerank_url(base_url),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": len(documents), "return_documents": False},
        },
        timeout=float(_cfg("RERANK_TIMEOUT_SECONDS", 300.0)),
    )
    results = []
    if response.status_code == 200:
        results = ((response.json() or {}).get("output") or {}).get("results") or []
    return SimpleNamespace(
        status_code=response.status_code, output=SimpleNamespace(results=results)
    )


def rerank_scored(
    query: str,
    records: list[dict],
    text_of: Callable[[dict], str],
    top_n: int,
    *,
    min_score: float | None = None,
    batch_size: int | None = None,
) -> list[tuple[dict, float]]:
    """Strictly score the complete candidate pool and globally merge batch scores.

    Every candidate must receive a valid relevance score. A failed or incomplete
    response raises an error.
    """
    if not records or top_n <= 0:
        return []
    base_url, api_key = _rerank_endpoint()
    if not api_key:
        raise RuntimeError("rerank API key is required for scored reranking")

    size = int(batch_size or len(records))
    if size <= 0:
        raise ValueError("rerank batch_size must be positive")

    scored: list[tuple[int, dict, float]] = []
    attempts = max(1, int(_cfg("RERANK_MAX_RETRIES", 3)))
    for start in range(0, len(records), size):
        batch = records[start : start + size]
        documents = [text_of(record) or "" for record in batch]
        response = None
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = _call_rerank(query, documents, api_key, base_url)
                status = getattr(response, "status_code", None)
                if status != 200:
                    raise RuntimeError(f"rerank status={status}")
                break
            except Exception as exc:  # noqa: BLE001 - retried, then raised strictly.
                last_error = exc
                if attempt >= attempts:
                    raise RuntimeError(
                        f"scored rerank batch failed after {attempt} attempt(s)"
                    ) from exc
                logger.warning(
                    "scored rerank batch failed (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(float(_cfg("RERANK_RETRY_BACKOFF_SECONDS", 1)) * attempt)
        if response is None:
            raise RuntimeError("scored rerank returned no response") from last_error

        results = getattr(getattr(response, "output", None), "results", None) or []
        batch_scores: dict[int, float] = {}
        for item in results:
            index = _result_index(item)
            score = _result_score(item)
            if index is None or score is None or not 0 <= index < len(batch):
                continue
            batch_scores[index] = score
        if len(batch_scores) != len(batch):
            raise RuntimeError(
                "scored rerank response did not contain a valid score for every document"
            )
        scored.extend(
            (start + index, batch[index], score)
            for index, score in batch_scores.items()
        )

    scored.sort(key=lambda item: (-item[2], item[0]))
    if min_score is not None:
        scored = [item for item in scored if item[2] >= float(min_score)]
    return [(record, score) for _, record, score in scored[:top_n]]
