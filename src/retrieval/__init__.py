"""Representation-isolated retrieval across three channels with strict reranking."""

from .projection import (
    answer_text,
    answer_content_with_source_evidence,
    long_retrieval_text,
    long_retrieval_content,
    mid_embedding_content,
    mid_index_text,
    mid_retrieval_content,
    source_evidence_text,
    source_evidence_content,
    text_sha256,
)
from .result import RetrievalResult


def __getattr__(name: str):
    # Pure projection users (including benchmark scoring) must not load config,
    # dotenv or model transports merely by importing this package.
    if name in {"MODE", "search_rrf90"}:
        from . import rrf90
        value = getattr(rrf90, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MODE",
    "RetrievalResult",
    "answer_content_with_source_evidence",
    "answer_text",
    "long_retrieval_content",
    "long_retrieval_text",
    "mid_embedding_content",
    "mid_index_text",
    "mid_retrieval_content",
    "search_rrf90",
    "source_evidence_content",
    "source_evidence_text",
    "text_sha256",
]
