"""JSON-serializable retrieval result."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """Selected memories and a complete explanation of their selection."""

    memories: list[dict]
    trace: dict
