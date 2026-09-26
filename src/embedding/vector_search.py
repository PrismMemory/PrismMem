"""Dependency-free local cosine ranking."""

from __future__ import annotations

import math


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("cosine vectors must have equal dimensions")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def rank_by_cosine(
    query: list[float], items: list[tuple[str, list[float]]]
) -> list[tuple[str, float]]:
    scored = [(item_id, cosine(query, vector)) for item_id, vector in items]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored

