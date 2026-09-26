"""Pure-Python Okapi BM25 with optional lemmatization."""

from __future__ import annotations

from functools import lru_cache
import math
import re

try:
    import config
except ImportError:  # pragma: no cover - makes this module independently inspectable.
    config = None  # type: ignore[assignment]

from .lemmatization import lemmatize_for_bm25

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _cfg(name: str, default: object) -> object:
    return getattr(config, name, default) if config is not None else default


def tokenize_legacy(text: str) -> list[str]:
    """Lowercase word/number tokenization when lemmatization is disabled."""
    return _TOKEN_RE.findall((text or "").lower())


@lru_cache(maxsize=16384)
def _lemmatize_cached(text: str) -> str:
    return lemmatize_for_bm25(text)


def tokenize(text: str, *, lemmatize: bool | None = None) -> list[str]:
    """Tokenize with keyword normalization enabled by default."""
    enabled = (
        bool(_cfg("BM25_LEMMATIZATION_ENABLED", True))
        if lemmatize is None
        else lemmatize
    )
    source = text or ""
    if not enabled:
        return tokenize_legacy(source)
    return _TOKEN_RE.findall(_lemmatize_cached(source).lower())


def clear_tokenization_cache() -> None:
    _lemmatize_cached.cache_clear()


class BM25:
    """Score a fixed tokenized corpus against ad-hoc tokenized queries."""

    def __init__(self, corpus: list[list[str]]):
        self.corpus = corpus
        self.n = len(corpus)
        self.doc_len = [len(document) for document in corpus]
        self.avg_len = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.df: dict[str, int] = {}
        for document in corpus:
            for term in set(document):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {
            term: math.log(1 + (self.n - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in self.df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        """Return scores in original corpus order."""
        k1 = float(_cfg("BM25_K1", 1.5))
        b = float(_cfg("BM25_B", 0.75))
        output = [0.0] * self.n
        if not self.avg_len:
            return output
        for index, document in enumerate(self.corpus):
            if not document:
                continue
            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            norm = k1 * (1 - b + b * self.doc_len[index] / self.avg_len)
            score = 0.0
            for term in query:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (k1 + 1)
                ) / (frequency + norm)
            output[index] = score
        return output
