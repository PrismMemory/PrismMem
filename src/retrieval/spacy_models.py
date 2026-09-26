"""Lazy, process-wide spaCy loading for BM25 keyword normalization."""

from __future__ import annotations

import logging
import threading
from typing import Any

try:
    import config
except ImportError:  # pragma: no cover - makes this module independently inspectable.
    config = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_nlp_lemma: Any | None = None
_lock = threading.Lock()


def _cfg(name: str, default: object) -> object:
    return getattr(config, name, default) if config is not None else default


def _ensure_model_available() -> None:
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is not installed. Install the package's nlp dependencies."
        ) from exc

    model_name = str(_cfg("BM25_LEMMATIZATION_MODEL", "en_core_web_sm"))
    if spacy.util.is_package(model_name):
        return
    if not bool(_cfg("BM25_LEMMATIZATION_AUTO_DOWNLOAD", True)):
        raise RuntimeError(
            f"spaCy model {model_name} is not installed; run "
            f"python -m spacy download {model_name}"
        )

    logger.info("Downloading spaCy model %s...", model_name)
    try:
        from spacy.cli import download

        download(model_name)
    except Exception as exc:  # noqa: BLE001 - optional dependency fallback below.
        raise RuntimeError(
            f"Failed to download spaCy model {model_name}: {exc}"
        ) from exc


def get_nlp_lemma() -> Any:
    """Return the configured lemmatizer; never silently change retrieval semantics."""
    global _nlp_lemma
    if _nlp_lemma is not None:
        return _nlp_lemma
    with _lock:
        if _nlp_lemma is not None:
            return _nlp_lemma
        try:
            _ensure_model_available()
            import spacy
            model_name = str(_cfg("BM25_LEMMATIZATION_MODEL", "en_core_web_sm"))
            _nlp_lemma = spacy.load(model_name, disable=["ner", "parser"])
        except Exception:
            raise RuntimeError(
                "BM25 lemmatization is enabled but unavailable. Run "
                "python -m spacy download en_core_web_sm, or explicitly disable "
                "lemmatization in a separate benchmark configuration."
            ) from None
    return _nlp_lemma


def reset_nlp_lemma_for_tests() -> None:
    """Reset process-wide loader state for deterministic tests."""
    global _nlp_lemma

    with _lock:
        _nlp_lemma = None
        