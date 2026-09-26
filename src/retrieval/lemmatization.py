"""Keyword normalization for BM25 retrieval."""

from __future__ import annotations

from .spacy_models import get_nlp_lemma


def lemmatize_for_bm25(text: str) -> str:
    """Return space-joined lemmas from the configured local spaCy model.

    """
    source = text or ""
    nlp = get_nlp_lemma()
    if nlp is None:
        raise RuntimeError("Configured BM25 lemmatizer is unavailable")

    doc = nlp(source.lower())
    tokens: list[str] = []
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)
    return " ".join(tokens)
