"""Text projections for memory retrieval and answer generation.
"""

from __future__ import annotations

import hashlib


def _text_items(value: object) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in values if str(item or "").strip()]


def _text_field(record: dict, *names: str) -> str:
    for name in names:
        value = str(record.get(name) or "").strip()
        if value:
            return value
    return "unknown"


def mid_embedding_content(record: dict) -> str:
    """Combine the topic and summary for semantic-scope dense embedding input."""
    return " ".join(
        part
        for part in (
            str(record.get("topic_subject") or record.get("topic") or "").strip(),
            str(record.get("summary") or "").strip(),
        )
        if part
    )


def mid_retrieval_content(record: dict) -> str:
    """Semantic-scope BM25/rerank projection, also used for LoCoMo answer context.
    """
    return "\n".join(
        (
            f"topic_subject: {_text_field(record, 'topic_subject', 'topic')}",
            f"summary: {_text_field(record, 'summary')}",
            f"intent_primary: {_text_field(record, 'intent_primary')}",
            f"intent_secondary: {_text_field(record, 'intent_secondary')}",
            f"intent_description: {_text_field(record, 'intent_description')}",
        )
    )


def long_retrieval_content(record: dict) -> str:
    """Facet text used by embedding, BM25 and rerank.
    """
    content = str(record.get("content") or "").strip()
    if not content:
        return ""
    memory_type = str(record.get("type") or "").strip().lower()
    user_id = str(record.get("user_id") or "").strip()
    parts: list[str] = []
    if memory_type == "core":
        entity_name = str(
            record.get("entityName") or record.get("entity_name") or ""
        ).strip()
        fact_subject = (
            user_id if entity_name.casefold() == "user" and user_id else entity_name
        )
        parts.append(f"Fact subject: {fact_subject or 'unknown'}")
    parts.extend(
        (
            f"Memory owner: {user_id or 'unknown'}",
            f"Content: {content}",
        )
    )
    if memory_type == "episodic":
        context = str(record.get("context") or "").strip()
        if context:
            parts.append(f"Context: {context}")
        parts.extend(
            (
                f"Event time: {str(record.get('eventTime') or 'unknown').strip()}",
                f"Mention time: {str(record.get('mentionTime') or 'unknown').strip()}",
            )
        )
    elif memory_type == "knowledge":
        name = str(record.get("name") or "").strip()
        if name:
            parts.insert(-1, f"Name: {name}")
    return "\n".join(parts)


def source_evidence_content(record: dict) -> str:
    """Render source quotes only for the final answer prompt."""
    raw_evidence = record.get("sourceEvidence") or record.get("source_evidence") or []
    evidence_items = (
        list(raw_evidence)
        if isinstance(raw_evidence, (list, tuple))
        else [raw_evidence]
    )
    lines: list[str] = []
    for item in evidence_items:
        if isinstance(item, dict):
            chat_id = " ".join(
                str(item.get("chatId") or item.get("chat_id") or "").split()
            )
            quote = " ".join(
                str(
                    item.get("quote")
                    or item.get("verbatimSpan")
                    or item.get("text")
                    or ""
                ).split()
            )
        else:
            chat_id = ""
            quote = " ".join(str(item or "").split())
        details = []
        if chat_id:
            details.append(f"chatId: {chat_id}")
        if quote:
            details.append(f"quote: {quote}")
        if details:
            lines.append("- " + " | ".join(details))
    return "\n".join(["sourceEvidence:", *lines]) if lines else ""


def answer_content_with_source_evidence(content: str, record: dict) -> str:
    """Append provenance once, after retrieval has finished."""
    base = str(content or "").strip()
    evidence = source_evidence_content(record)
    if not evidence or evidence in base:
        return base
    return "\n".join(part for part in (base, evidence) if part)


def answer_memory_content(record: dict) -> str:
    """Render a selected row exactly as it should enter the answer prompt."""
    retrieval_text = str(record.get("content") or "").strip()
    return answer_content_with_source_evidence(retrieval_text, record)


# Stable public names used by extraction/backfill/evaluation entry points.  Keep the
# longer names above as readable aliases for callers.
def mid_index_text(record: dict) -> str:
    return mid_embedding_content(record)


def long_retrieval_text(record: dict) -> str:
    return long_retrieval_content(record)


def source_evidence_text(record: dict) -> str:
    return source_evidence_content(record)


def answer_text(record: dict) -> str:
    """Return LoCoMo answer-visible text for a selected semantic scope or facet."""
    is_mid = str(record.get("memory_pool") or "").lower() == "mid" or (
        not str(record.get("type") or "").strip()
        and bool(record.get("topic_subject") or record.get("summary"))
    )
    if is_mid:
        # Selected rows already have the BM25/rerank projection in ``content``.
        selected_text = str(record.get("content") or "").strip()
        return selected_text or mid_retrieval_content(record)

    # Selected rows already contain the retrieval projection in ``content``; raw store
    # records do not.  The marker avoids rendering "Content: Memory owner: ...".
    selected_long = str(record.get("content") or "").strip()
    if record.get("memory_pool") in {"core", "other"}:
        retrieval_text = selected_long
    else:
        retrieval_text = long_retrieval_text(record)
    return answer_content_with_source_evidence(retrieval_text, record)


def text_sha256(text: str) -> str:
    """Hash the exact UTF-8 projection stored alongside each facet embedding."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
