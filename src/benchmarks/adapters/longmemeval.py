"""Adapt original LongMemEval records with one isolated history per question.

The source format and abstention convention are documented at
https://github.com/xiaowu0162/LongMemEval . This adapter targets that 500-question
benchmark, not the separately released LongMemEval-V2.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


SOURCE_URL = "https://github.com/xiaowu0162/LongMemEval"
QUESTION_TYPES = frozenset({
    "single-session-user", "single-session-assistant", "single-session-preference",
    "multi-session", "temporal-reasoning", "knowledge-update",
})


def _date(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a LongMemEval timestamp string")
    try:
        return datetime.strptime(value.strip(), "%Y/%m/%d (%a) %H:%M").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"{label} must use YYYY/MM/DD (Day) HH:MM") from exc


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _limit(options: dict, name: str) -> int | None:
    value = options.get(name)
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _record(record: object, index: int) -> tuple[dict, dict]:
    label = f"LongMemEval record {index}"
    if not isinstance(record, dict):
        raise ValueError(f"{label} must be an object")
    source_id = _string(record.get("question_id"), f"{label}.question_id")
    question_text = _string(record.get("question"), f"{label}.question")
    category = record.get("question_type")
    if category not in QUESTION_TYPES:
        raise ValueError(f"{label}.question_type is not a recognized LongMemEval type")
    question_date = _date(record.get("question_date"), f"{label}.question_date")
    answer = record.get("answer")
    if answer is None or isinstance(answer, (dict, list)):
        raise ValueError(f"{label}.answer must be a scalar")
    fields = ("haystack_sessions", "haystack_dates", "haystack_session_ids")
    if any(not isinstance(record.get(field), list) for field in fields):
        raise ValueError(f"{label} haystack fields must be arrays")
    if not record[fields[0]] or len({len(record[field]) for field in fields}) != 1:
        raise ValueError(f"{label} haystack arrays must have equal non-zero lengths")
    source_session_ids = record["haystack_session_ids"]
    if any(not isinstance(s, str) or not s for s in source_session_ids):
        raise ValueError(f"{label}.haystack_session_ids must contain non-empty strings")
    evidence_ids = record.get("answer_session_ids")
    if not isinstance(evidence_ids, list) or any(not isinstance(s, str) for s in evidence_ids):
        raise ValueError(f"{label}.answer_session_ids must be an array of strings")
    if set(evidence_ids) - set(source_session_ids):
        raise ValueError(f"{label} contains answer_session_ids outside its haystack")

    # Raw ids can disclose `_abs`; session ids can disclose `answer_`. Only an
    # opaque stable case id and positional Dn ids may enter the memory corpus.
    conversation_id = "lme-" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
    conversation = {"speaker_a": "user", "speaker_b": "assistant"}
    session_id_map = []
    answer_turn_ids = []
    evidence_dialogue_ids = []
    for session_index, (session, date, session_id) in enumerate(zip(
        record["haystack_sessions"], record["haystack_dates"], source_session_ids
    ), start=1):
        session_label = f"{label}.haystack_sessions[{session_index - 1}]"
        if not isinstance(session, list) or not session:
            raise ValueError(f"{session_label} must be a non-empty array")
        conversation[f"session_{session_index}_date_time"] = _date(date, session_label + ".date")
        internal_session_id = f"D{session_index}"
        session_id_map.append({
            "internal_session_id": internal_session_id,
            "source_session_id": session_id,
            "source_session_index": session_index - 1,
            "source_date": date,
        })
        turns = []
        for turn_index, turn in enumerate(session, start=1):
            turn_label = f"{session_label}[{turn_index - 1}]"
            if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
                raise ValueError(f"{turn_label} must have role user or assistant")
            if not isinstance(turn.get("content"), str):
                raise ValueError(f"{turn_label}.content must be a string")
            if "has_answer" in turn and type(turn["has_answer"]) is not bool:
                raise ValueError(f"{turn_label}.has_answer must be boolean")
            dialogue_id = f"{internal_session_id}:{turn_index}"
            turns.append({"dia_id": dialogue_id, "speaker": turn["role"], "text": turn["content"]})
            if turn.get("has_answer") is True:
                answer_turn_ids.append(f"{session_id}_{turn_index}")
                evidence_dialogue_ids.append(dialogue_id)
        conversation[f"session_{session_index}"] = turns
    return (
        {"sample_id": conversation_id, "conversation": conversation},
        {
            "id": source_id,
            "conversation_id": conversation_id,
            "question": question_text,
            "answer": answer,
            "category": category,
            "question_date": question_date,
            "evidence": evidence_dialogue_ids,
            "metadata": {
                "source_question_id": source_id,
                "source_question_index": index,
                "source_question_date": record["question_date"],
                "abstention": source_id.endswith("_abs"),
                "answer_session_ids": list(evidence_ids),
                "answer_turn_ids": answer_turn_ids,
                "session_id_map": session_id_map,
            },
        },
    )


def prepare(input_path: Path, *, options: dict | None = None) -> dict:
    """Return isolated corpus histories, gold questions and protocol provenance.

    ``limit_questions`` and ``limit_conversations`` are positive integers selecting
    the first N source records (the smaller limit wins). They never limit sessions
    or turns. ``variant`` may be ``s``, ``m``, ``oracle`` or ``custom``; if omitted it
    is inferred from the official filename and otherwise reported as ``custom``.
    The adapter never constructs an oracle subset from evidence labels. Input order,
    role, content and dates are preserved, including repeated filler session IDs.

    Labels, raw source IDs, QA and evidence maps remain exclusively in ``questions``.
    One opaque conversation ID per source question prevents cross-question memory
    sharing and hides the source ``_abs`` suffix from memory construction.
    """
    opts = dict(options or {})
    unknown = set(opts) - {"limit_questions", "limit_conversations", "variant"}
    if unknown:
        raise ValueError(f"unsupported LongMemEval options: {sorted(unknown)}")
    question_limit = _limit(opts, "limit_questions")
    conversation_limit = _limit(opts, "limit_conversations")
    limits = [value for value in (question_limit, conversation_limit) if value is not None]
    effective_limit = min(limits) if limits else None
    path = Path(input_path)
    inferred = {
        "longmemeval_s_cleaned.json": "s", "longmemeval_s.json": "s",
        "longmemeval_m_cleaned.json": "m", "longmemeval_m.json": "m",
        "longmemeval_oracle.json": "oracle",
    }.get(path.name, "custom")
    variant = opts.get("variant", inferred)
    if variant not in {"s", "m", "oracle", "custom"}:
        raise ValueError("LongMemEval variant must be s, m, oracle or custom")
    raw = path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("LongMemEval input must be a non-empty JSON array")
    questions = []
    conversations = []
    source_ids: set[str] = set()
    source_type_counts: Counter = Counter()
    for index, record in enumerate(records):
        conversation, question = _record(record, index)
        if question["id"] in source_ids:
            raise ValueError(f"duplicate LongMemEval question_id: {question['id']}")
        source_ids.add(question["id"])
        source_type_counts[question["category"]] += 1
        if effective_limit is None or index < effective_limit:
            conversations.append(conversation)
            questions.append(question)
    return {
        "conversations": conversations,
        "questions": questions,
        "metadata": {
            "benchmark": "longmemeval",
            "adapter_schema_version": 1,
            "protocol": "longmemeval_independent_question_histories",
            "variant": variant,
            "variant_declared_by_option": "variant" in opts,
            "official_source": SOURCE_URL,
            "source": {"name": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "filters": {"limit_questions": question_limit, "limit_conversations": conversation_limit, "history_truncated": False, "include_abstention": True},
            "source_counts": {"conversations": len(records), "questions": len(records), "by_category": dict(sorted(source_type_counts.items()))},
            "prepared_counts": {"conversations": len(conversations), "questions": len(questions), "abstention": sum(q["metadata"]["abstention"] for q in questions), "by_category": dict(sorted(Counter(q["category"] for q in questions).items()))},
            "synthetic_smoke": all(record.get("_fixture") == "synthetic_smoke" for record in records),
            "history_order": "source order, with normalized timestamps and positional session IDs",
            "annotation_policy": "gold answers, has_answer, source IDs and evidence maps are excluded from corpus",
        },
    }
