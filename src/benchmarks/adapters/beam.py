"""Load BEAM's official chat batches and independent probing questions.

Source: https://github.com/mohammadtavakoli78/BEAM
Accept ``chats/1M`` (35 conversations, 700 questions in the original release),
a repository root, or an explicitly marked synthetic smoke JSON envelope.
Only chat.json enters memory construction. Rubrics and every reference field
remain in the separate questions output and are available to scoring only.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path


SOURCE_URL = "https://github.com/mohammadtavakoli78/BEAM"
PROTOCOL = "beam-full-history-per-conversation-oss-v1"
_GOLD_FIELDS = {
    "abstention": "ideal_response", "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer", "information_extraction": "answer",
    "instruction_following": "expected_compliance", "knowledge_update": "answer",
    "multi_session_reasoning": "answer", "preference_following": "expected_compliance",
    "summarization": "ideal_summary", "temporal_reasoning": "answer",
}


def _positive_limit(options: dict, key: str) -> int | None:
    value = options.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"BEAM {key} must be a positive integer")
    return value


def _fingerprint(path: Path, conversation_id: str | None = None) -> dict:
    raw = path.read_bytes()
    result = {"name": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if conversation_id is not None:
        result["conversation_id"] = conversation_id
    return result


def _date(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    for fmt in ("%B-%d-%Y", "%b-%d-%Y", "%Y-%m-%d", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def _flatten_turns(turns: list) -> list[dict]:
    result = []
    for turn in turns:
        if isinstance(turn, list):
            result.extend(_flatten_turns(turn))
        elif isinstance(turn, dict):
            result.append(turn)
        else:
            raise ValueError("BEAM turns must contain message objects or nested message lists")
    return result


def _source_ids(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _source_ids(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _source_ids(nested)]
    return [str(value)] if isinstance(value, (int, str)) else []


def _conversation(chat: list, conversation_id: str) -> tuple[dict, dict[str, list[str]], str]:
    if not isinstance(chat, list) or not chat:
        raise ValueError(f"BEAM {conversation_id}: chat must be a nonempty batch list")
    conversation: dict = {"speaker_a": "User", "speaker_b": "Assistant"}
    id_mapping: dict[str, list[str]] = {}
    seen_sessions: set[int] = set()
    dates = []
    for batch in chat:
        if not isinstance(batch, dict) or not isinstance(batch.get("turns"), list):
            raise ValueError(f"BEAM {conversation_id}: each batch requires turns")
        number = int(batch["batch_number"])
        if number < 0 or number in seen_sessions:
            raise ValueError(f"BEAM {conversation_id}: duplicate or negative batch number")
        seen_sessions.add(number)
        turns = _flatten_turns(batch["turns"])
        if not turns:
            raise ValueError(f"BEAM {conversation_id}: empty batch")
        date = _date(batch.get("time_anchor") or next((t["time_anchor"] for t in turns if t.get("time_anchor")), ""))
        if date:
            dates.append(date)
        converted = []
        seen_batch_ids: set[str] = set()
        for turn in turns:
            role = str(turn.get("role", "")).lower()
            if role not in {"user", "assistant", "system"} or not isinstance(turn.get("content"), str):
                raise ValueError(f"BEAM {conversation_id}: invalid message role or content")
            raw_id = str(turn["id"])
            dia_id = f"D{number}:{raw_id}"
            if raw_id in seen_batch_ids:
                raise ValueError(f"BEAM {conversation_id}: duplicate source turn ID {raw_id!r}")
            seen_batch_ids.add(raw_id)
            # The official release can restart numeric IDs in a later batch
            # (e.g. conversation 5, batch 10). Keep D<batch>:<id> unique and
            # report ambiguous evidence instead of silently choosing a turn.
            id_mapping.setdefault(raw_id, []).append(dia_id)
            id_mapping[dia_id] = [dia_id]
            converted.append({"dia_id": dia_id, "speaker": role.capitalize(), "text": turn["content"], "source_turn_id": turn["id"], "source_role": role})
        conversation[f"session_{number}"] = converted
        conversation[f"session_{number}_date_time"] = date
    return {"sample_id": conversation_id, "conversation": conversation}, id_mapping, dates[-1] if dates else ""


def prepare(input_path: Path, *, options: dict | None = None) -> dict:
    """Prepare BEAM using full history within each conversation's own scope.

    ``size`` defaults to ``1M`` when a repository root is passed. Limits select
    conversations and questions in source order without shortening histories. No question
    evidence IDs are used to select or truncate the construction history.
    JSON input is a container with ``conversations`` rows containing
    ``conversation_id``, ``chat`` and ``probing_questions``; it carries an
    explicit ``dataset_kind`` so smoke fixtures cannot look like real results.
    """
    input_path = Path(input_path)
    options = options or {}
    unknown = set(options) - {"size", "limit_questions", "limit_conversations"}
    if unknown:
        raise ValueError(f"Unknown BEAM prepare options: {sorted(unknown)}")
    question_limit = _positive_limit(options, "limit_questions")
    conversation_limit = _positive_limit(options, "limit_conversations")
    size = str(options.get("size", "1M"))
    dataset_kind = "benchmark"
    source_files = []
    if input_path.is_file():
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("conversations"), list):
            raise ValueError("BEAM JSON input requires a conversations envelope; use chats/1M for official data")
        dataset_kind = str(payload.get("dataset_kind", "converted_benchmark"))
        records = payload["conversations"]
        source_files.append(_fingerprint(input_path))
    else:
        root = input_path / "chats" / size if (input_path / "chats" / size).is_dir() else input_path
        paths = sorted(root.glob("*/chat.json"), key=lambda p: (not p.parent.name.isdigit(), int(p.parent.name) if p.parent.name.isdigit() else p.parent.name))
        records = []
        for path in paths:
            questions_path = path.parent / "probing_questions" / "probing_questions.json"
            if not questions_path.is_file():
                raise ValueError(f"BEAM missing probing questions for conversation {path.parent.name}")
            conversation_id = f"beam-{size}-{path.parent.name}"
            source_files.extend([_fingerprint(path, conversation_id), _fingerprint(questions_path, conversation_id)])
            records.append({"conversation_id": conversation_id, "chat": json.loads(path.read_text(encoding="utf-8")), "probing_questions": json.loads(questions_path.read_text(encoding="utf-8"))})
    if not records:
        raise ValueError("BEAM input contains no conversations")
    conversations, questions = [], []
    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("BEAM conversation envelope rows must be objects")
        conversation_id = str(record.get("conversation_id", "")).strip()
        if not conversation_id or conversation_id in seen_ids:
            raise ValueError("BEAM conversation IDs must be nonempty and unique")
        seen_ids.add(conversation_id)
        corpus, id_mapping, last_date = _conversation(record["chat"], conversation_id)
        conversations.append(corpus)
        grouped = record["probing_questions"]
        if not isinstance(grouped, dict) or not grouped:
            raise ValueError("BEAM probing questions must map category names to question lists")
        qa_index = 0
        for category, rows in grouped.items():
            if category not in _GOLD_FIELDS or not isinstance(rows, list):
                raise ValueError(f"BEAM unknown category or malformed question list: {category}")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
                    raise ValueError(f"BEAM {conversation_id}: malformed question")
                rubric = row.get("rubric")
                if not isinstance(rubric, list) or not rubric or any(not isinstance(x, str) or not x.strip() for x in rubric):
                    raise ValueError(f"BEAM {conversation_id} question {qa_index}: a nonempty string rubric list is required")
                gold_field = _GOLD_FIELDS[category]
                if gold_field not in row:
                    raise ValueError(f"BEAM {conversation_id} question {qa_index}: missing reference field {gold_field}")
                raw_evidence = list(dict.fromkeys(_source_ids(row.get("source_chat_ids", []))))
                questions.append({
                    "id": f"{conversation_id}::q{qa_index}", "conversation_id": conversation_id,
                    "question": row["question"].strip(), "answer": row[gold_field],
                    "category": category, "question_date": last_date,
                    "evidence": [id_mapping[x][0] for x in raw_evidence if len(id_mapping.get(x, [])) == 1],
                    "metadata": {**row, "qa_index": qa_index, "reference_field": gold_field,
                        "unresolved_source_chat_ids": [x for x in raw_evidence if x not in id_mapping],
                        "ambiguous_source_chat_ids": {x: id_mapping[x] for x in raw_evidence if len(id_mapping.get(x, [])) > 1},
                        "history_policy": "full chat history for this conversation only"},
                })
                qa_index += 1
    source_counts = {"conversations": len(conversations), "questions": len(questions), "by_category": dict(Counter(q["category"] for q in questions))}
    conversations = conversations[:conversation_limit]
    selected_ids = {sample["sample_id"] for sample in conversations}
    questions = [q for q in questions if q["conversation_id"] in selected_ids][:question_limit]
    used_ids = {q["conversation_id"] for q in questions}
    conversations = [sample for sample in conversations if sample["sample_id"] in used_ids]
    return {
        "conversations": conversations, "questions": questions,
        "metadata": {"benchmark": "beam", "size": size, "source_url": SOURCE_URL,
            "protocol": PROTOCOL, "dataset_kind": dataset_kind,
            "synthetic_smoke": dataset_kind == "synthetic_smoke",
            "source_files": source_files,
            "source_counts": source_counts,
            "prepared_counts": {"conversations": len(conversations), "questions": len(questions), "by_category": dict(Counter(q["category"] for q in questions))},
            "filters": {"limit_questions": question_limit, "limit_conversations": conversation_limit, "history_truncated": False},
            "question_count": len(questions), "conversation_count": len(conversations),
            "category_counts": dict(Counter(q["category"] for q in questions)),
            "history_policy": "full per-conversation chat; independent probing questions excluded from memory construction",
            "reference_policy": "category-specific reference field plus full original rubric metadata retained for scoring only",
            "paper_result_parity": False},
    }
