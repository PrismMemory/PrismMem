"""Prepare LoCoMo's question-independent histories and category 1--4 evaluation.

Input is the original ``locomo10.json`` JSON array documented at
https://github.com/snap-research/locomo . Dataset annotations (QA, observations,
event summaries and session summaries) are never copied into the corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


_SESSION = re.compile(r"session_([1-9][0-9]*)\Z")
_CATEGORIES = frozenset({"1", "2", "3", "4"})
SOURCE_URL = "https://github.com/snap-research/locomo"


def _positive_limit(options: dict, name: str) -> int | None:
    value = options.get(name)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _conversation(sample: dict, sample_id: str) -> dict:
    source = sample.get("conversation")
    if not isinstance(source, dict):
        raise ValueError(f"{sample_id}.conversation must be an object")
    conversation = {
        field: _nonempty(source.get(field), f"{sample_id}.{field}")
        for field in ("speaker_a", "speaker_b")
    }
    sessions = sorted(
        (int(match.group(1)), key)
        for key in source
        if (match := _SESSION.fullmatch(key))
    )
    if not sessions:
        raise ValueError(f"{sample_id} has no sessions")
    seen_turns: set[str] = set()
    for number, key in sessions:
        turns = source[key]
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{sample_id}.{key} must be a non-empty array")
        date_key = f"{key}_date_time"
        conversation[date_key] = _nonempty(
            source.get(date_key), f"{sample_id}.{date_key}"
        )
        clean_turns = []
        for index, turn in enumerate(turns):
            label = f"{sample_id}.{key}[{index}]"
            if not isinstance(turn, dict):
                raise ValueError(f"{label} must be an object")
            dialogue_id = _nonempty(turn.get("dia_id"), f"{label}.dia_id")
            if not re.fullmatch(fr"D{number}:[1-9][0-9]*", dialogue_id):
                raise ValueError(f"{label}.dia_id does not match its session")
            if dialogue_id in seen_turns:
                raise ValueError(f"duplicate dialogue id in {sample_id}: {dialogue_id}")
            seen_turns.add(dialogue_id)
            text = turn.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{label}.text must be a string")
            clean = {
                "dia_id": dialogue_id,
                "speaker": _nonempty(turn.get("speaker"), f"{label}.speaker"),
                "text": text,
            }
            # These are source image descriptions consumed by the core loader.
            # Do not copy arbitrary fields, which can include answer annotations.
            for field in ("query", "blip_caption"):
                if field in turn:
                    if not isinstance(turn[field], str):
                        raise ValueError(f"{label}.{field} must be a string")
                    clean[field] = turn[field]
            clean_turns.append(clean)
        conversation[key] = clean_turns
    return {"sample_id": sample_id, "conversation": conversation}


def prepare(input_path: Path, *, options: dict | None = None) -> dict:
    """Return separate ``conversations``, gold ``questions`` and provenance metadata.

    The default keeps categories 1--4 (1,540 QA in the 10-conversation release).
    Category 5 is excluded: its ``adversarial_answer`` is not an ordinary gold
    answer. No source QA or oracle summaries enter memory construction.

    Supported options: ``limit_conversations`` and ``limit_questions`` are positive
    integers for deterministic source-order smoke subsets. A question limit never
    truncates its conversation history. ``categories`` can select a non-empty subset
    of [1, 2, 3, 4]. ``include_adversarial=True`` is explicitly unsupported. Unknown
    options are rejected to catch misspelled protocol settings.
    """
    opts = dict(options or {})
    unknown = set(opts) - {
        "limit_conversations", "limit_questions", "categories", "include_adversarial"
    }
    if unknown:
        raise ValueError(f"unsupported LoCoMo options: {sorted(unknown)}")
    if opts.get("include_adversarial", False) is not False:
        raise ValueError("LoCoMo category 5 requires a separate adversarial protocol")
    conversation_limit = _positive_limit(opts, "limit_conversations")
    question_limit = _positive_limit(opts, "limit_questions")
    selected_categories = opts.get("categories", [1, 2, 3, 4])
    if not isinstance(selected_categories, (list, tuple)) or not selected_categories:
        raise ValueError("categories must be a non-empty list drawn from [1, 2, 3, 4]")
    categories = {str(value) for value in selected_categories}
    if not categories <= _CATEGORIES:
        raise ValueError("categories must be drawn from [1, 2, 3, 4]")
    path = Path(input_path)
    raw = path.read_bytes()
    samples = json.loads(raw)
    if not isinstance(samples, list) or not samples:
        raise ValueError("LoCoMo input must be a non-empty JSON array")

    conversations = []
    questions = []
    source_category_counts: Counter = Counter()
    sample_ids: set[str] = set()
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {sample_index} must be an object")
        sample_id = _nonempty(sample.get("sample_id"), f"sample {sample_index}.sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        clean_conversation = _conversation(sample, sample_id)
        qa = sample.get("qa")
        if not isinstance(qa, list):
            raise ValueError(f"{sample_id}.qa must be an array")
        sample_questions = []
        for question_index, question in enumerate(qa):
            label = f"{sample_id}.qa[{question_index}]"
            if not isinstance(question, dict):
                raise ValueError(f"{label} must be an object")
            category = str(question.get("category"))
            if category not in _CATEGORIES | {"5"}:
                raise ValueError(f"{label}.category must be one of 1, 2, 3, 4, 5")
            source_category_counts[category] += 1
            question_text = _nonempty(question.get("question"), f"{label}.question")
            evidence = question.get("evidence", [])
            if not isinstance(evidence, list) or any(not isinstance(e, str) for e in evidence):
                raise ValueError(f"{label}.evidence must be an array of strings")
            if category == "5":
                continue
            answer = question.get("answer")
            if answer is None or isinstance(answer, (dict, list)):
                raise ValueError(f"{label}.answer must be a scalar")
            if category not in categories:
                continue
            sample_questions.append({
                "id": f"{sample_id}:q{question_index:04d}",
                "conversation_id": sample_id,
                "question": question_text,
                "answer": answer,
                "category": category,
                "evidence": list(evidence),
                "metadata": {
                    "source_sample_id": sample_id,
                    "source_question_index": question_index,
                },
            })
        if conversation_limit is not None and sample_index >= conversation_limit:
            continue
        remaining = None if question_limit is None else max(0, question_limit - len(questions))
        sample_questions = sample_questions[:remaining]
        if sample_questions:
            conversations.append(clean_conversation)
            questions.extend(sample_questions)

    return {
        "conversations": conversations,
        "questions": questions,
        "metadata": {
            "benchmark": "locomo",
            "adapter_schema_version": 1,
            "protocol": "locomo_categories_1_to_4" if categories == _CATEGORIES else "locomo_categories_subset",
            "official_source": SOURCE_URL,
            "source": {"name": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "filters": {
                "categories": sorted(categories),
                "excluded_categories": sorted((_CATEGORIES | {"5"}) - categories),
                "limit_conversations": conversation_limit,
                "limit_questions": question_limit,
                "history_truncated": False,
            },
            "source_counts": {"conversations": len(samples), "questions": sum(source_category_counts.values()), "by_category": dict(sorted(source_category_counts.items()))},
            "prepared_counts": {"conversations": len(conversations), "questions": len(questions), "by_category": dict(sorted(Counter(q["category"] for q in questions).items()))},
            "synthetic_smoke": all(sample.get("_fixture") == "synthetic_smoke" for sample in samples),
            "image_policy": "retain source query and blip_caption text; no image downloads",
            "annotation_policy": "QA, evidence and dataset-generated summaries are excluded from corpus",
        },
    }
