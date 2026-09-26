"""PersonaMem v1 CSV + shared-context adapter (32K by default).

Source: https://github.com/bowen-upenn/PersonaMem
The official history is ``context[:int(end_index_in_shared_context)]``.
Every effective prefix is a separate memory scope; future turns, questions,
answer choices and reference answers are never copied into the memory corpus.
Each question uses only its permitted visible history prefix.
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re


SOURCE_URL = "https://github.com/bowen-upenn/PersonaMem"
PROTOCOL = "personamem-v1-exclusive-prefix-oss-v1"
_SYNTHETIC_BASE = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=8)))
_LABEL = re.compile(r"^\s*\(([a-d])\)\s*", re.IGNORECASE)
_NAMES = (
    r"Current user persona:\s*Name:\s*([^\n\r]+)",
    r"Current user persona:\s*Meet\s+([^,\n\r]+)",
    r"Current user persona:\s*(?:Her|His|Their) name is\s+([^,\n\r]+)",
)


def _positive_limit(options: dict, key: str) -> int | None:
    value = options.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"PersonaMem {key} must be a positive integer")
    return value


def _fingerprint(path: Path) -> dict:
    raw = path.read_bytes()
    return {"name": path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load_contexts(path: Path) -> dict[str, list[dict]]:
    """Read official one-key-per-line JSONL, or equivalent JSON mapping."""
    if path.suffix.lower() == ".jsonl":
        payloads = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads = payload if isinstance(payload, list) else [payload]
    contexts: dict[str, list[dict]] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            raise ValueError(
                "PersonaMem contexts must contain mappings of context IDs to message lists"
            )
        for context_id, turns in payload.items():
            if not context_id or str(context_id) in contexts or not isinstance(turns, list):
                raise ValueError("PersonaMem has a duplicate context ID or invalid message list")
            if any(not isinstance(turn, dict) for turn in turns):
                raise ValueError(f"PersonaMem context {context_id!r} contains a non-object message")
            contexts[str(context_id)] = turns
    return contexts


def _corpus(context_id: str, turns: list[dict], cutoff: int) -> dict:
    # Do not inspect content beyond the visible prefix, including persona names.
    visible = turns[:cutoff]
    if not visible or visible[0].get("role") != "system":
        raise ValueError("PersonaMem visible history must include the initial system persona")
    initial_persona = str(visible[0].get("content", ""))
    persona_name = "User"
    for pattern in _NAMES:
        match = re.search(pattern, initial_persona, re.IGNORECASE)
        if match:
            persona_name = " ".join(match.group(1).split())
            break
    scope_hash = hashlib.sha256(json.dumps([context_id, cutoff]).encode()).hexdigest()[:20]
    sample_id = f"personamem-{scope_hash}-c{cutoff}"
    conversation: dict = {"speaker_a": persona_name, "speaker_b": "Assistant"}
    session_number = -1
    session_extraction_start = 0
    for raw_index, turn in enumerate(visible):
        role = str(turn.get("role", "")).lower()
        if role not in {"user", "assistant", "system"} or not isinstance(turn.get("content"), str):
            raise ValueError(f"PersonaMem invalid message at {context_id}:{raw_index}")
        if role == "system":
            session_number += 1
            session_extraction_start = (
                raw_index if raw_index == 0 or turn["content"] != initial_persona else raw_index + 1
            )
            # v1 repeats an identical persona at each session boundary. Preserve
            # changed system content if supplied, rather than silently deleting it.
            if raw_index and turn["content"] == initial_persona:
                continue
        key = f"session_{session_number}"
        conversation.setdefault(key, [])
        conversation[f"{key}_date_time"] = ""  # v1 supplies no real session dates.
        text = re.sub(
            r"^(?:User|Assistant)\s*:\s*", "", turn["content"], count=1, flags=re.IGNORECASE
        ).strip()
        conversation[key].append(
            {
                "dia_id": f"D{session_number}:{raw_index}",
                "speaker": "System Persona"
                if role == "system"
                else persona_name
                if role == "user"
                else "Assistant",
                "text": text,
                "source_turn_index": raw_index,
                "source_role": role,
                # Answer-only ordering metadata. Never assign this synthetic clock
                # to session_N_date_time or expose it as a real event timestamp.
                "source_session_number": session_number,
                "source_session_extraction_start": session_extraction_start,
                "synthetic_source_mention_timestamp": (
                    _SYNTHETIC_BASE
                    + timedelta(
                        days=session_number, seconds=30 * (raw_index - session_extraction_start)
                    )
                ).isoformat(),
            }
        )
    return {"sample_id": sample_id, "conversation": conversation}


def prepare(input_path: Path, *, options: dict | None = None) -> dict:
    """Load a v1 directory or questions CSV; optional ``contexts_path`` overrides.

    ``size`` selects the official filenames (default ``32k``). Limits select
    questions and effective prefix scopes in source order, without truncating
    any selected history. Raw stops obey
    Python slicing, including negative or oversize stops. The normalized stop
    is recorded separately so equivalent prefixes share exactly one scope.
    """
    options = options or {}
    unknown = set(options) - {"size", "contexts_path", "limit_questions", "limit_conversations"}
    if unknown:
        raise ValueError(f"Unknown PersonaMem prepare options: {sorted(unknown)}")
    question_limit = _positive_limit(options, "limit_questions")
    conversation_limit = _positive_limit(options, "limit_conversations")
    input_path = Path(input_path)
    size = str(options.get("size", "32k"))
    questions_path = input_path / f"questions_{size}.csv" if input_path.is_dir() else input_path
    contexts_path = (
        Path(options["contexts_path"])
        if options.get("contexts_path")
        else questions_path.with_name(
            questions_path.name.replace("questions_", "shared_contexts_")
        ).with_suffix(".jsonl")
    )
    contexts = _load_contexts(contexts_path)
    with questions_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "question_id",
            "question_type",
            "user_question_or_message",
            "correct_answer",
            "all_options",
            "shared_context_id",
            "end_index_in_shared_context",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError(
                f"PersonaMem CSV missing columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("PersonaMem questions CSV is empty")
    scopes: dict[tuple[str, int], dict] = {}
    questions = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        question_id = str(row["question_id"]).strip()
        if not question_id or question_id in seen_ids:
            raise ValueError("PersonaMem question IDs must be nonempty and unique")
        seen_ids.add(question_id)
        context_id = row["shared_context_id"].strip()
        if context_id not in contexts:
            raise ValueError(
                f"PersonaMem question {question_id} references unknown context {context_id!r}"
            )
        raw_cutoff = int(row["end_index_in_shared_context"])
        cutoff = slice(None, raw_cutoff).indices(len(contexts[context_id]))[1]
        scope_key = (context_id, cutoff)
        if scope_key not in scopes:
            scopes[scope_key] = _corpus(context_id, contexts[context_id], cutoff)
        try:
            labelled_choices = ast.literal_eval(row["all_options"])
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"PersonaMem question {question_id} has invalid all_options") from exc
        if (
            not isinstance(labelled_choices, list)
            or len(labelled_choices) != 4
            or any(not isinstance(x, str) for x in labelled_choices)
        ):
            raise ValueError("PersonaMem v1 requires four labelled choices")
        labels = [_LABEL.match(choice) for choice in labelled_choices]
        if [match.group(1).lower() if match else None for match in labels] != list("abcd"):
            raise ValueError("PersonaMem v1 choices must be labelled (a), (b), (c), (d) in order")
        gold = row["correct_answer"].strip().lower()
        if not re.fullmatch(r"\([a-d]\)", gold):
            raise ValueError(f"PersonaMem question {question_id} has invalid correct_answer")
        question = row["user_question_or_message"].strip()
        if not question:
            raise ValueError(f"PersonaMem question {question_id} has no question text")
        questions.append(
            {
                "id": question_id,
                "conversation_id": scopes[scope_key]["sample_id"],
                "question": question,
                "answer": gold,
                "category": row["question_type"],
                "retrieval_query": question + "\n\nOptions:\n" + "\n".join(labelled_choices),
                "choices": [_LABEL.sub("", choice, count=1).strip() for choice in labelled_choices],
                "correct_choice": ord(gold[1]) - ord("a"),
                "evidence": [],
                "metadata": {
                    **row,
                    "source_row_index": row_index,
                    "raw_end_index_in_shared_context": raw_cutoff,
                    "effective_end_index_in_shared_context": cutoff,
                    "cutoff_semantics": "shared_context[:end_index_in_shared_context] (exclusive Python slice)",
                },
            }
        )
    source_counts = {
        "conversations": len(scopes),
        "questions": len(questions),
        "by_category": dict(Counter(q["category"] for q in questions)),
    }
    selected_scopes = list(scopes.values())[:conversation_limit]
    selected_ids = {sample["sample_id"] for sample in selected_scopes}
    questions = [q for q in questions if q["conversation_id"] in selected_ids][:question_limit]
    used_ids = {q["conversation_id"] for q in questions}
    selected_scopes = [sample for sample in selected_scopes if sample["sample_id"] in used_ids]
    synthetic_smoke = all(r.get("dataset_kind") == "synthetic_smoke" for r in rows)
    return {
        "conversations": selected_scopes,
        "questions": questions,
        "metadata": {
            "benchmark": "personamem",
            "version": "v1",
            "size": size,
            "protocol": PROTOCOL,
            "source_url": SOURCE_URL,
            "dataset_kind": "synthetic_smoke" if synthetic_smoke else "benchmark",
            "synthetic_smoke": synthetic_smoke,
            "source_files": [_fingerprint(questions_path), _fingerprint(contexts_path)],
            "filters": {
                "limit_questions": question_limit,
                "limit_conversations": conversation_limit,
                "history_truncated": False,
            },
            "source_counts": source_counts,
            "prepared_counts": {
                "conversations": len(selected_scopes),
                "questions": len(questions),
                "by_category": dict(Counter(q["category"] for q in questions)),
            },
            "question_count": len(questions),
            "conversation_count": len(selected_scopes),
            "shared_context_count": len({q["metadata"]["shared_context_id"] for q in questions}),
            "category_counts": dict(Counter(q["category"] for q in questions)),
            "cutoff_semantics": "exclusive Python slice; one isolated memory scope per effective prefix",
            "system_persona_policy": "include initial persona once; omit identical repeated personas; preserve changed visible system messages",
            "session_dates": "real session dates unavailable; synthetic ordering metadata is kept separate",
            "answer_timeline": {
                "profile": "hypermem_synthetic_v1",
                "semantics": "source_mention_order_only; not real-world event time",
                "base_datetime": _SYNTHETIC_BASE.isoformat(),
                "session_step_days": 1,
                "turn_step_seconds": 30,
                "ordering": "answer-only stable ascending span start/end, then original retrieval rank",
                "extraction_uses_synthetic_time": False,
            },
            "retrieval_query_policy": "question plus all labelled choices; no reference answer or gold-only metadata",
            "answer_context_policy": "allowed history supplied through memory retrieval; official v1 has no separately mandated recent-turn window",
            "paper_result_parity": False,
        },
    }
