"""Load LoCoMo and render its original sessions without losing provenance."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Iterator

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "datasets", "locomo10.json")
_SESSION_RE = re.compile(r"^session_(\d+)$")
_DATE_TIME_FMT = "%I:%M %p on %d %B, %Y"


def normalize_date_time(value: str | None) -> str | None:
    """Normalize LoCoMo's natural-language session timestamp when possible."""
    if not value:
        return value
    try:
        return datetime.strptime(value.strip(), _DATE_TIME_FMT).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return value


def load_samples(path: str = _DEFAULT_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("LoCoMo dataset root must be a JSON array")
    return payload


def extraction_samples(samples: list[dict]) -> list[dict]:
    """Validate and isolate corpus fields before any extraction call.
    """
    isolated = []
    seen_samples: set[str] = set()
    allowed_turn_fields = {"dia_id", "speaker", "text", "query", "blip_caption", "source_role"}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each corpus sample must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen_samples:
            raise ValueError("corpus sample_id values must be unique non-empty strings")
        seen_samples.add(sample_id)
        raw = sample.get("conversation")
        if not isinstance(raw, dict):
            raise ValueError(f"{sample_id}: conversation must be an object")
        speakers = participants(sample)
        if not speakers or any(not isinstance(name, str) or not name.strip() for name in speakers):
            raise ValueError(f"{sample_id}: non-empty speaker names are required")
        conversation = {key: raw[key] for key in ("speaker_a", "speaker_b") if key in raw}
        chat_ids: set[str] = set()
        for key, value in raw.items():
            match = _SESSION_RE.fullmatch(key)
            if not match:
                continue
            number = int(match.group(1))
            if key != f"session_{number}":
                raise ValueError(f"{sample_id}: noncanonical session key {key}")
            if not isinstance(value, list):
                raise ValueError(f"{sample_id}/{key}: turns must be an array")
            turns = []
            for turn in value:
                if not isinstance(turn, dict):
                    raise ValueError(f"{sample_id}/{key}: each turn must be an object")
                chat_id = turn.get("dia_id")
                if not isinstance(chat_id, str) or not re.fullmatch(rf"D{number}:\d+", chat_id) or chat_id in chat_ids:
                    raise ValueError(f"{sample_id}/{key}: invalid or duplicate dia_id {chat_id!r}")
                chat_ids.add(chat_id)
                if not isinstance(turn.get("speaker"), str) or not turn["speaker"].strip():
                    raise ValueError(f"{sample_id}/{chat_id}: speaker is required")
                if not isinstance(turn.get("text"), str):
                    raise ValueError(f"{sample_id}/{chat_id}: text must be a string")
                turns.append({field: turn[field] for field in allowed_turn_fields if field in turn})
            date_key = f"session_{number}_date_time"
            date = raw.get(date_key)
            if date is not None and not isinstance(date, str):
                raise ValueError(f"{sample_id}/{key}: session date must be a string or null")
            conversation[key] = turns
            conversation[date_key] = date
        isolated.append({"sample_id": sample_id, "conversation": conversation})
    return isolated


def participants(sample: dict) -> list[str]:
    """Return the two speaker names used for strict output validation."""
    conversation = sample["conversation"]
    return [
        conversation[key]
        for key in ("speaker_a", "speaker_b")
        if conversation.get(key)
    ]


def iter_sessions(sample: dict) -> Iterator[dict]:
    """Yield non-empty source sessions in numeric order as ``D1``, ``D2``, ..."""
    conversation = sample["conversation"]
    numbers = sorted(
        int(match.group(1))
        for key in conversation
        if (match := _SESSION_RE.match(key)) and conversation.get(key)
    )
    for number in numbers:
        yield {
            "session_id": f"D{number}",
            "session_date": normalize_date_time(
                conversation.get(f"session_{number}_date_time")
            ),
            "turns": conversation[f"session_{number}"],
        }


def _image_note(turn: dict) -> str:
    parts = []
    if turn.get("query"):
        parts.append(f"subject: {turn['query']}")
    if turn.get("blip_caption"):
        parts.append(f"caption: {turn['blip_caption']}")
    return f" [image | {' | '.join(parts)}]" if parts else ""


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def _format_turn(turn: dict, base: datetime | None, index: int) -> str:
    timestamp = (
        f"{(base + timedelta(minutes=index)).strftime('%Y-%m-%d %H:%M:%S')} "
        if base
        else ""
    )
    return (
        f"{timestamp}[{turn.get('dia_id')}] {turn.get('speaker')}: "
        f"{turn.get('text') or ''}{_image_note(turn)}"
    )


def format_session_history(turns: list[dict], start_time: str | None = None) -> str:
    """Render every original turn in order for session-level semantic-scope extraction."""
    base = _parse_ts(start_time)
    return "\n".join(_format_turn(turn, base, index) for index, turn in enumerate(turns))


def format_dialogue_for_ids(
    turns: list[dict],
    chat_ids: list,
    start_time: str | None = None,
) -> str:
    """Render only selected source turns, preserving their full-session timestamps."""
    base = _parse_ts(start_time)
    wanted = set(chat_ids or [])
    return "\n".join(
        _format_turn(turn, base, index)
        for index, turn in enumerate(turns)
        if turn.get("dia_id") in wanted
    )
