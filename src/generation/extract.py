"""Session-level semantic-scope extraction with optional strict validation."""

from __future__ import annotations

import time

from llm import get_response, prompts

from .util import new_id, now, parse_json

_DEFAULT_CONFIDENCE = 1.0
_STRICT_RESPONSE_ATTEMPTS = 3
_STRICT_RETRY_BACKOFF_SECONDS = 0.2


def _strict_mid_groups(
    response: str | None, valid_users: set[str], *, require_beam_state: bool = False,
) -> list[dict]:
    if not str(response or "").strip():
        raise RuntimeError("mid extraction model returned no content")
    data = parse_json(response)
    if data is None:
        raise RuntimeError("mid extraction model returned invalid JSON")
    if not isinstance(data, dict):
        raise RuntimeError(
            "mid extraction model returned invalid schema: expected a JSON object"
        )
    if "chat_group" not in data or not isinstance(data["chat_group"], list):
        raise RuntimeError(
            "mid extraction model returned invalid schema: expected a chat_group array"
        )
    groups = data["chat_group"]
    for index, group in enumerate(groups):
        prefix = f"mid extraction model returned invalid schema: chat_group[{index}]"
        if not isinstance(group, dict):
            raise RuntimeError(f"{prefix} must be an object")
        user_id = group.get("user_id")
        if not isinstance(user_id, str) or user_id not in valid_users:
            raise RuntimeError(f"{prefix}.user_id must name a participant")
        if not isinstance(group.get("chat_ids"), list):
            raise RuntimeError(f"{prefix}.chat_ids must be an array")
        for field in ("topic_subject", "summary"):
            value = group.get(field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"{prefix}.{field} must be a non-empty string")
        if require_beam_state:
            if not isinstance(group.get("dialogue_phase"), str) or not group["dialogue_phase"].strip():
                raise RuntimeError(f"{prefix}.dialogue_phase must be a non-empty string")
            attitude = group.get("user_attitude")
            if not isinstance(attitude, dict) or any(
                not isinstance(attitude.get(key), str) or not attitude[key].strip()
                for key in ("satisfaction_level", "reasoning")
            ):
                raise RuntimeError(f"{prefix}.user_attitude requires satisfaction_level and reasoning")
    return groups


def _strict_mid_groups_with_retries(
    prompt: str,
    valid_users: set[str],
    *,
    require_beam_state: bool = False,
) -> list[dict]:
    last_error: RuntimeError | None = None
    for attempt in range(1, _STRICT_RESPONSE_ATTEMPTS + 1):
        try:
            return _strict_mid_groups(
                get_response(prompt), valid_users, require_beam_state=require_beam_state,
            )
        except RuntimeError as exc:
            last_error = exc
            if attempt < _STRICT_RESPONSE_ATTEMPTS:
                time.sleep(_STRICT_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def extract_mid_memories(
    *,
    session_history: str,
    conversation_id: str,
    session_id: str,
    session_date: str,
    participants: list[str],
    current_time: str | None = None,
    require_valid_response: bool = False,
    prompt_name: str = "mid_extraction",
) -> list[dict]:
    """Extract semantic scopes from one complete original session.

    """
    valid_users = set(participants)
    if prompt_name not in {"mid_extraction", "mid_extraction_beam"}:
        raise ValueError(f"unsupported Mid prompt: {prompt_name}")
    prompt = prompts.render(
        prompt_name,
        CURRENT_TIME=current_time or session_date,
        PARTICIPANTS=", ".join(participants),
        SESSION_HISTORY=session_history,
    )
    if require_valid_response:
        groups = _strict_mid_groups_with_retries(
            prompt, valid_users, require_beam_state=prompt_name == "mid_extraction_beam",
        )
    else:
        data = parse_json(get_response(prompt))
        if not isinstance(data, dict):
            return []
        groups = data.get("chat_group") or []

    memories: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        user_id = group.get("user_id")
        if user_id not in valid_users:
            continue
        memories.append(
            {
                "id": new_id(),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "session_id": session_id,
                "session_date": session_date,
                "chat_ids": group.get("chat_ids") or [],
                "topic_subject": group.get("topic_subject"),
                "summary": group.get("summary"),
                "tags": group.get("tags") or [],
                "confidence": _DEFAULT_CONFIDENCE,
                "created_at": now(),
            }
        )
        # Keep the complete attitude object, including additional model fields.
        # The LoCoMo semantic-scope prompt does not request these fields.
        if prompt_name == "mid_extraction_beam":
            for field in ("dialogue_phase", "user_attitude"):
                if field in group:
                    memories[-1][field] = group[field]
    return memories
