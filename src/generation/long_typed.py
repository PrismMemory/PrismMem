"""Raw-source facet extraction with a default LoCoMo profile."""

from __future__ import annotations

import time
from collections.abc import Iterable

from llm import get_response, prompts

from .util import new_id, now, parse_json

# The default LoCoMo profile selects durable-state, event, and knowledge facets.
_TYPE_PROMPTS = (
    ("memory_extraction_core", "CORE_MEMORY", "core"),
    ("memory_extraction_episodic", "EPISODIC_MEMORY", "episodic"),
    ("memory_extraction_knowledge", "KNOWLEDGE_MEMORY", "knowledge"),
)
TYPED_LONG_TYPES = ("core", "episodic", "knowledge")
SUPPORTED_LONG_TYPES = tuple(memory_type for _, _, memory_type in _TYPE_PROMPTS)
_STRICT_RESPONSE_ATTEMPTS = 3
_STRICT_RETRY_BACKOFF_SECONDS = 0.2


def resolve_typed_long_memory_types(
    memory_types: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Validate a requested subset and return it in canonical extraction order."""
    if memory_types is None:
        return TYPED_LONG_TYPES
    if isinstance(memory_types, str):
        raise TypeError("memory_types must be an iterable of type names, not a string")
    requested = tuple(str(value).strip().casefold() for value in memory_types)
    if not requested:
        raise ValueError("at least one typed-long memory type must be selected")
    unknown = sorted(set(requested) - set(SUPPORTED_LONG_TYPES))
    if unknown:
        raise ValueError(f"unsupported typed-long memory types: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("typed-long memory types must not contain duplicates")
    requested_set = set(requested)
    return tuple(item for item in SUPPORTED_LONG_TYPES if item in requested_set)


def normalize_tags(value: object, *, limit: int = 5) -> list[str]:
    if limit < 0:
        raise ValueError("tag limit cannot be negative")
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        tag = " ".join(item.split())
        key = tag.casefold()
        if not tag or key in seen:
            continue
        seen.add(key)
        tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def _records(data: object, output_key: str) -> list[dict]:
    if not isinstance(data, dict):
        return []
    items = data.get(output_key)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _strict_type_data(
    response: str | None,
    *,
    output_key: str,
    memory_type: str,
) -> dict:
    if not str(response or "").strip():
        raise RuntimeError(
            f"{memory_type} typed-long extraction model returned no content"
        )
    data = parse_json(response)
    if data is None:
        raise RuntimeError(
            f"{memory_type} typed-long extraction model returned invalid JSON"
        )
    if not isinstance(data, dict):
        raise RuntimeError(
            f"{memory_type} typed-long extraction model returned invalid schema: "
            "expected a JSON object"
        )
    if output_key not in data or not isinstance(data[output_key], list):
        raise RuntimeError(
            f"{memory_type} typed-long extraction model returned invalid schema: "
            f"expected a {output_key} array"
        )
    for index, item in enumerate(data[output_key]):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"{memory_type} typed-long extraction model returned invalid schema: "
                f"{output_key}[{index}] must be an object"
            )
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            raise RuntimeError(
                f"{memory_type} typed-long extraction model returned invalid schema: "
                f"{output_key}[{index}].content must be a non-empty string"
            )
    return data


def _strict_type_data_with_retries(
    prompt: str,
    *,
    output_key: str,
    memory_type: str,
) -> dict:
    last_error: RuntimeError | None = None
    for attempt in range(1, _STRICT_RESPONSE_ATTEMPTS + 1):
        try:
            return _strict_type_data(
                get_response(prompt),
                output_key=output_key,
                memory_type=memory_type,
            )
        except RuntimeError as exc:
            last_error = exc
            if attempt < _STRICT_RESPONSE_ATTEMPTS:
                time.sleep(_STRICT_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def _extract_type(
    mid: dict,
    *,
    prompt_name: str,
    output_key: str,
    memory_type: str,
    dialogue: str,
    require_valid_response: bool,
    participants: list[str] | None,
    participant_mode: str,
) -> list[dict]:
    # USER_INPUT is the only memory-bearing placeholder. No semantic-scope topic, summary,
    # tags, identifier or content is rendered into any facet-extraction prompt.
    # In literal compatibility mode, PARTICIPANTS remains unexpanded in the
    # facet prompts. Speaker names remain visible on every raw dialogue line.
    variables = {"USER_INPUT": dialogue}
    if prompt_name == "memory_extraction_core":
        variables["CORE_EXISTING_TOPICS_AND_SUBTOPICS"] = "(none)"
    if participant_mode == "explicit":
        variables["PARTICIPANTS"] = ", ".join(participants or [])
    prompt = prompts.render(prompt_name, **variables)
    if require_valid_response:
        data = _strict_type_data_with_retries(
            prompt,
            output_key=output_key,
            memory_type=memory_type,
        )
    else:
        data = parse_json(get_response(prompt))

    records: list[dict] = []
    for raw in _records(data, output_key):
        record = dict(raw)
        if memory_type == "core":
            record["tags"] = normalize_tags(record.get("tags"))
        record["id"] = new_id()
        record["mid_id"] = mid["id"]
        record["user_id"] = mid.get("user_id")
        record["type"] = memory_type
        record["created_at"] = now()
        records.append(record)
    return records


def extract_typed_long_memories(
    mid: dict,
    dialogue: str | None = None,
    *,
    require_valid_response: bool = False,
    memory_types: Iterable[str] | None = None,
    participants: list[str] | None = None,
    participant_mode: str = "legacy_literal",
) -> dict[str, list[dict]]:
    """Extract facets from raw turns only, separately for each selected type.

    """
    if participant_mode not in {"legacy_literal", "explicit"}:
        raise ValueError("participant_mode must be legacy_literal or explicit")
    if participant_mode == "explicit" and (
        not isinstance(participants, list) or not participants
        or any(not isinstance(item, str) or not item.strip() for item in participants)
    ):
        raise ValueError("explicit participant_mode requires non-empty participants")
    user_input = dialogue or "(not provided)"
    selected_types = set(resolve_typed_long_memory_types(memory_types))
    result = {memory_type: [] for memory_type in TYPED_LONG_TYPES}
    result.update({memory_type: [] for memory_type in selected_types})
    for prompt_name, output_key, memory_type in _TYPE_PROMPTS:
        if memory_type not in selected_types:
            continue
        result[memory_type] = _extract_type(
            mid,
            prompt_name=prompt_name,
            output_key=output_key,
            memory_type=memory_type,
            dialogue=user_input,
            require_valid_response=require_valid_response,
            participants=participants,
            participant_mode=participant_mode,
        )
    return result


__all__ = [
    "TYPED_LONG_TYPES",
    "SUPPORTED_LONG_TYPES",
    "extract_typed_long_memories",
    "normalize_tags",
    "resolve_typed_long_memory_types",
]
