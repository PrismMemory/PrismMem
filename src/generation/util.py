"""Small helpers shared by the extraction modules."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json(text: str | None) -> Any | None:
    """Parse JSON from a model response, tolerating fences and surrounding prose."""
    if not text:
        return None
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
    candidate = re.sub(r"```$", "", candidate).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None
