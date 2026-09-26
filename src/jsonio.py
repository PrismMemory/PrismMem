"""Dependency-free, crash-safe JSON file helpers."""

from __future__ import annotations

import json
import os
from typing import Any


def atomic_write_json(path: str, obj: Any) -> None:
    """Write *obj* atomically so an interrupted run cannot leave truncated JSON."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON from *path*, returning *default* when the file is absent."""
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
