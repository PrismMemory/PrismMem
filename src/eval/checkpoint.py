"""Atomic, concurrency-safe checkpoints for resumable extraction."""

from __future__ import annotations

import threading
import time

import jsonio


class Checkpoint:
    """Track completed ``(conversation_id, unit)`` pairs in one JSON file.

    Call :meth:`mark` only after the corresponding output has been durably written.
    This makes a resumed run idempotently repeat at most the interrupted unit.
    """

    def __init__(self, path: str, flow: str) -> None:
        self.path = path
        self.flow = flow
        self._lock = threading.RLock()
        data = jsonio.read_json(path, default=None) or {}
        if not isinstance(data, dict) or (data and data.get("flow") != flow):
            raise ValueError(f"checkpoint flow mismatch: {path}")
        raw = data.get("completed", {})
        if not isinstance(raw, dict) or any(
            not isinstance(group, str) or not isinstance(units, list)
            or any(type(unit) is not int or unit < 0 for unit in units)
            for group, units in raw.items()
        ):
            raise ValueError(f"malformed completed units in checkpoint: {path}")
        self._completed: dict[str, set[int]] = {
            group: set(units) for group, units in raw.items()
        }
        self._output_hashes = data.get("output_sha256", {})
        if not isinstance(self._output_hashes, dict) or any(
            not isinstance(units, dict) or any(
                not isinstance(unit, str) or not isinstance(digest, str)
                for unit, digest in units.items()
            )
            for units in self._output_hashes.values()
        ):
            raise ValueError(f"malformed output hashes in checkpoint: {path}")

    def done(self, group: str, unit: int) -> bool:
        with self._lock:
            return unit in self._completed.get(group, set())

    def done_units(self, group: str) -> set[int]:
        with self._lock:
            return set(self._completed.get(group, set()))

    def output_sha256(self, group: str, unit: int) -> str | None:
        with self._lock:
            return self._output_hashes.get(group, {}).get(str(unit))

    def completed(self) -> dict[str, set[int]]:
        with self._lock:
            return {group: set(units) for group, units in self._completed.items()}

    def mark(self, group: str, unit: int, *, output_sha256: str | None = None) -> None:
        with self._lock:
            self._completed.setdefault(group, set()).add(unit)
            if output_sha256 is not None:
                self._output_hashes.setdefault(group, {})[str(unit)] = output_sha256
            self._save()

    def clear(self, group: str | None = None) -> None:
        with self._lock:
            if group is None:
                self._completed.clear()
                self._output_hashes.clear()
            else:
                self._completed.pop(group, None)
                self._output_hashes.pop(group, None)
            self._save()

    def _save(self) -> None:
        jsonio.atomic_write_json(
            self.path,
            {
                "flow": self.flow,
                "completed": {
                    group: sorted(units)
                    for group, units in self._completed.items()
                },
                "output_sha256": self._output_hashes,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
