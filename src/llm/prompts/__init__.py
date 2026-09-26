"""Load immutable release prompts using ``{{PLACEHOLDER}}`` substitution."""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def render(name: str, **variables: object) -> str:
    text = (_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text

