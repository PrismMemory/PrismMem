"""Adapters convert public benchmark histories and keep evaluation labels separate."""

from importlib import import_module

BENCHMARKS = ("locomo", "longmemeval", "personamem", "beam")


def get_adapter(name: str):
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}; choose from {', '.join(BENCHMARKS)}")
    return import_module(f"benchmarks.adapters.{name}")

