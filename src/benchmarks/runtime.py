"""Conversation-scoped embedding execution and an explicitly synthetic offline backend.

Importing this module never loads credentials or model libraries. The live backend
shares extraction and representation-isolated retrieval with the LoCoMo entry points.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import logging
import math
import os
from pathlib import Path
import re

from benchmarks.artifacts import read_json, write_json


SOURCE_ROOT = Path(__file__).resolve().parents[1]
_MODEL_ENV = {
    "extraction": "PRISM_EXTRACTION_MODEL",
    "embedding": "PRISM_MODEL",
    "rerank": "PRISM_RERANK_MODEL",
    "answer": "PRISM_ANSWER_MODEL",
    "judge": "PRISM_JUDGE_MODEL",
}


class BenchmarkRuntimeError(RuntimeError):
    """An execution error safe to include in shared benchmark logs."""


def _conversation_id(conversation: dict) -> str:
    value = conversation.get("sample_id") or conversation.get("conversation_id")
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkRuntimeError("Conversation requires a stable identifier.")
    return value


def _memory_dir(run_dir: Path, conversation_id: str) -> Path:
    return run_dir / "memory" / hashlib.sha256(conversation_id.encode()).hexdigest()[:20]


@contextmanager
def _quiet_service_logs():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def _load_live_core(profile: dict) -> dict:
    dotenv = importlib.import_module("dotenv")
    project = SOURCE_ROOT.parent if SOURCE_ROOT.name == "src" else Path.cwd()
    dotenv.load_dotenv(project / ".env", override=False)
    from benchmarks.run import ENVIRONMENT_KEYS

    # Bind every semantic setting to the saved profile, not a previous run or .env.
    for name in ENVIRONMENT_KEYS | set(_MODEL_ENV.values()):
        os.environ.pop(name, None)
    for name, value in profile.get("environment", {}).items():
        if name not in ENVIRONMENT_KEYS:
            raise BenchmarkRuntimeError("Unsupported benchmark environment setting.")
        os.environ[name] = str(value)
    os.environ["PRISM_LLM_MAX_TOKENS"] = "20000"
    for role, name in _MODEL_ENV.items():
        value = profile["models"].get(role)
        if value is not None:
            os.environ[name] = value
    config = importlib.import_module("config")
    importlib.reload(config)
    providers = importlib.import_module("llm.providers")
    importlib.reload(providers)
    client = importlib.import_module("llm.client")
    client._clients.clear()
    encoder = importlib.import_module("embedding.encoder")
    encoder._client = None
    retrieval = importlib.import_module("retrieval.rrf90")
    retrieval.clear_pool_caches()
    return {
        "config": config,
        "client": client,
        "extract": importlib.import_module("eval.extract_locomo"),
        "backfill": importlib.import_module("eval.backfill_embeddings"),
        "evaluate": importlib.import_module("eval.evaluate"),
        "retrieval": retrieval,
    }


def _valid_vector(vector: object) -> bool:
    return (
        isinstance(vector, list) and bool(vector)
        and all(type(value) in (int, float) and math.isfinite(value) for value in vector)
        and any(value != 0 for value in vector)
    )


def _attach_mention_span(destination: dict, record: dict, times: dict, parent: dict | None) -> None:
    if parent is None:
        ids = record.get("chat_ids") or []
    else:
        evidence = record.get("sourceEvidence", record.get("source_evidence", []))
        if not isinstance(evidence, list):
            raise BenchmarkRuntimeError("Persona source evidence must be an array.")
        ids = [item.get("chatId") or item.get("chat_id") for item in evidence if isinstance(item, dict)]
        ids = [value for value in ids if value] or record.get("source_chat_ids") or []
    if not isinstance(ids, list) or not ids:
        raise BenchmarkRuntimeError("Persona memory has no source turn IDs.")
    resolved = []
    for value in ids:
        key = str(value)
        if key not in times:
            raise BenchmarkRuntimeError("Persona source evidence is outside the visible prefix.")
        resolved.append(times[key])
    destination["synthetic_source_mention_start"] = min(resolved)
    destination["synthetic_source_mention_end"] = max(resolved)


class LiveBackend:
    synthetic = False

    def __init__(self, profile: dict, run_dir: Path):
        self.profile = deepcopy(profile)
        self.run_dir = Path(run_dir).resolve()
        self._stores: dict[str, dict] = {}
        self._current_id: str | None = None
        self._source_corpus: dict[str, dict] | None = None
        try:
            with _quiet_service_logs():
                self._core = _load_live_core(self.profile)
        except Exception:
            raise BenchmarkRuntimeError("Cannot initialize live execution; check dependencies and configuration.") from None

    def _store(self, cid: str) -> dict:
        if cid in self._stores:
            return self._stores[cid]
        store = self._core["evaluate"].load_memory_store(_memory_dir(self.run_dir, cid))
        mids = store.get("mids", [])
        if not mids:
            raise BenchmarkRuntimeError("Memory construction produced no Mid memories.")
        all_rows = [row for rows in store.values() for row in rows]
        if any(str(row.get("conversation_id")) != cid for row in all_rows):
            raise BenchmarkRuntimeError("Memory artifacts cross conversation boundaries.")
        ids = [str(row.get("id") or "") for row in all_rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise BenchmarkRuntimeError("Memory identifiers must be non-empty and unique.")
        if any(not _valid_vector(row.get("embedding")) for row in all_rows):
            raise BenchmarkRuntimeError("Memory vectors must be complete, finite and nonzero.")
        if len({len(row["embedding"]) for row in all_rows}) != 1:
            raise BenchmarkRuntimeError("Memory vector dimensions differ.")
        self._stores[cid] = store
        return store

    def build(self, conversation: dict, corpus_path: Path) -> dict:
        cid = _conversation_id(conversation)
        memory = _memory_dir(self.run_dir, cid)
        settings = dict(self.profile.get("build", {}))
        try:
            with _quiet_service_logs():
                self._core["extract"].run_pipeline(
                    dataset_path=str(Path(corpus_path).resolve()), output_dir=str(memory),
                    phase="all", sample_id=cid, expected_model=self.profile["models"]["extraction"],
                    **settings,
                )
                self._core["backfill"].build_long_embeddings(memory_dir=str(memory))
                self._stores.pop(cid, None)
                store = self._store(cid)
        except Exception:
            raise BenchmarkRuntimeError("Memory construction failed; check extraction and embedding configuration or prepared input.") from None
        self._current_id = cid
        return {"counts": {key: len(rows) for key, rows in store.items()}, "synthetic": False}

    def _source(self, cid: str) -> dict:
        if self._source_corpus is None:
            rows = read_json(self.run_dir / "prepared" / "corpus.json")
            self._source_corpus = {_conversation_id(row): row for row in rows}
        try:
            return self._source_corpus[cid]["conversation"]
        except KeyError:
            raise BenchmarkRuntimeError("Prepared source conversation is missing.") from None

    def _project(self, memories: list[dict], cid: str, store: dict) -> list[dict]:
        benchmark = self.profile["benchmark"]
        raw = {str(row["id"]): row for rows in store.values() for row in rows}
        mid_ids = {str(row["id"]) for row in store["mids"]}
        times = {}
        if benchmark == "personamem":
            for key, turns in self._source(cid).items():
                if re.fullmatch(r"session_\d+", key):
                    for turn in turns:
                        stamp = turn.get("synthetic_source_mention_timestamp")
                        if not isinstance(stamp, str) or not stamp:
                            raise BenchmarkRuntimeError("Persona source ordering metadata is missing.")
                        times[str(turn["dia_id"])] = stamp
        projected = deepcopy(memories)
        for memory in projected:
            original = raw.get(str(memory.get("id")))
            if original is None:
                raise BenchmarkRuntimeError("Retrieved memory is absent from its scoped source store.")
            is_mid = str(memory["id"]) in mid_ids
            memory["memory_type"] = "mid" if is_mid else original.get("type")
            memory["source_record"] = {
                key: deepcopy(value) for key, value in original.items()
                if key not in {"embedding", "vector"}
            }
            if benchmark == "personamem":
                parent = None if is_mid else raw.get(str(original.get("mid_id")))
                if not is_mid and parent is None:
                    raise BenchmarkRuntimeError("Persona Long memory is missing its owning Mid.")
                _attach_mention_span(memory, original, times, parent)
        return projected

    def retrieve(self, question: dict) -> dict:
        cid = str(question.get("conversation_id") or "")
        query = question.get("retrieval_query") or question.get("question")
        if not cid or not isinstance(query, str) or not query.strip():
            raise BenchmarkRuntimeError("Retrieval requires public question text and a conversation ID.")
        try:
            with _quiet_service_logs():
                store = self._store(cid)
                result = self._core["retrieval"].search_rrf90(query, store)
                memories = self._project(result.memories, cid, store)
        except BenchmarkRuntimeError:
            raise
        except Exception:
            raise BenchmarkRuntimeError("Retrieval failed; check memory vectors, embedding, rerank and lexical-model setup.") from None
        self._current_id = cid
        return {"memories": memories, "trace": result.trace}

    def complete(self, role: str, messages: list) -> dict:
        if role not in {"answer", "judge"}:
            raise BenchmarkRuntimeError("Only answer and judge completion roles are supported.")
        model = self.profile["models"].get(role)
        if not isinstance(model, str) or not model.strip():
            raise BenchmarkRuntimeError(f"No {role} model is configured.")
        options = self.profile.get(role, {})
        try:
            with _quiet_service_logs():
                response, tokens = self._core["client"].complete_with_usage(
                    messages, model=model, role=role,
                    max_tokens=options.get("max_tokens", 20000),
                    temperature=options.get("temperature"),
                )
            if not isinstance(response, str) or not response.strip():
                raise ValueError("Empty model completion")
        except Exception:
            raise BenchmarkRuntimeError(f"The {role} request failed; check its model, endpoint and credentials.") from None
        return {"response": response, "usage": {"prompt_tokens": tokens}}


class SmokeBackend:
    """Deterministic source-text plumbing check; never a model or accuracy measurement."""

    synthetic = True

    def __init__(self, profile: dict, run_dir: Path):
        self.profile = deepcopy(profile)
        self.run_dir = Path(run_dir)

    def build(self, conversation: dict, corpus_path: Path) -> dict:
        cid = _conversation_id(conversation)
        memories = []
        for key, turns in conversation["conversation"].items():
            if not re.fullmatch(r"session_\d+", key):
                continue
            text = "\n".join(f"{row['speaker']}: {row['text']}" for row in turns)
            if not text.strip():
                continue
            record = {
                "id": "smoke-" + hashlib.sha256((cid + key).encode()).hexdigest()[:20],
                "conversation_id": cid, "memory_type": "mid", "memory_pool": "mid",
                "topic_subject": key, "summary": text, "content": text,
                "session_id": "D" + key.split("_")[1],
                "session_date": conversation["conversation"].get(key + "_date_time", ""),
                "chat_ids": [row["dia_id"] for row in turns],
                "dialogue_phase": "synthetic_smoke", "user_attitude": "synthetic fixture",
                "synthetic": True,
            }
            times = [row["synthetic_source_mention_timestamp"] for row in turns
                     if row.get("synthetic_source_mention_timestamp")]
            if times:
                record.update(synthetic_source_mention_start=min(times), synthetic_source_mention_end=max(times))
            memories.append(record)
        if not memories:
            raise BenchmarkRuntimeError("Synthetic input has no source text.")
        write_json(_memory_dir(self.run_dir, cid) / "mid_memories.json", memories)
        return {"mid_count": len(memories), "synthetic": True, "backend": "SMOKE_ONLY"}

    def retrieve(self, question: dict) -> dict:
        cid = str(question.get("conversation_id") or "")
        rows = read_json(_memory_dir(self.run_dir, cid) / "mid_memories.json")
        if not rows or any(row.get("conversation_id") != cid or row.get("synthetic") is not True for row in rows):
            raise BenchmarkRuntimeError("Invalid synthetic memory artifacts.")
        query = question.get("retrieval_query") or question["question"]
        terms = set(re.findall(r"\w+", query.lower()))
        ranked = sorted(rows, key=lambda row: -len(terms & set(re.findall(r"\w+", row["summary"].lower()))))
        selected = deepcopy(ranked[:5])
        for row in selected:
            row["source_record"] = deepcopy(row)
        return {"memories": selected, "trace": {"backend": "SMOKE_ONLY", "synthetic": True, "query": query}}

    def complete(self, role: str, messages: list) -> dict:
        return {"response": "SMOKE_ONLY: no model was called.", "usage": {}}
