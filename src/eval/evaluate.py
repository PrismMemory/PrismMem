"""Run LoCoMo with three-pool retrieval and the package's current prompts."""

from __future__ import annotations

import argparse
import hashlib
import logging
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
for _path in (_SRC, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import config
import jsonio
import loader
from llm import complete_with_usage, prompts
from retrieval import search_rrf90
from retrieval.projection import answer_text, long_retrieval_text, text_sha256

logger = logging.getLogger("prism.evaluate")
ADVERSARIAL_CATEGORY = 5
_MEMORY_FILES = {
    "mids": "mid_memories.json",
    "long_core": "long_core.json",
    "long_episodic": "long_episodic.json",
    "long_knowledge": "long_knowledge.json",
}


def _read_array(path: Path, *, required: bool = True) -> list[dict]:
    payload = jsonio.read_json(str(path), default=None)
    if payload is None and not required:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [row for row in payload if isinstance(row, dict)]


def load_memory_store(memory_dir: str | Path) -> dict[str, list[dict]]:
    """Load the release store and strictly attach facet-vector sidecars."""
    root = Path(memory_dir).resolve()
    store = {
        key: _read_array(root / filename)
        for key, filename in _MEMORY_FILES.items()
    }
    sidecar = _read_array(root / "long_embeddings.json")
    vectors = {
        str(row.get("id") or ""): row
        for row in sidecar
        if str(row.get("id") or "").strip()
    }
    if len(vectors) != len(sidecar):
        raise ValueError("long_embeddings.json has duplicate or empty ids")

    for table in ("long_core", "long_episodic", "long_knowledge"):
        attached = []
        for raw in store[table]:
            record = dict(raw)
            record_id = str(record.get("id") or "").strip()
            side = vectors.get(record_id)
            if not record_id or side is None:
                raise ValueError(f"{table} memory {record_id or '(missing id)'} has no vector")
            expected_hash = text_sha256(long_retrieval_text(record))
            if side.get("embedding_model") != config.PRISM_MODEL:
                raise ValueError(f"{record_id}: unexpected embedding model")
            if side.get("text_sha256") != expected_hash:
                raise ValueError(f"{record_id}: stale embedding text hash")
            vector = side.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"{record_id}: invalid embedding vector")
            record["embedding"] = vector
            record["embedding_model"] = side["embedding_model"]
            record["embedding_text_sha256"] = side["text_sha256"]
            attached.append(record)
        store[table] = attached
    return store


def conversation_scope(store: dict[str, list[dict]], conversation_id: str) -> dict:
    """Merge both participants while preserving the conversation boundary."""
    return {
        key: [
            row
            for row in rows
            if str(row.get("conversation_id") or "") == conversation_id
        ]
        for key, rows in store.items()
    }


def _question_date(sample: dict) -> str:
    explicit = str(sample.get("question_date") or "").strip()
    if explicit:
        return explicit
    dates = [
        str(session.get("session_date") or "").strip()
        for session in loader.iter_sessions(sample)
    ]
    return next((value for value in reversed(dates) if value), "")


def _format_memories(memories: list[dict]) -> str:
    """LoCoMo answer projection; facet source quotes are appended only after retrieval."""
    if not memories:
        return "(no memories retrieved)"
    return "\n".join(f"- {answer_text(memory)}" for memory in memories)


def _without_vectors(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"embedding", "anonymous_embedding"}
    }


def evaluate_question(
    sample: dict,
    qa_index: int,
    qa: dict,
    scope: dict,
    *,
    retrieval_only: bool,
    cached_record: dict | None = None,
) -> dict:
    """Retrieve (or reuse a cached retrieval) then optionally answer.
    """
    started = time.time()
    question = str(qa.get("question") or "")
    if cached_record is not None:
        memories = list(cached_record.get("memories") or [])
        retrieval_trace = cached_record.get("retrieval_trace")
        retrieval_time = float(cached_record.get("retrieval_time") or 0.0)
    else:
        retrieval_started = time.time()
        result = search_rrf90(question, scope)
        retrieval_time = time.time() - retrieval_started
        memories = result.memories
        retrieval_trace = result.trace
    response = ""
    prompt_tokens = 0
    response_time = 0.0
    if not retrieval_only:
        rendered = prompts.render(
            "qa_answer",
            MEMORIES=_format_memories(memories),
            CURRENT_DATE=_question_date(sample),
            QUESTION=question,
        )
        response_started = time.time()
        response, prompt_tokens = complete_with_usage(
            [{"role": "user", "content": rendered}],
            model=config.QA_ANSWER_MODEL,
            role="answer",
            temperature=None,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        response_time = time.time() - response_started
        response = str(response or "").strip()
        if not response:
            raise RuntimeError("answer model returned empty content")
    return {
        "conversation_id": str(sample["sample_id"]),
        "qa_index": qa_index,
        "question": question,
        "question_date": _question_date(sample),
        "answer": qa.get("answer", ""),
        "adversarial_answer": qa.get("adversarial_answer", ""),
        "category": qa.get("category"),
        "evidence": qa.get("evidence", []),
        "memories": [_without_vectors(memory) for memory in memories],
        "retrieval_trace": retrieval_trace,
        "retrieval_mode": "prism_bm25_rrf90_typed_rerank",
        "prompt_memory_projection": "typed_content_plus_source_evidence",
        "retrieval_time": retrieval_time,
        "response": response,
        "response_time": response_time,
        "prompt_tokens": prompt_tokens,
        "retrieval_only": retrieval_only,
        "retrieval_reused": cached_record is not None,
        "total_time": time.time() - started,
    }


def _category(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


RETRIEVAL_DEFAULTS = {
    "RRF_CANDIDATE_TOP_N": 70,
    "MID_RERANK_TOP_N": 10,
    "CORE_RERANK_TOP_N": 60,
    "OTHER_RERANK_TOP_N": 50,
    "LONG_RERANK_MIN_SCORE": 0.5,
    "RRF_K": 60,
    "BM25_K1": 1.5,
    "BM25_B": 0.75,
}


def _retrieval_settings() -> dict:
    """Report active retrieval settings and overrides of their defaults."""
    active = {
        name: getattr(config, name) for name in RETRIEVAL_DEFAULTS
    }
    settings = {
        "candidate_top_n_per_pool": active["RRF_CANDIDATE_TOP_N"],
        "mid_top_n": active["MID_RERANK_TOP_N"],
        "core_top_n": active["CORE_RERANK_TOP_N"],
        "other_top_n": active["OTHER_RERANK_TOP_N"],
        "long_min_score": active["LONG_RERANK_MIN_SCORE"],
        "rrf_k": active["RRF_K"],
        "bm25_k1": active["BM25_K1"],
        "bm25_b": active["BM25_B"],
        "bm25_lemmatization": config.BM25_LEMMATIZATION_ENABLED,
        "uses_default_retrieval_settings": active == RETRIEVAL_DEFAULTS,
        "configuration_overrides": {
            name: {"default": default, "active": active[name]}
            for name, default in RETRIEVAL_DEFAULTS.items()
            if active[name] != default
        },
        "graph": False,
        "sufficiency": False,
    }
    return settings


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_fingerprints(*directories: str) -> dict:
    paths = {_HERE / "evaluate.py", _SRC / "config.py", _SRC / "jsonio.py", _HERE / "loader.py"}
    for directory in directories:
        paths.update((_SRC / directory).rglob("*.py"))
    return {str(path.relative_to(_SRC)): _file_hash(path) for path in sorted(paths)}


def _endpoint_fingerprint(*names: str) -> str:
    # Deployment changes invalidate results; credentials themselves can rotate.
    return _digest(next((getattr(config, name, None) for name in names
                         if getattr(config, name, None)), None))


def _answer_settings() -> dict:
    return {"model": config.QA_ANSWER_MODEL, "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": None, "system_message": None, "response_policy": "full_response",
            "enable_thinking": getattr(config, "ANSWER_ENABLE_THINKING", getattr(config, "QA_ENABLE_THINKING", False)),
            "endpoint_sha256": _endpoint_fingerprint("ANSWER_BASE_URL", "QA_BASE_URL"),
            "prompt_sha256": _file_hash(_SRC / "llm/prompts/qa_answer.txt"),
            "implementation_sha256": _code_fingerprints("llm", "retrieval")}


def _manifest(dataset: Path, memory_dir: Path, *, retrieval_only: bool) -> dict:
    retrieval = {"dataset_sha256": _file_hash(dataset),
                 "memory_sha256": {name: _file_hash(memory_dir / name)
                                   for name in sorted(set(_MEMORY_FILES.values()) | {"long_embeddings.json"})},
                 "models": {"embedding": config.PRISM_MODEL, "rerank": config.RERANK_MODEL},
                 "settings": _retrieval_settings(),
                 "bm25_lemmatization_model": getattr(config, "BM25_LEMMATIZATION_MODEL", None),
                 "rerank_batch_size": getattr(config, "RERANK_BATCH_SIZE", None),
                 "endpoints_sha256": {"embedding": _endpoint_fingerprint("PRISM_BASE_URL"),
                                      "rerank": _endpoint_fingerprint("RERANK_BASE_URL")},
                 "implementation_sha256": _code_fingerprints("retrieval", "embedding", "llm")}
    return {"schema_version": 2, "release": "prismmem",
            "retrieval_contract": retrieval,
            "answer_contract": None if retrieval_only else _answer_settings()}


def _record_digest(record: dict) -> str:
    return _digest({key: value for key, value in record.items() if key != "artifact_sha256"})


def _seal_record(record: dict, manifest: dict) -> dict:
    record = dict(record)
    record["retrieval_contract_sha256"] = _digest(manifest["retrieval_contract"])
    record["answer_contract_sha256"] = (_digest(manifest["answer_contract"])
                                         if str(record.get("response") or "").strip() else None)
    record["artifact_sha256"] = _record_digest(record)
    return record


def _answer_paths(output: Path) -> list[Path]:
    return sorted(output.glob("conv-*.json"))


def _score_paths(output: Path) -> list[Path]:
    return [output / name for name in ("scores.json", "progress_score.json", "score_manifest.json")]


def _validate_resume(output: Path, expected: dict, all_samples: list[dict], *, restart: bool,
                     sample_id: str | None) -> dict:
    """Read and verify everything before changing a manifest or output artifact."""
    existing = jsonio.read_json(str(output / "run_manifest.json"), default=None)
    paths = _answer_paths(output)
    if restart and sample_id is None:
        return expected
    if existing is None:
        if paths or any(path.exists() for path in _score_paths(output)):
            raise ValueError("Unlabelled prior evaluation output; use a new directory or full --restart")
        return expected
    if not isinstance(existing, dict) or existing.get("schema_version") != 2:
        raise ValueError("Unsupported prior evaluation manifest; use a new directory or full --restart")
    if existing.get("retrieval_contract") != expected["retrieval_contract"]:
        raise ValueError("Dataset, memory, retrieval model/settings or code changed; use a new directory or full --restart")
    prior_answer, next_answer = existing.get("answer_contract"), expected["answer_contract"]
    if prior_answer is not None and next_answer is not None and prior_answer != next_answer:
        raise ValueError("Answer model, prompt, settings or code changed; use a new directory or full --restart")
    # A retrieval-only inspection must never erase an established answer contract.
    merged = {**expected, "answer_contract": prior_answer if next_answer is None else next_answer}
    samples = {str(row["sample_id"]): row for row in all_samples}
    for path in paths:
        cid = path.stem
        if cid not in samples:
            raise ValueError("Prior output contains an unknown conversation; use full --restart")
        seen = set()
        rows = jsonio.read_json(str(path), default=None)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Prior evaluation output is not an array of records")
        for row in rows:
            index = row.get("qa_index")
            qas = samples[cid].get("qa") or []
            if type(index) is not int or index in seen or not 0 <= index < len(qas):
                raise ValueError("Prior output contains invalid or duplicate QA indices")
            seen.add(index)
            if row.get("conversation_id") != cid or row.get("question") != str(qas[index].get("question") or ""):
                raise ValueError("Prior QA identity does not match the dataset")
            if row.get("retrieval_contract_sha256") != _digest(expected["retrieval_contract"]):
                raise ValueError("Prior retrieval record has a different contract")
            if row.get("artifact_sha256") != _record_digest(row):
                raise ValueError("Prior evaluation record content changed; use full --restart")
            has_answer = bool(str(row.get("response") or "").strip())
            if has_answer and (prior_answer is None or row.get("answer_contract_sha256") != _digest(prior_answer)):
                raise ValueError("Prior answer record has a different contract")
            if not has_answer and (row.get("retrieval_only") is not True or not isinstance(row.get("memories"), list)):
                raise ValueError("Prior incomplete record is not a reusable retrieval-only result")
    return merged


def run_evaluation(
    *,
    dataset_path: str,
    memory_dir: str,
    output_dir: str,
    sample_id: str | None = None,
    limit: int | None = None,
    workers: int = config.EVAL_QA_WORKERS,
    restart: bool = False,
    retrieval_only: bool = False,
) -> dict[str, int]:
    if workers < 1:
        raise ValueError("workers must be positive")
    dataset = Path(dataset_path).resolve()
    memories = Path(memory_dir).resolve()
    output = Path(output_dir).resolve()
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be positive")
    store = load_memory_store(memories)
    all_samples = loader.load_samples(str(dataset))
    ids = [str(row["sample_id"]) for row in all_samples]
    if len(set(ids)) != len(ids) or any(not ident.startswith("conv-") or Path(ident).name != ident for ident in ids):
        raise ValueError("Legacy LoCoMo output requires unique safe conv-* sample IDs")
    samples = [row for row in all_samples if not sample_id or str(row["sample_id"]) == sample_id]
    if not samples:
        raise ValueError(f"sample not found: {sample_id}")
    manifest = _validate_resume(output, _manifest(dataset, memories, retrieval_only=retrieval_only),
                                all_samples, restart=restart, sample_id=sample_id)

    by_sample: dict[str, dict[int, dict]] = {}
    cached_by_task: dict[tuple[str, int], dict] = {}
    tasks = []
    for sample in samples:
        conversation_id = str(sample["sample_id"])
        path = output / f"{conversation_id}.json"
        prior = [] if restart else _read_array(path, required=False)
        prior_by_index: dict[int, dict] = {
            int(row["qa_index"]): row
            for row in prior
            if type(row.get("qa_index")) is int
        }
        completed = {
            index: row
            for index, row in prior_by_index.items()
            if retrieval_only or str(row.get("response") or "").strip()
        }
        # The per-conversation file is rewritten in full after every answer, so the write
        # buffer must start from *every* prior record rather than only the completed ones.
        # Otherwise a retrieval-only seed row would be dropped the moment its neighbour is
        # answered, and a resumed run would silently re-retrieve it instead of reusing the
        # configured retrieval. For an ordinary run the two sets coincide.
        by_sample[conversation_id] = dict(prior_by_index)
        selected = [
            (index, qa)
            for index, qa in enumerate(sample.get("qa") or [])
            if _category(qa.get("category")) != ADVERSARIAL_CATEGORY
        ]
        if limit is not None:
            selected = selected[:limit]
        scope = conversation_scope(store, conversation_id)
        for index, qa in selected:
            if index in completed:
                continue
            tasks.append((sample, index, qa, scope))
            # Reuse retrieval from a prior retrieval-only pass when available.
            prior_row = prior_by_index.get(index)
            if (
                not retrieval_only
                and prior_row is not None
                and prior_row.get("retrieval_only") is True
                and prior_row.get("memories") is not None
            ):
                cached_by_task[(conversation_id, index)] = prior_row

    if tasks and not restart and any(path.exists() for path in _score_paths(output)):
        raise ValueError("Answer outputs have downstream scores; use --restart before changing them")
    # Restart deletes whole selected sample files before any new task runs. A
    # smaller --limit therefore cannot retain old QA rows beyond that limit.
    output.mkdir(parents=True, exist_ok=True)
    if restart:
        selected_paths = (_answer_paths(output) if sample_id is None
                          else [output / f"{sample_id}.json"])
        for path in [*selected_paths, *_score_paths(output), output / "failures.json"]:
            path.unlink(missing_ok=True)
    jsonio.atomic_write_json(str(output / "run_manifest.json"), manifest)

    failures = []
    total = len(tasks)
    logger.info("running %d pending QA task(s) with total workers=%d", total, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                evaluate_question,
                sample,
                index,
                qa,
                scope,
                retrieval_only=retrieval_only,
                cached_record=cached_by_task.get(
                    (str(sample["sample_id"]), index)
                ),
            ): (str(sample["sample_id"]), index)
            for sample, index, qa, scope in tasks
        }
        for done, future in enumerate(as_completed(futures), start=1):
            conversation_id, index = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # successful units remain durably resumable.
                logger.exception("[%s/%d] failed", conversation_id, index)
                failures.append(
                    {
                        "conversation_id": conversation_id,
                        "qa_index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                record = _seal_record(record, manifest)
                by_sample[conversation_id][index] = record
                ordered = [
                    by_sample[conversation_id][key]
                    for key in sorted(by_sample[conversation_id])
                ]
                jsonio.atomic_write_json(
                    str(output / f"{conversation_id}.json"), ordered
                )
            logger.info("progress %d/%d", done, total)
    jsonio.atomic_write_json(str(output / "failures.json"), failures)
    if failures:
        raise RuntimeError(
            f"{len(failures)} question(s) failed; rerun the same command to resume"
        )
    return {sample_id: len(rows) for sample_id, rows in by_sample.items()}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=config.EVAL_QA_WORKERS)
    parser.add_argument("--restart", action="store_true",
                        help="clear selected answer files and all score state; omit --sample to accept changed inputs/configuration")
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    logger.info(
        "DONE: %s",
        run_evaluation(
            dataset_path=args.dataset,
            memory_dir=args.memory_dir,
            output_dir=args.output_dir,
            sample_id=args.sample,
            limit=args.limit,
            workers=args.workers,
            restart=args.restart,
            retrieval_only=args.retrieval_only,
        ),
    )


if __name__ == "__main__":
    main()
