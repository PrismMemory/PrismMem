"""Memory construction with the default LoCoMo extraction protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
for _path in (_SRC, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import config
import embedding
import jsonio
import loader
from checkpoint import Checkpoint
from generation import (
    TYPED_LONG_TYPES,
    extract_mid_memories,
    extract_typed_long_memories,
    resolve_typed_long_memory_types,
)
from generation.long_typed import SUPPORTED_LONG_TYPES
from generation.util import new_id, now
from retrieval import mid_index_text

logger = logging.getLogger("prism.extract_locomo")
_WRITE_LOCK = threading.RLock()

_MID_FILE = "mid_memories.json"
_LONG_FILES = {
    "core": "long_core.json",
    "episodic": "long_episodic.json",
    "knowledge": "long_knowledge.json",
}
_MID_FAILURE_FILE = "mid_extraction_failures.json"
_LONG_FAILURE_FILE = "long_extraction_failures.json"
_MID_PROGRESS_FILE = "progress_mid_extraction.json"
_LONG_PROGRESS_FILE = "progress_long_extraction.json"
_PROMPT_BY_TYPE = {
    "core": "memory_extraction_core",
    "episodic": "memory_extraction_episodic",
    "knowledge": "memory_extraction_knowledge",
}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _session_number(session_id: str) -> int:
    value = _clean(session_id)
    if not value.startswith("D") or not value[1:].isdigit():
        raise ValueError(f"invalid session id: {session_id!r}")
    return int(value[1:])


def _read_list(path: Path) -> list[dict]:
    payload = jsonio.read_json(str(path), default=[])
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    if any(not isinstance(row, dict) for row in payload):
        raise ValueError(f"expected only objects in JSON array: {path}")
    return payload


def _record_sort_key(record: dict) -> tuple:
    return (
        _clean(record.get("conversation_id")),
        _session_number(_clean(record.get("session_id"))),
        _clean(record.get("user_id")),
        _clean(record.get("type")),
        _clean(record.get("id")),
    )


def _replace_scope_rows(
    path: Path,
    rows: list[dict],
    *,
    conversation_id: str,
    session_id: str,
) -> None:
    existing = [
        row
        for row in _read_list(path)
        if not (
            _clean(row.get("conversation_id")) == conversation_id
            and _clean(row.get("session_id")) == session_id
        )
    ]
    existing.extend(rows)
    existing.sort(key=_record_sort_key)
    jsonio.atomic_write_json(str(path), existing)


def _work_items(
    samples: list[dict],
    *,
    sample_id: str | None,
    session_number: int | None,
) -> list[tuple[dict, dict]]:
    selected_samples = samples
    if sample_id is not None:
        selected_samples = [
            sample
            for sample in samples
            if _clean(sample.get("sample_id")) == sample_id
        ]
        if not selected_samples:
            raise ValueError(f"sample not found: {sample_id}")

    target_session = f"D{session_number}" if session_number is not None else None
    items: list[tuple[dict, dict]] = []
    for sample in selected_samples:
        matched = False
        for session in loader.iter_sessions(sample):
            if target_session is not None and session["session_id"] != target_session:
                continue
            matched = True
            items.append((sample, session))
        if target_session is not None and not matched:
            raise ValueError(
                f"session {session_number} not found in sample {sample['sample_id']}"
            )
    return items


def _normalize_mid_source_chat_ids(
    mids: list[dict],
    *,
    conversation_id: str,
    session_id: str,
    turns: list[dict],
) -> int:
    """Validate semantic-scope provenance and repair only unambiguous bare numeric aliases."""
    available_ids = {
        _clean(turn.get("dia_id"))
        for turn in turns
        if _clean(turn.get("dia_id"))
    }
    numeric_aliases: dict[str, str] = {}
    ambiguous_aliases: set[str] = set()
    for chat_id in available_ids:
        prefix, separator, suffix = chat_id.partition(":")
        if separator != ":" or prefix != session_id or not suffix.isdigit():
            continue
        alias = str(int(suffix))
        existing = numeric_aliases.get(alias)
        if existing is not None and existing != chat_id:
            ambiguous_aliases.add(alias)
        else:
            numeric_aliases[alias] = chat_id
    for alias in ambiguous_aliases:
        numeric_aliases.pop(alias, None)

    repair_count = 0
    for mid in mids:
        mid_id = _clean(mid.get("id")) or "(missing-id)"
        raw_chat_ids = mid.get("chat_ids")
        if not isinstance(raw_chat_ids, list) or not raw_chat_ids:
            raise ValueError(
                f"{conversation_id}/{session_id}/{mid_id}: source Mid has no chat_ids"
            )
        normalized: list[str] = []
        unknown: list[str] = []
        for raw_chat_id in raw_chat_ids:
            chat_id = _clean(raw_chat_id)
            resolved = chat_id if chat_id in available_ids else None
            if resolved is None and chat_id.isdigit():
                resolved = numeric_aliases.get(str(int(chat_id)))
            if resolved is None:
                unknown.append(chat_id or repr(raw_chat_id))
                continue
            normalized.append(resolved)
            repair_count += resolved != chat_id
        if unknown:
            raise ValueError(
                f"{conversation_id}/{session_id}/{mid_id}: source Mid references "
                f"unknown chat_ids {sorted(set(unknown))}"
            )
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                f"{conversation_id}/{session_id}/{mid_id}: source Mid references "
                "duplicate chat_ids"
            )
        mid["chat_ids"] = normalized
    return repair_count


def _system_persona_mids(sample: dict, session: dict) -> list[dict]:
    """Retain PersonaMem's initial D0 persona as a semantic scope via direct source-text projection."""
    if session["session_id"] != "D0":
        return []
    turns = [turn for turn in session["turns"] if turn.get("source_role") == "system"]
    if not turns:
        return []
    if len(turns) != 1 or not turns[0]["text"].strip():
        raise ValueError("D0 system persona requires exactly one non-empty system turn")
    owner = loader.participants(sample)[0]
    return [{
        "id": new_id(), "user_id": owner, "conversation_id": sample["sample_id"],
        "session_id": "D0", "session_date": session.get("session_date"),
        "chat_ids": [turns[0]["dia_id"]],
        "topic_subject": f"Initial system persona profile for {owner}",
        "summary": turns[0]["text"].strip(),
        "tags": ["initial persona", "system persona", owner],
        "confidence": 1.0, "created_at": now(),
        "extraction_scope": "system_persona_baseline", "source_role": "system",
    }]


def _validate_vectors(vectors: object, count: int, scope: str) -> None:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError(f"{scope}: embedding count mismatch")
    dimensions = set()
    for vector in vectors:
        if not isinstance(vector, list) or not vector or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in vector
        ) or not any(vector):
            raise ValueError(f"{scope}: embedding must be finite, nonzero numeric vector")
        dimensions.add(len(vector))
    if len(dimensions) > 1:
        raise ValueError(f"{scope}: embedding dimensions are inconsistent")


def _extract_mid_scope(
    sample: dict, session: dict, *, mid_prompt: str = "mid_extraction",
) -> list[dict]:
    conversation_id = _clean(sample["sample_id"])
    session_id = _clean(session["session_id"])
    session_date = session.get("session_date")
    history = loader.format_session_history(
        session["turns"],
        start_time=session_date,
    )
    mids = extract_mid_memories(
        session_history=history,
        conversation_id=conversation_id,
        session_id=session_id,
        session_date=session_date,
        participants=loader.participants(sample),
        require_valid_response=True,
        prompt_name=mid_prompt,
    )
    mids.extend(_system_persona_mids(sample, session))
    _normalize_mid_source_chat_ids(
        mids,
        conversation_id=conversation_id,
        session_id=session_id,
        turns=session["turns"],
    )
    if mids:
        vectors = embedding.embed_texts([mid_index_text(record) for record in mids])
        _validate_vectors(vectors, len(mids), f"{conversation_id}/{session_id}")
        for record, vector in zip(mids, vectors):
            record["embedding"] = vector
    return mids


def _validate_long_source_evidence(
    record: dict,
    *,
    source_chat_ids: list[str],
    scope: str,
) -> None:
    evidence = record.get("sourceEvidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{scope}: long memory has no sourceEvidence")
    if any(not isinstance(item, dict) or not _clean(item.get("chatId")) for item in evidence):
        raise ValueError(f"{scope}: every sourceEvidence item must identify a source chatId")
    evidence_ids = {
        _clean(item.get("chatId"))
        for item in evidence
        if isinstance(item, dict) and _clean(item.get("chatId"))
    }
    allowed = set(source_chat_ids)
    if not evidence_ids or not evidence_ids <= allowed:
        raise ValueError(
            f"{scope}: sourceEvidence chatIds {sorted(evidence_ids)} escape "
            f"source Mid chat_ids {sorted(allowed)}"
        )


def _extract_long_scope(
    sample: dict,
    session: dict,
    mids: list[dict],
    *,
    memory_types: tuple[str, ...] = TYPED_LONG_TYPES,
    participant_mode: str = "legacy_literal",
) -> dict[str, list[dict]]:
    """Extract each semantic scope's facets only from that scope's original source turns."""
    conversation_id = _clean(sample["sample_id"])
    session_id = _clean(session["session_id"])
    session_date = session.get("session_date")
    _normalize_mid_source_chat_ids(
        mids,
        conversation_id=conversation_id,
        session_id=session_id,
        turns=session["turns"],
    )
    available_chat_ids = {
        _clean(turn.get("dia_id"))
        for turn in session["turns"]
        if _clean(turn.get("dia_id"))
    }
    result = {memory_type: [] for memory_type in memory_types}
    for mid in mids:
        mid_id = _clean(mid.get("id"))
        owner = _clean(mid.get("user_id"))
        source_chat_ids = [
            _clean(chat_id)
            for chat_id in (mid.get("chat_ids") or [])
            if _clean(chat_id)
        ]
        if not mid_id or not owner:
            raise ValueError(
                f"{conversation_id}/{session_id}: source Mid needs id and user_id"
            )
        if not source_chat_ids:
            raise ValueError(
                f"{conversation_id}/{session_id}/{mid_id}: source Mid has no chat_ids"
            )
        missing_chat_ids = sorted(set(source_chat_ids) - available_chat_ids)
        if missing_chat_ids:
            raise ValueError(
                f"{conversation_id}/{session_id}/{mid_id}: source Mid references "
                f"unknown chat_ids {missing_chat_ids}"
            )
        dialogue = loader.format_dialogue_for_ids(
            session["turns"],
            source_chat_ids,
            start_time=session_date,
        )
        if not dialogue.strip():
            raise ValueError(
                f"{conversation_id}/{session_id}/{mid_id}: source dialogue is empty"
            )

        # This anchor is only for output provenance. extract_typed_long_memories uses
        # `dialogue` as source content, not the semantic scope's generated fields.
        provenance_anchor = {"id": mid_id, "user_id": owner}
        extracted = extract_typed_long_memories(
            provenance_anchor,
            dialogue,
            require_valid_response=True,
            memory_types=memory_types,
            participants=loader.participants(sample),
            participant_mode=participant_mode,
        )
        for memory_type, records in extracted.items():
            for raw in records:
                _validate_long_source_evidence(
                    raw,
                    source_chat_ids=source_chat_ids,
                    scope=f"{conversation_id}/{session_id}/{mid_id}/{memory_type}",
                )
                record = dict(raw)
                record.update(
                    {
                        "conversation_id": conversation_id,
                        "session_id": session_id,
                        "session_date": session_date,
                        "user_id": owner,
                        "type": memory_type,
                        "extraction_scope": "mid_source_original_dialogue_only",
                        "source_chat_ids": source_chat_ids,
                    }
                )
                if memory_type not in memory_types:
                    raise ValueError(f"unexpected extracted memory type: {memory_type}")
                result[memory_type].append(record)
    return result


def _persist_mid_scope(
    output_dir: Path,
    rows: list[dict],
    *,
    conversation_id: str,
    session_id: str,
) -> None:
    with _WRITE_LOCK:
        retained = [row for row in _read_list(output_dir / _MID_FILE) if not (
            row.get("conversation_id") == conversation_id and row.get("session_id") == session_id
        )]
        vectors = [row.get("embedding") for row in retained + rows]
        _validate_vectors(vectors, len(vectors), "all Mid outputs")
        _replace_scope_rows(
            output_dir / _MID_FILE,
            rows,
            conversation_id=conversation_id,
            session_id=session_id,
        )


def _persist_long_scope(
    output_dir: Path,
    rows_by_type: dict[str, list[dict]],
    *,
    conversation_id: str,
    session_id: str,
) -> None:
    with _WRITE_LOCK:
        for memory_type, filename in _LONG_FILES.items():
            _replace_scope_rows(
                output_dir / filename,
                rows_by_type.get(memory_type, []),
                conversation_id=conversation_id,
                session_id=session_id,
            )


def _run_stage(
    items: list[tuple[dict, dict]],
    *,
    output_dir: Path,
    stage: str,
    workers: int,
    restart: bool,
    memory_types: tuple[str, ...] = TYPED_LONG_TYPES,
    mid_prompt: str = "mid_extraction",
    participant_mode: str = "legacy_literal",
) -> dict:
    if workers <= 0:
        raise ValueError(f"{stage} workers must be positive")
    if stage == "mid":
        checkpoint_path = output_dir / _MID_PROGRESS_FILE
        checkpoint_flow = "prism_mid_extraction"
        failure_path = output_dir / _MID_FAILURE_FILE
        def extractor(sample: dict, session: dict):
            return _extract_mid_scope(sample, session, mid_prompt=mid_prompt)
    elif stage == "long":
        checkpoint_path = output_dir / _LONG_PROGRESS_FILE
        checkpoint_flow = "prism_source_long_extraction"
        failure_path = output_dir / _LONG_FAILURE_FILE
        mid_checkpoint = Checkpoint(
            str(output_dir / _MID_PROGRESS_FILE),
            "prism_mid_extraction",
        )
        incomplete = [
            f"{_clean(sample['sample_id'])}/{_clean(session['session_id'])}"
            for sample, session in items
            if not mid_checkpoint.done(
                _clean(sample["sample_id"]),
                _session_number(session["session_id"]),
            )
        ]
        if incomplete:
            raise RuntimeError(
                "long extraction requires completed fresh Mid extraction for every "
                f"selected scope; missing {incomplete[:10]}"
            )
        mids_by_scope: dict[tuple[str, str], list[dict]] = {}
        for mid in _read_list(output_dir / _MID_FILE):
            scope = (
                _clean(mid.get("conversation_id")),
                _clean(mid.get("session_id")),
            )
            mids_by_scope.setdefault(scope, []).append(mid)

        def extractor(sample: dict, session: dict):
            scope = (_clean(sample["sample_id"]), _clean(session["session_id"]))
            return _extract_long_scope(
                sample,
                session,
                mids_by_scope.get(scope, []),
                memory_types=memory_types,
                participant_mode=participant_mode,
            )
    else:
        raise ValueError(f"unsupported stage: {stage}")

    checkpoint = Checkpoint(str(checkpoint_path), checkpoint_flow)
    if restart:
        for conversation_id in sorted(
            {_clean(sample["sample_id"]) for sample, _ in items}
        ):
            checkpoint.clear(conversation_id)

    selected: list[tuple[dict, dict]] = []
    skipped = 0
    for sample, session in items:
        conversation_id = _clean(sample["sample_id"])
        session_no = _session_number(session["session_id"])
        if not restart and checkpoint.done(conversation_id, session_no):
            skipped += 1
            continue
        selected.append((sample, session))

    failures: list[dict] = []
    counts: Counter[str] = Counter()
    logger.info(
        "%s extraction: %d requested, %d selected, %d skipped, workers=%d",
        stage,
        len(items),
        len(selected),
        skipped,
        workers,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_by_scope = {
            pool.submit(extractor, sample, session): (sample, session)
            for sample, session in selected
        }
        for completed, future in enumerate(as_completed(future_by_scope), start=1):
            sample, session = future_by_scope[future]
            conversation_id = _clean(sample["sample_id"])
            session_id = _clean(session["session_id"])
            try:
                payload = future.result()
                if stage == "mid":
                    assert isinstance(payload, list)
                    _persist_mid_scope(
                        output_dir,
                        payload,
                        conversation_id=conversation_id,
                        session_id=session_id,
                    )
                    counts["mid"] += len(payload)
                else:
                    assert isinstance(payload, dict)
                    _persist_long_scope(
                        output_dir,
                        payload,
                        conversation_id=conversation_id,
                        session_id=session_id,
                    )
                    for memory_type, rows in payload.items():
                        counts[memory_type] += len(rows)
                checkpoint.mark(
                    conversation_id, _session_number(session_id),
                    output_sha256=_scope_output_hash(
                        output_dir, stage, conversation_id, session_id,
                    ),
                )
                logger.info(
                    "[%s/%s] %s extraction complete (%d/%d)",
                    conversation_id,
                    session_id,
                    stage,
                    completed,
                    len(selected),
                )
            except Exception as exc:
                logger.exception(
                    "[%s/%s] %s extraction failed",
                    conversation_id,
                    session_id,
                    stage,
                )
                failures.append(
                    {
                        "conversation_id": conversation_id,
                        "session_id": session_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    jsonio.atomic_write_json(str(failure_path), failures)
    return {
        "requested_session_count": len(items),
        "processed_session_count": len(selected) - len(failures),
        "skipped_session_count": skipped,
        "failure_count": len(failures),
        "new_memory_counts": dict(sorted(counts.items())),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _scope_output_hash(output_dir: Path, stage: str, cid: str, sid: str) -> str:
    files = [_MID_FILE] if stage == "mid" else list(_LONG_FILES.values())
    payload = {}
    for filename in files:
        path = output_dir / filename
        if not path.is_file():
            raise ValueError(f"missing committed {stage} output: {filename}")
        payload[filename] = sorted([
            row for row in _read_list(path)
            if row.get("conversation_id") == cid and row.get("session_id") == sid
        ], key=_record_sort_key)
    return _digest(payload)


def _extraction_contract(
    dataset_path: Path, items: list[tuple[dict, dict]], *, memory_types: tuple[str, ...],
    mid_prompt: str, participant_mode: str,
) -> dict:
    prompt_names = (mid_prompt,) + tuple(_PROMPT_BY_TYPE[kind] for kind in memory_types)
    source_files = (
        "eval/extract_locomo.py", "eval/loader.py", "eval/checkpoint.py", "jsonio.py",
        "generation/extract.py", "generation/long_typed.py", "generation/util.py",
        "embedding/encoder.py", "llm/client.py", "llm/providers.py",
        "llm/prompts/__init__.py", "retrieval/projection.py",
    )
    # Credentials never enter artifacts. Hash endpoint identities since gateway URLs
    # may themselves include deployment-specific tokens.
    endpoints = {key: _digest(getattr(config, key, None)) for key in (
        "EXTRACTION_BASE_URL", "PRISM_BASE_URL",
    )}
    return {
        "schema_version": 2,
        "dataset_sha256": _file_sha256(dataset_path),
        "corpus_sha256": _digest(items),
        "scopes": [[sample["sample_id"], session["session_id"]] for sample, session in items],
        "models": {"extraction": config.EXTRACTION_MODEL, "embedding": config.PRISM_MODEL},
        "extraction_settings": {
            "memory_types": list(memory_types), "mid_prompt": mid_prompt,
            "participant_mode": participant_mode,
            "enable_thinking": config.EXTRACTION_ENABLE_THINKING,
            "max_tokens": config.LLM_MAX_TOKENS,
            "embedding_batch_size": config.PRISM_BATCH_SIZE,
        },
        "endpoint_sha256": endpoints,
        "prompt_sha256": {
            name: _file_sha256(_SRC / "llm" / "prompts" / f"{name}.txt")
            for name in prompt_names
        },
        "code_sha256": {name: _file_sha256(_SRC / name) for name in source_files},
    }


def _validate_committed_outputs(
    output_dir: Path, items: list[tuple[dict, dict]], *, only_mid: bool = False,
) -> None:
    """Verify durable output contents rather than trusting completed unit flags."""
    scopes = {(sample["sample_id"], session["session_id"]) for sample, session in items}
    for stage, filename, flow in (
        ("mid", _MID_PROGRESS_FILE, "prism_mid_extraction"),
        ("long", _LONG_PROGRESS_FILE, "prism_source_long_extraction"),
    ):
        if only_mid and stage != "mid":
            continue
        checkpoint = Checkpoint(str(output_dir / filename), flow)
        for cid, units in checkpoint.completed().items():
            for unit in units:
                sid = f"D{unit}"
                if (cid, sid) not in scopes:
                    raise ValueError(f"checkpoint contains an unknown scope: {cid}/{sid}")
                expected = checkpoint.output_sha256(cid, unit)
                if not expected or expected != _scope_output_hash(output_dir, stage, cid, sid):
                    raise ValueError(f"{cid}/{sid}: {stage} checkpoint/output mismatch; use a new output directory or --restart")
    mids = _read_list(output_dir / _MID_FILE)
    _validate_vectors([row.get("embedding") for row in mids], len(mids), "saved Mid outputs")
    ids: set[str] = set()
    for filename in ((_MID_FILE,) if only_mid else (_MID_FILE, *_LONG_FILES.values())):
        for row in _read_list(output_dir / filename):
            if (row.get("conversation_id"), row.get("session_id")) not in scopes:
                raise ValueError(f"{filename}: output row outside selected corpus")
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id or record_id in ids:
                raise ValueError(f"{filename}: empty or duplicate memory id")
            ids.add(record_id)


def _owned_output_names(output_dir: Path) -> set[str]:
    """Extraction-owned files and derived sidecars in this memory directory."""
    names = {
        _MID_FILE, *_LONG_FILES.values(), _MID_FAILURE_FILE, _LONG_FAILURE_FILE,
        _MID_PROGRESS_FILE, _LONG_PROGRESS_FILE, "reextraction_manifest.json",
        "reextraction_metrics.json", "long_embeddings.json",
        "long_embeddings_manifest.json", "long_embeddings_metrics.json",
        "mid_enriched_embeddings.json", "mid_enriched_embeddings_all.json",
    }
    return {name for name in names if (output_dir / name).exists()}


def _check_resume_contract(
    output_dir: Path, contract: dict, items: list[tuple[dict, dict]], *, restart: bool, phase: str,
) -> None:
    manifest_path = output_dir / "reextraction_manifest.json"
    previous = jsonio.read_json(str(manifest_path), default=None)
    matches = isinstance(previous, dict) and previous.get("contract") == contract and (
        previous.get("contract_sha256") == _digest(contract)
    )
    occupied = bool(_owned_output_names(output_dir))
    if occupied and not matches and (not restart or phase == "long"):
        raise ValueError(
            "extraction fingerprint mismatch or missing compatible manifest; use a new "
            "output directory or --restart with --phase all/mid"
        )
    if not restart and matches:
        _validate_committed_outputs(output_dir, items)
    if restart and phase == "long":
        # Reusing semantic scopes still requires their intact outputs and original contract.
        _validate_committed_outputs(output_dir, items, only_mid=True)
    if phase == "long":
        checkpoint = Checkpoint(
            str(output_dir / _MID_PROGRESS_FILE), "prism_mid_extraction",
        )
        if any(not checkpoint.done(sample["sample_id"], _session_number(session["session_id"]))
               for sample, session in items):
            raise ValueError("long extraction requires completed Mid outputs for every selected scope")
    # This function is read-only. Cleanup and manifest writes occur only after it returns.


def _restart_outputs(output_dir: Path, *, phase: str) -> None:
    names = _owned_output_names(output_dir)
    if phase == "long":
        names -= {_MID_FILE, _MID_PROGRESS_FILE, _MID_FAILURE_FILE, "reextraction_manifest.json"}
    for name in sorted(names):
        path = output_dir / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"refusing to remove non-regular extraction output: {path}")
    for name in sorted(names):
        (output_dir / name).unlink()


def _write_manifest(
    output_dir: Path,
    *,
    dataset_path: Path,
    mid_workers: int,
    long_workers: int,
    memory_types: tuple[str, ...],
    contract: dict,
) -> None:
    prompt_dir = _SRC / "llm" / "prompts"
    mid_prompt = contract["extraction_settings"]["mid_prompt"]
    participant_mode = contract["extraction_settings"]["participant_mode"]
    prompt_names = (mid_prompt,) + tuple(
        _PROMPT_BY_TYPE[memory_type] for memory_type in memory_types
    )
    jsonio.atomic_write_json(
        str(output_dir / "reextraction_manifest.json"),
        {
            "schema_version": 2,
            "pipeline": "prism_memory_construction",
            "contract": contract,
            "contract_sha256": _digest(contract),
            "dataset": {
                "path": str(dataset_path.resolve()),
                "sha256": _file_sha256(dataset_path),
            },
            "models": {
                "memory_extraction": config.EXTRACTION_MODEL,
                "mid_embedding": config.PRISM_MODEL,
            },
            "mid_extraction": {
                "prompt": mid_prompt,
                "beam_prompt_provenance": (
                    "BEAM-specific Mid schema extension for dialogue state"
                    if mid_prompt == "mid_extraction_beam" else None
                ),
                "input": "complete original session transcript",
                "unit": "one source session",
                "strict_response_validation": True,
                "participants_placeholder_substituted": True,
                "participants_validation_source": "LoCoMo speaker_a and speaker_b",
                "chat_id_validation": (
                    "same-session raw dia_id; unambiguous bare numeric aliases normalized"
                ),
                "embedding_fields": "topic_subject + summary",
                "workers": mid_workers,
                "output": _MID_FILE,
            },
            "long_extraction": {
                "source_selector": "each freshly extracted Mid's chat_ids",
                "model_input": "only selected raw original dialogue turns",
                "unit": "one source Mid per selected memory type",
                "memory_types": list(memory_types),
                "strict_response_validation": True,
                "uses_mid_memory_content": False,
                "uses_mid_topic_or_summary": False,
                "uses_mid_tags": False,
                "uses_mid_identifier_in_prompt": False,
                "participants_placeholder_substituted": participant_mode == "explicit",
                "participant_mode": participant_mode,
                "participants_visible_in_raw_dialogue_speaker_labels": True,
                "source_evidence_restricted_to_mid_chat_ids": True,
                "workers": long_workers,
                "outputs": {
                    memory_type: _LONG_FILES[memory_type]
                    for memory_type in memory_types
                },
            },
            "relations_extracted": False,
            "prompt_sha256": {
                name: _file_sha256(prompt_dir / f"{name}.txt")
                for name in prompt_names
            },
            "implementation": {
                "runner": str(Path(__file__).resolve()),
                "runner_sha256": _file_sha256(Path(__file__).resolve()),
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _checkpoint_coverage(path: Path, items: list[tuple[dict, dict]]) -> dict:
    payload = jsonio.read_json(str(path), default={}) or {}
    completed = payload.get("completed") or {}
    expected = {
        (_clean(sample["sample_id"]), _session_number(session["session_id"]))
        for sample, session in items
    }
    actual = {
        (_clean(conversation_id), int(session_no))
        for conversation_id, session_numbers in completed.items()
        for session_no in session_numbers
    }
    return {
        "expected_session_count": len(expected),
        "completed_session_count": len(expected & actual),
        "complete": expected <= actual,
        "missing": [
            f"{conversation_id}/D{session_no}"
            for conversation_id, session_no in sorted(expected - actual)
        ],
    }


def _write_metrics(
    output_dir: Path,
    *,
    items: list[tuple[dict, dict]],
    mid_run: dict | None,
    long_run: dict | None,
    memory_types: tuple[str, ...],
) -> dict:
    mids = _read_list(output_dir / _MID_FILE)
    longs_by_type = {
        memory_type: _read_list(output_dir / filename)
        for memory_type, filename in _LONG_FILES.items()
    }
    mid_coverage = _checkpoint_coverage(output_dir / _MID_PROGRESS_FILE, items)
    long_coverage = _checkpoint_coverage(output_dir / _LONG_PROGRESS_FILE, items)
    long_rows = [row for rows in longs_by_type.values() for row in rows]
    mid_by_id = {
        _clean(row.get("id")): row for row in mids if _clean(row.get("id"))
    }
    metrics = {
        "configured_long_memory_types": list(memory_types),
        "mid_memory_count": len(mids),
        "mid_missing_embedding_count": sum(
            not isinstance(row.get("embedding"), list) or not row.get("embedding")
            for row in mids
        ),
        "long_memory_counts": {
            memory_type: len(rows)
            for memory_type, rows in longs_by_type.items()
        },
        "long_memory_total": len(long_rows),
        "long_records_missing_mid_id": sum(
            not _clean(row.get("mid_id")) for row in long_rows
        ),
        "long_records_with_unknown_mid_id": sum(
            _clean(row.get("mid_id")) not in mid_by_id for row in long_rows
        ),
        "long_records_with_source_chat_id_mismatch": sum(
            set(row.get("source_chat_ids") or [])
            != set(mid_by_id.get(_clean(row.get("mid_id")), {}).get("chat_ids") or [])
            for row in long_rows
        ),
        "long_records_with_wrong_scope": sum(
            row.get("extraction_scope") != "mid_source_original_dialogue_only"
            for row in long_rows
        ),
        "coverage": {"mid": mid_coverage, "long": long_coverage},
        "current_run": {"mid": mid_run, "long": long_run},
    }
    metrics["complete_and_valid"] = bool(
        mid_coverage["complete"]
        and long_coverage["complete"]
        and metrics["mid_missing_embedding_count"] == 0
        and metrics["long_records_missing_mid_id"] == 0
        and metrics["long_records_with_unknown_mid_id"] == 0
        and metrics["long_records_with_source_chat_id_mismatch"] == 0
        and metrics["long_records_with_wrong_scope"] == 0
        and not (mid_run or {}).get("failure_count")
        and not (long_run or {}).get("failure_count")
    )
    jsonio.atomic_write_json(str(output_dir / "reextraction_metrics.json"), metrics)
    return metrics


def run_pipeline(
    dataset_path: str,
    output_dir: str,
    *,
    phase: str = "all",
    sample_id: str | None = None,
    session_number: int | None = None,
    mid_workers: int = 2,
    long_workers: int = 4,
    restart: bool = False,
    expected_model: str = "glm-5.2",
    memory_types: tuple[str, ...] | list[str] | None = None,
    mid_prompt: str = "mid_extraction",
    participant_mode: str = "legacy_literal",
) -> dict:
    """Build semantic scopes and facets from a conversation-only corpus.

    LoCoMo defaults use durable-state, event, and knowledge facets and the literal
    PARTICIPANTS placeholder. Converted PersonaMem/LongMemEval use explicit participants;
    BEAM uses its extended semantic-scope prompt and the same three facet types.
    Semantic-scope extraction receives actual participant names. No relations are built.

    Resume requires identical dataset contents, selected scopes, model/settings,
    prompt bytes and extraction implementation, plus intact committed outputs.
    ``restart=True`` with all/mid clears this directory's extraction artifacts and
    derived embedding sidecars; long-only restart retains verified semantic-scope outputs and
    requires the same extraction contract. Evaluation outputs elsewhere are the
    outer benchmark runner's responsibility. Workers and phase may change on resume.
    """
    if phase not in {"all", "mid", "long"}:
        raise ValueError(f"unsupported phase: {phase}")
    if session_number is not None and sample_id is None:
        raise ValueError("--session requires --sample")
    if mid_workers <= 0 or long_workers <= 0:
        raise ValueError("extraction workers must be positive")
    if mid_prompt not in {"mid_extraction", "mid_extraction_beam"}:
        raise ValueError(f"unsupported Mid prompt: {mid_prompt}")
    if participant_mode not in {"legacy_literal", "explicit"}:
        raise ValueError("participant_mode must be legacy_literal or explicit")
    if config.EXTRACTION_MODEL.casefold() != expected_model.casefold():
        raise ValueError(
            "memory-model mismatch: "
            f"configured={config.EXTRACTION_MODEL!r}, expected={expected_model!r}; "
            "set PRISM_EXTRACTION_MODEL before starting Python"
        )
    if not config.EXTRACTION_API_KEY:
        raise ValueError("PRISM_EXTRACTION_API_KEY is required for extraction")
    selected_memory_types = resolve_typed_long_memory_types(memory_types)

    dataset = Path(dataset_path).resolve()
    if not dataset.is_file():
        raise ValueError(f"dataset does not exist: {dataset}")
    destination = Path(output_dir).resolve()
    dataset_digest_before_load = _file_sha256(dataset)
    samples = loader.extraction_samples(loader.load_samples(str(dataset)))
    items = _work_items(
        samples,
        sample_id=sample_id,
        session_number=session_number,
    )
    if not items:
        raise ValueError("no non-empty extraction sessions selected")
    contract = _extraction_contract(
        dataset, items, memory_types=selected_memory_types,
        mid_prompt=mid_prompt, participant_mode=participant_mode,
    )
    if contract["dataset_sha256"] != dataset_digest_before_load:
        raise ValueError("dataset changed during extraction preflight; retry with a stable input")
    if dataset.parent == destination and dataset.name in _owned_output_names(destination):
        raise ValueError("dataset input cannot overwrite or reuse an extraction-owned output file")
    _check_resume_contract(destination, contract, items, restart=restart, phase=phase)
    # Everything above is read-only and runs before any remote call or output write.
    if restart:
        _restart_outputs(destination, phase=phase)
    destination.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        destination,
        dataset_path=dataset,
        mid_workers=mid_workers,
        long_workers=long_workers,
        memory_types=selected_memory_types,
        contract=contract,
    )
    for filename in (_MID_FILE, *_LONG_FILES.values(), _MID_FAILURE_FILE, _LONG_FAILURE_FILE):
        path = destination / filename
        if not path.exists():
            jsonio.atomic_write_json(str(path), [])

    mid_run = None
    long_run = None
    if phase in {"all", "mid"}:
        mid_run = _run_stage(
            items,
            output_dir=destination,
            stage="mid",
            workers=mid_workers,
            restart=False,
            memory_types=selected_memory_types,
            mid_prompt=mid_prompt,
            participant_mode=participant_mode,
        )
        if mid_run.get("failure_count"):
            _write_metrics(destination, items=items, mid_run=mid_run, long_run=None,
                           memory_types=selected_memory_types)
            raise RuntimeError("Mid extraction scopes failed; rerun the same command to resume")
    if phase in {"all", "long"}:
        long_run = _run_stage(
            items,
            output_dir=destination,
            stage="long",
            workers=long_workers,
            restart=False,
            memory_types=selected_memory_types,
            mid_prompt=mid_prompt,
            participant_mode=participant_mode,
        )
    metrics = _write_metrics(
        destination,
        items=items,
        mid_run=mid_run,
        long_run=long_run,
        memory_types=selected_memory_types,
    )
    if (mid_run or {}).get("failure_count") or (long_run or {}).get("failure_count"):
        raise RuntimeError(
            "one or more extraction scopes failed; rerun the same command to resume"
        )
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parents[2] / "datasets" / "locomo10.json"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("all", "mid", "long"), default="all")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--mid-workers", type=int, default=2)
    parser.add_argument("--long-workers", type=int, default=4)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--expected-model", default="glm-5.2")
    parser.add_argument(
        "--memory-types",
        nargs="+",
        choices=SUPPORTED_LONG_TYPES,
        default=list(TYPED_LONG_TYPES),
    )
    parser.add_argument("--mid-prompt", choices=("mid_extraction", "mid_extraction_beam"), default="mid_extraction")
    parser.add_argument("--participant-mode", choices=("legacy_literal", "explicit"), default="legacy_literal")
    args = parser.parse_args()
    metrics = run_pipeline(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        phase=args.phase,
        sample_id=args.sample,
        session_number=args.session,
        mid_workers=args.mid_workers,
        long_workers=args.long_workers,
        restart=args.restart,
        expected_model=args.expected_model,
        memory_types=args.memory_types,
        mid_prompt=args.mid_prompt,
        participant_mode=args.participant_mode,
    )
    logger.info("DONE: %s", metrics)


if __name__ == "__main__":
    main()
