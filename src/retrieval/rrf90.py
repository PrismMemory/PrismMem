"""Representation-isolated RRF retrieval across three channels with strict reranking."""

from __future__ import annotations

import hashlib
import math
import threading
from array import array
from collections import Counter, OrderedDict
from typing import Callable, Sequence

try:
    import config
except ImportError:  # pragma: no cover - makes this module independently inspectable.
    config = None  # type: ignore[assignment]

from .bm25 import BM25, tokenize
from .projection import long_retrieval_text, mid_retrieval_content, text_sha256
from .rerank import reciprocal_rank_fusion, rerank_scored
from .result import RetrievalResult

MODE = "prism_rrf90_rerank_mid10_core60_other50"
RRF_K = 60
RRF_POOL_TOP_N = 70
MID_TOP_N = 10
CORE_TOP_N = 60
OTHER_TOP_N = 50
LONG_MIN_SCORE = 0.5
_VECTOR_FIELDS = frozenset({"embedding", "anonymous_embedding"})


def _cfg(name: str, default: object) -> object:
    return getattr(config, name, default) if config is not None else default


# The constants above provide defaults. Overrides come from ``config`` through
# environment variables and are resolved per call.
def _rrf_k() -> int:
    return int(_cfg("RRF_K", RRF_K))


def _pool_top_n() -> int:
    return int(_cfg("RRF_CANDIDATE_TOP_N", RRF_POOL_TOP_N))


def _mid_top_n() -> int:
    return int(_cfg("MID_RERANK_TOP_N", MID_TOP_N))


def _core_top_n() -> int:
    return int(_cfg("CORE_RERANK_TOP_N", CORE_TOP_N))


def _other_top_n() -> int:
    return int(_cfg("OTHER_RERANK_TOP_N", OTHER_TOP_N))


def _long_min_score() -> float:
    return float(_cfg("LONG_RERANK_MIN_SCORE", LONG_MIN_SCORE))


def _finite_vector(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in value
        )
    )


def _validate_records(records: list[dict], *, pool: str) -> list[dict]:
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get("id") or "").strip()
        if not record_id:
            raise ValueError(f"{pool} memory requires a non-empty id")
        if record_id in seen:
            raise ValueError(f"duplicate {pool} memory id: {record_id}")
        seen.add(record_id)
        if not _finite_vector(record.get("embedding")):
            raise ValueError(f"{pool} memory {record_id} has no finite stored embedding")
    return records


def _validate_dimensions(query_vector: list[float], records: list[dict]) -> None:
    for record in records:
        if len(record["embedding"]) != len(query_vector):
            raise ValueError(
                f"memory {record['id']} embedding dimension {len(record['embedding'])} "
                f"does not match query dimension {len(query_vector)}"
            )


def _validate_long_provenance(records: list[dict]) -> None:
    """Refuse stale sidecars, including knowledge-facet embeddings predating Name projection."""
    expected_model = str(_cfg("PRISM_MODEL", "text-embedding-v4"))
    for record in records:
        record_id = str(record["id"])
        if record.get("embedding_model") != expected_model:
            raise ValueError(
                f"long memory {record_id} has no {expected_model} embedding metadata"
            )
        expected_hash = text_sha256(long_retrieval_text(record))
        if record.get("embedding_text_sha256") != expected_hash:
            raise ValueError(
                f"long memory {record_id} embedding text is stale; rebuild embeddings"
            )


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


# Every question of a conversation scores the same channels, so the per-record vector norm
# and the tokenized BM25 corpus are recomputed identically once per question. Both are
# memoized below. The cached values are bit-for-bit what the uncached path produced:
# only the number of times they are computed changes.
_NORM_CACHE: dict[tuple[str, bytes], float] = {}
_NORM_CACHE_LOCK = threading.Lock()
_NORM_CACHE_MAX = 200_000


def _record_norm(record: dict) -> float:
    # IDs may be reused across datasets or after embeddings are rebuilt. Snapshot the
    # actual float64 values so replacements and in-place edits cannot reuse a stale norm.
    vector = array("d", record["embedding"])
    key = (str(record["id"]), hashlib.sha256(vector).digest())
    with _NORM_CACHE_LOCK:
        cached = _NORM_CACHE.get(key)
    if cached is not None:
        return cached
    norm = _vector_norm(vector)
    with _NORM_CACHE_LOCK:
        if len(_NORM_CACHE) >= _NORM_CACHE_MAX:
            _NORM_CACHE.clear()
        _NORM_CACHE[key] = norm
    return norm


def _embedding_hits(
    query_vector: list[float], records: list[dict]
) -> list[tuple[dict, float]]:
    """Rank the complete channel by cosine; ties retain input order."""
    query_norm = _vector_norm(query_vector)
    scored: list[tuple[dict, float]] = []
    for record in records:
        vector = record["embedding"]
        record_norm = _record_norm(record)
        if query_norm == 0.0 or record_norm == 0.0:
            scored.append((record, 0.0))
            continue
        dot = sum(float(a) * float(b) for a, b in zip(query_vector, vector))
        scored.append((record, dot / (query_norm * record_norm)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


_BM25_CACHE: OrderedDict[tuple, BM25] = OrderedDict()
_BM25_CACHE_LOCK = threading.Lock()
_BM25_BUILD_LOCKS: dict[tuple, threading.Lock] = {}
# Three channels of the sample in flight plus the tail of the previous one; the tokenized
# corpus of a large channel is the dominant resident object, so the window stays small.
_BM25_CACHE_MAX = 6


def _bm25_for_corpus(texts: list[str]) -> BM25:
    """Build the BM25 index once per (corpus, lemmatization) pair, then reuse it."""
    key = (
        bool(_cfg("BM25_LEMMATIZATION_ENABLED", True)),
        len(texts),
        hash(tuple(texts)),
    )
    with _BM25_CACHE_LOCK:
        cached = _BM25_CACHE.get(key)
        if cached is not None:
            _BM25_CACHE.move_to_end(key)
            return cached
        build_lock = _BM25_BUILD_LOCKS.setdefault(key, threading.Lock())
    # Concurrent questions of the same sample wait for one build instead of each paying
    # the full tokenization cost; the losers re-read the cache below.
    with build_lock:
        with _BM25_CACHE_LOCK:
            cached = _BM25_CACHE.get(key)
            if cached is not None:
                _BM25_CACHE.move_to_end(key)
                return cached
        index = BM25([tokenize(text) for text in texts])
        with _BM25_CACHE_LOCK:
            _BM25_CACHE[key] = index
            _BM25_CACHE.move_to_end(key)
            while len(_BM25_CACHE) > _BM25_CACHE_MAX:
                evicted, _value = _BM25_CACHE.popitem(last=False)
                _BM25_BUILD_LOCKS.pop(evicted, None)
        return index


def clear_pool_caches() -> None:
    """Drop the memoized norms and BM25 indexes (tests and long-lived processes)."""
    with _NORM_CACHE_LOCK:
        _NORM_CACHE.clear()
    with _BM25_CACHE_LOCK:
        _BM25_CACHE.clear()
        _BM25_BUILD_LOCKS.clear()


def _hybrid_seed_hits(
    question: str,
    query_vector: list[float],
    records: list[dict],
    text_of: Callable[[dict], str],
    top_n: int | None = None,
) -> tuple[list[tuple[dict, dict]], dict]:
    """Fuse full embedding ranking and only positive-score BM25 results."""
    top_n = _pool_top_n() if top_n is None else top_n
    rrf_k = _rrf_k()
    if not records or top_n <= 0:
        return [], {
            "candidate_count": len(records),
            "embedding_ranking_count": 0,
            "bm25_positive_count": 0,
            "top_n": max(0, top_n),
            "selected_count": 0,
            "selected": [],
        }

    embedding_hits = _embedding_hits(query_vector, records)
    embedding_ids = [str(record["id"]) for record, _score in embedding_hits]
    embedding_score_by_id = {
        str(record["id"]): score for record, score in embedding_hits
    }
    embedding_rank_by_id = {
        record_id: rank for rank, record_id in enumerate(embedding_ids)
    }

    bm25 = _bm25_for_corpus([text_of(record) for record in records])
    bm25_scores = bm25.scores(tokenize(question))
    bm25_order = sorted(
        (index for index, score in enumerate(bm25_scores) if score > 0.0),
        key=lambda index: (-bm25_scores[index], index),
    )
    bm25_ids = [str(records[index]["id"]) for index in bm25_order]
    bm25_score_by_id = {
        str(records[index]["id"]): bm25_scores[index] for index in bm25_order
    }
    bm25_rank_by_id = {
        record_id: rank for rank, record_id in enumerate(bm25_ids)
    }

    rankings = [embedding_ids]
    if bm25_ids:
        rankings.append(bm25_ids)
    fused_ids = reciprocal_rank_fusion(rankings, k=rrf_k)
    rrf_score_by_id: dict[str, float] = {}
    for ranking in rankings:
        for rank, record_id in enumerate(ranking):
            rrf_score_by_id[record_id] = rrf_score_by_id.get(record_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )

    by_id = {str(record["id"]): record for record in records}
    selected: list[tuple[dict, dict]] = []
    selected_trace: list[dict] = []
    for record_id in fused_ids[:top_n]:
        metadata = {
            "rrf_score": rrf_score_by_id[record_id],
            "embedding_score": embedding_score_by_id[record_id],
            "embedding_rank": embedding_rank_by_id[record_id] + 1,
            "bm25_score": bm25_score_by_id.get(record_id),
            "bm25_rank": (
                bm25_rank_by_id[record_id] + 1
                if record_id in bm25_rank_by_id
                else None
            ),
        }
        record = by_id[record_id]
        selected.append((record, metadata))
        selected_trace.append(
            {
                "id": record_id,
                "mid_id": record.get("mid_id"),
                "user_id": record.get("user_id"),
                "type": record.get("type"),
                **metadata,
            }
        )
    return selected, {
        "candidate_count": len(records),
        "candidate_count_by_type": dict(
            sorted(Counter(str(record.get("type") or "") for record in records).items())
        ),
        "embedding_ranking_count": len(embedding_ids),
        "bm25_positive_count": len(bm25_ids),
        "bm25_zero_score_policy": "excluded_from_bm25_ranking",
        "fusion": "reciprocal_rank_fusion",
        "rrf_k": rrf_k,
        "rrf_rank_origin": 0,
        "top_n": top_n,
        "selected_count": len(selected),
        "selected": selected_trace,
    }


def _without_vectors(record: dict) -> dict:
    return {key: value for key, value in record.items() if key not in _VECTOR_FIELDS}


def _metadata_by_id(hits: list[tuple[dict, dict]]) -> dict[str, dict]:
    return {str(record["id"]): metadata for record, metadata in hits}


def _memory_rows(
    hits: list[tuple[dict, float]],
    metadata_by_id: dict[str, dict],
    *,
    pool: str,
    text_of: Callable[[dict], str],
) -> list[dict]:
    rows: list[dict] = []
    for record, rerank_score in hits:
        rows.append(
            {
                **_without_vectors(record),
                "content": text_of(record),
                "memory_pool": pool,
                "retrieval_origin": "text_embedding_v4_bm25_rrf90_then_rerank",
                **metadata_by_id[str(record["id"])],
                "rerank_score": rerank_score,
            }
        )
    return rows


def _selected_rerank_trace(
    hits: list[tuple[dict, float]], metadata_by_id: dict[str, dict]
) -> list[dict]:
    return [
        {
            "id": record.get("id"),
            "mid_id": record.get("mid_id"),
            "user_id": record.get("user_id"),
            "type": record.get("type"),
            **metadata_by_id[str(record["id"])],
            "rerank_score": score,
        }
        for record, score in hits
    ]


def _stage(
    records: list[dict],
    rrf_trace: dict,
    hits: list[tuple[dict, float]],
    metadata: dict[str, dict],
    *,
    rerank_top_n: int,
    min_score: float | None,
    embedding_dimension: int,
    embedding_source: str,
) -> dict:
    return {
        "candidate_count": len(records),
        "rrf": {
            **rrf_trace,
            "embedding_model": str(_cfg("PRISM_MODEL", "text-embedding-v4")),
            "embedding_dimension": embedding_dimension,
            "embedding_source": embedding_source,
        },
        "rerank": {
            "candidate_count": int(rrf_trace.get("selected_count") or 0),
            "top_n": rerank_top_n,
            "min_score": min_score,
            "selected_count": len(hits),
            "selected": _selected_rerank_trace(hits, metadata),
        },
    }


def _default_embed_text(text: str) -> list[float]:
    import embedding

    return embedding.embed_text(text)


def search_rrf90(
    question: str,
    scope: dict,
    *,
    top_k: int | None = None,
    embed_text: Callable[[str], list[float]] | None = None,
    rerank_fn: Callable[..., list[tuple[dict, float]]] | None = None,
) -> RetrievalResult:
    """Search one conversation across both owners with representation-isolated retrieval.

    ``scope`` must already represent one conversation.  No owner filter, relation file,
    graph expansion, sufficiency call, or global ``top_k`` truncation is performed.
    """
    del top_k
    scope_mids = scope.get("mids") or []
    scope_core = scope.get("long_core") or []
    scope_other = [
        record
        for table in ("long_episodic", "long_knowledge")
        for record in (scope.get(table) or [])
    ]
    mids = _validate_records(
        [record for record in scope_mids if mid_retrieval_content(record)],
        pool="mid",
    )
    core = _validate_records(
        [record for record in scope_core if long_retrieval_text(record)],
        pool="core",
    )
    other = _validate_records(
        [record for record in scope_other if long_retrieval_text(record)],
        pool="other long",
    )
    all_long_ids = [str(record["id"]) for record in [*core, *other]]
    if len(all_long_ids) != len(set(all_long_ids)):
        raise ValueError("duplicate typed long-memory id across Core and Other pools")
    _validate_long_provenance([*core, *other])

    all_records = [*mids, *core, *other]
    encoder = embed_text or _default_embed_text
    query_vector = encoder(question) if all_records else []
    if all_records and not _finite_vector(query_vector):
        raise RuntimeError("query encoder returned an invalid vector")
    if all_records:
        _validate_dimensions(query_vector, all_records)

    pool_top_n = _pool_top_n()
    mid_top_n = _mid_top_n()
    core_top_n = _core_top_n()
    other_top_n = _other_top_n()
    long_min_score = _long_min_score()

    mid_candidates, mid_rrf_trace = _hybrid_seed_hits(
        question, query_vector, mids, mid_retrieval_content, pool_top_n
    )
    core_candidates, core_rrf_trace = _hybrid_seed_hits(
        question, query_vector, core, long_retrieval_text, pool_top_n
    )
    other_candidates, other_rrf_trace = _hybrid_seed_hits(
        question, query_vector, other, long_retrieval_text, pool_top_n
    )
    mid_metadata = _metadata_by_id(mid_candidates)
    core_metadata = _metadata_by_id(core_candidates)
    other_metadata = _metadata_by_id(other_candidates)

    reranker = rerank_fn or rerank_scored
    batch_size = int(_cfg("RERANK_BATCH_SIZE", 100))
    mid_hits = reranker(
        question,
        [record for record, _metadata in mid_candidates],
        mid_retrieval_content,
        mid_top_n,
        min_score=long_min_score,
        batch_size=batch_size,
    )
    core_hits = reranker(
        question,
        [record for record, _metadata in core_candidates],
        long_retrieval_text,
        core_top_n,
        min_score=long_min_score,
        batch_size=batch_size,
    )
    other_hits = reranker(
        question,
        [record for record, _metadata in other_candidates],
        long_retrieval_text,
        other_top_n,
        min_score=long_min_score,
        batch_size=batch_size,
    )

    mid_memories = _memory_rows(
        mid_hits, mid_metadata, pool="mid", text_of=mid_retrieval_content
    )
    core_memories = _memory_rows(
        core_hits, core_metadata, pool="core", text_of=long_retrieval_text
    )
    other_memories = _memory_rows(
        other_hits, other_metadata, pool="other", text_of=long_retrieval_text
    )
    memories = [*mid_memories, *core_memories, *other_memories]
    type_counts = Counter(str(record.get("type") or "mid") for record in memories)
    dimension = len(query_vector)
    return RetrievalResult(
        memories=memories,
        trace={
            "mode": MODE,
            "query": question,
            "conversation_scope": "one_conversation_both_owners",
            "candidate_policy": (
                "three_independent_full_embedding_positive_bm25_rrf90_pools"
            ),
            "graph_expansion": False,
            "relation_inputs": "not_read",
            "mid_stage": _stage(
                mids,
                mid_rrf_trace,
                mid_hits,
                mid_metadata,
                rerank_top_n=mid_top_n,
                min_score=long_min_score,
                embedding_dimension=dimension,
                embedding_source="mid_memories.json:inline",
            ),
            "core_stage": _stage(
                core,
                core_rrf_trace,
                core_hits,
                core_metadata,
                rerank_top_n=core_top_n,
                min_score=long_min_score,
                embedding_dimension=dimension,
                embedding_source="long_embeddings.json",
            ),
            "other_stage": _stage(
                other,
                other_rrf_trace,
                other_hits,
                other_metadata,
                rerank_top_n=other_top_n,
                min_score=long_min_score,
                embedding_dimension=dimension,
                embedding_source="long_embeddings.json",
            ),
            "final": {
                "mid_count": len(mid_memories),
                "core_count": len(core_memories),
                "other_count": len(other_memories),
                "long_count": len(core_memories) + len(other_memories),
                "count_by_type": dict(sorted(type_counts.items())),
                "count": len(memories),
                "selected": [record.get("id") for record in memories],
                "selection_policy": (
                    f"rrf_mid{pool_top_n}_core{pool_top_n}_other{pool_top_n}"
                    f"_then_rerank_mid{mid_top_n}_core{core_top_n}_other{other_top_n}"
                ),
                "long_min_rerank_score": long_min_score,
                "source_evidence_stage": "answer_only_after_retrieval",
            },
        },
    )


__all__ = [
    "MODE",
    "search_rrf90",
]
