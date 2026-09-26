

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
for _path in (_SRC, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import config
import embedding
import jsonio
from retrieval.projection import long_retrieval_text, text_sha256

logger = logging.getLogger("prism.backfill")
_LONG_FILES = (
    "long_core.json",
    "long_episodic.json",
    "long_knowledge.json",
)


def _read_array(path: Path) -> list[dict]:
    payload = jsonio.read_json(str(path), default=None)
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return [row for row in payload if isinstance(row, dict)]


def build_long_embeddings(
    *,
    memory_dir: str,
    output_path: str | None = None,
    batch_size: int = 100,
    workers: int = 1,
    restart: bool = False,
) -> dict:
    if batch_size < 1 or workers < 1:
        raise ValueError("batch_size and workers must be positive")
    root = Path(memory_dir).resolve()
    output = Path(output_path).resolve() if output_path else root / "long_embeddings.json"
    records = [
        row
        for filename in _LONG_FILES
        for row in _read_array(root / filename)
        if long_retrieval_text(row)
    ]
    ids = [str(row.get("id") or "").strip() for row in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("long memories require unique non-empty ids")

    existing = [] if restart else _read_array(output, optional=True)
    by_id = {
        str(row.get("id") or ""): row
        for row in existing
        if str(row.get("id") or "").strip()
    }
    pending = []
    cached = 0
    for record in records:
        record_id = str(record["id"])
        text = long_retrieval_text(record)
        digest = text_sha256(text)
        row = by_id.get(record_id) or {}
        valid = (
            row.get("embedding_model") == config.PRISM_MODEL
            and row.get("text_sha256") == digest
            and isinstance(row.get("embedding"), list)
            and bool(row.get("embedding"))
        )
        if valid:
            cached += 1
        else:
            pending.append((record, text, digest))

    chunks = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]

    def encode(chunk: list[tuple[dict, str, str]]):
        vectors = embedding.embed_texts([text for _record, text, _digest in chunk])
        if len(vectors) != len(chunk):
            raise RuntimeError("embedding backend returned the wrong vector count")
        return chunk, vectors

    embedded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(encode, chunk) for chunk in chunks]
        for future in as_completed(futures):
            chunk, vectors = future.result()
            for (record, _text, digest), vector in zip(chunk, vectors):
                by_id[str(record["id"])] = {
                    "id": str(record["id"]),
                    "mid_id": record.get("mid_id"),
                    "type": record.get("type"),
                    "embedding_model": config.PRISM_MODEL,
                    "text_sha256": digest,
                    "embedding": vector,
                }
            embedded += len(chunk)
            jsonio.atomic_write_json(str(output), list(by_id.values()))
            logger.info("embedded %d/%d pending memories", embedded, len(pending))
    if not output.exists():
        jsonio.atomic_write_json(str(output), list(by_id.values()))
    return {
        "selected": len(records),
        "cached": cached,
        "embedded": embedded,
        "sidecar_rows": len(by_id),
        "output": str(output),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    logger.info(
        "DONE: %s",
        build_long_embeddings(
            memory_dir=args.memory_dir,
            output_path=args.output,
            batch_size=args.batch_size,
            workers=args.workers,
            restart=args.restart,
        ),
    )


if __name__ == "__main__":
    main()

