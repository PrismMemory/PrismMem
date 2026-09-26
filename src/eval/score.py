

from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("score")

# Make src/ importable (config / llm are top-level there) and this dir (checkpoint).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import jsonio
from checkpoint import Checkpoint
from llm import complete, prompts


_SCORE_MANIFEST = "score_manifest.json"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_fingerprint(path: str) -> dict:
    target = Path(path)
    return {"exists": target.is_file(),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None}


def _score_manifest_payload(eval_dir: str) -> dict:
    answer_files = {name: _file_fingerprint(os.path.join(eval_dir, name))
                    for name in sorted(os.listdir(eval_dir))
                    if name.startswith("conv-") and name.endswith(".json")}
    code = [Path(__file__), Path(_HERE) / "checkpoint.py", Path(_SRC) / "jsonio.py", Path(_SRC) / "config.py"]
    code.extend((Path(_SRC) / "llm").rglob("*.py"))
    endpoint = getattr(config, "JUDGE_BASE_URL", None) or getattr(config, "QA_BASE_URL", None)
    return {"schema_version": 2, "judge_model": config.EVAL_JUDGE_MODEL,
            "judge_settings": {"max_tokens": config.LLM_MAX_TOKENS, "temperature": None,
                               "enable_thinking": getattr(config, "QA_ENABLE_THINKING", False),
                               "max_label_retries": config.EVAL_JUDGE_MAX_RETRIES,
                               "endpoint_sha256": _digest(endpoint)},
            "judge_prompt": _file_fingerprint(os.path.join(_SRC, "llm/prompts/qa_judge.txt")),
            "implementation_sha256": {str(path.relative_to(_SRC)): _file_fingerprint(str(path)) for path in sorted(code)},
            "run_manifest": jsonio.read_json(os.path.join(eval_dir, "run_manifest.json"), default=None),
            "answer_files": answer_files}


def _ensure_score_manifest(eval_dir: str, *, restart: bool, sample: str | None) -> dict:
    """Validate without writing, so rejected resumes leave every old file intact."""
    expected = _score_manifest_payload(eval_dir)
    existing = jsonio.read_json(os.path.join(eval_dir, _SCORE_MANIFEST), default=None)
    has_old_state = any(os.path.exists(os.path.join(eval_dir, name))
                        for name in ("scores.json", "progress_score.json"))
    if existing is None and has_old_state and not (restart and sample is None):
        raise ValueError("Unlabelled prior scores/checkpoint; use a new directory or full --restart")
    if existing is not None and existing != expected and not (restart and sample is None):
        raise ValueError("Answer content, run manifest, judge configuration/template or code changed; use full --restart")
    return expected


def _scored_digest(record: dict) -> str:
    return _digest({key: value for key, value in record.items() if key != "artifact_sha256"})

# LoCoMo category labels (5 = adversarial, skipped during scoring).
CATEGORY_LABELS = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}
ADVERSARIAL_CATEGORY = 5


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace/punctuation (matches the reference tokenizer)."""
    text = str(text).lower()
    for ch in ".,!?":
        text = text.replace(ch, " ")
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    """Token-set F1 between prediction and reference (0 if either is empty)."""
    pred = set(_tokenize(prediction))
    ref = set(_tokenize(reference))
    if not pred or not ref:
        return 0.0
    common = len(pred & ref)
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, reference: str) -> float:
    """BLEU-1: clipped unigram precision with brevity penalty (0 if no overlap).

    A dependency-free stand-in for nltk's sentence_bleu with weights (1,0,0,0):
    unigram matches are clipped by their reference count, then scaled by the standard
    brevity penalty. Returns 0.0 when there is no lexical overlap.
    """
    pred = prediction.lower().split()
    ref = reference.lower().split()
    if not pred or not ref:
        return 0.0
    ref_counts = Counter(ref)
    clipped = sum(min(c, ref_counts[t]) for t, c in Counter(pred).items())
    if clipped == 0:
        return 0.0
    precision = clipped / len(pred)
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity * precision


def llm_judge(question: str, gold_answer: str, generated_answer: str) -> int:
    """Ask the model to grade the answer CORRECT/WRONG; return 1 for CORRECT else 0."""
    prompt = prompts.render(
        "qa_judge",
        QUESTION=question,
        GOLD_ANSWER=gold_answer,
        GENERATED_ANSWER=generated_answer,
    )
    for _ in range(config.EVAL_JUDGE_MAX_RETRIES):
        raw = complete(
            [{"role": "user", "content": prompt}],
            model=config.EVAL_JUDGE_MODEL,
            role="judge",
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=None,
        )
        label = (raw or "").strip().strip('"').upper()
        if label in ("CORRECT", "WRONG"):
            return 1 if label == "CORRECT" else 0
    raise RuntimeError(
        "judge returned no valid CORRECT/WRONG label after retries; "
        "the question was not scored"
    )


def score_record(record: dict) -> dict:
    """Score one QA record; return {question, answer, response, category, *_score, time}."""
    gold = str(record.get("answer") if record.get("answer") is not None else "")
    response = str(record.get("response") if record.get("response") is not None else "")
    question = record.get("question", "")
    scored = {
        "question": question,
        "answer": gold,
        "response": response,
        "category": record.get("category"),
        "bleu_score": bleu1(response, gold),
        "f1_score": token_f1(response, gold),
        "llm_score": llm_judge(question, gold, response),
        "response_time": record.get("response_time", ""),
    }
    if type(record.get("qa_index")) is int:
        scored["qa_index"] = record["qa_index"]
    return scored


def _score_key(record: dict, fallback_index: int) -> str:
    """Stable score identity; question text is not unique in LoCoMo."""
    qa_index = record.get("qa_index")
    if type(qa_index) is int:
        return f"qa:{qa_index}"
    return f"legacy-row:{fallback_index}"


def _score_unit(record: dict, fallback_index: int) -> int:
    """Checkpoint against source QA index when available, else legacy result position."""
    qa_index = record.get("qa_index")
    return qa_index if type(qa_index) is int else fallback_index


def _ordered_scores(records: dict[str, dict]) -> list[dict]:
    return sorted(
        records.values(),
        key=lambda record: (
            0 if type(record.get("qa_index")) is int else 1,
            record.get("qa_index") if type(record.get("qa_index")) is int else 0,
        ),
    )


def _print_progress(label: str, done: int, total: int) -> None:
    width = 30
    ratio = done / total if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r{label} [{bar}] {done}/{total}",
        end="" if done < total else "\n",
        file=sys.stderr,
        flush=True,
    )


def _category_int(value) -> int | None:
    """Coerce a category value (int or str) to int, or None if not parseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _aggregate(scored: list[dict]) -> dict:
    """Build overall and per-category means (bleu / f1 / llm) with counts."""
    def summarize(items: list[dict]) -> dict:
        return {
            "bleu": _mean([i["bleu_score"] for i in items]),
            "f1": _mean([i["f1_score"] for i in items]),
            "llm": _mean([i["llm_score"] for i in items]),
            "count": len(items),
        }

    by_category: dict[str, dict] = {}
    for cat in sorted({_category_int(i["category"]) for i in scored} - {None}):
        items = [i for i in scored if _category_int(i["category"]) == cat]
        summary = summarize(items)
        summary["type"] = CATEGORY_LABELS.get(cat, str(cat))
        by_category[str(cat)] = summary

    return {"overall": summarize(scored), "by_category": by_category}


def _result_files(sample: str | None, eval_dir: str) -> list[str]:
    """Result files under EVAL_DIR to score (excludes scores/progress bookkeeping)."""
    if sample is not None:
        path = os.path.join(eval_dir, f"{sample}.json")
        if not os.path.exists(path):
            raise ValueError(f"result file not found: {path} (run run.py first)")
        return [path]
    files = [
        os.path.join(eval_dir, f)
        for f in sorted(os.listdir(eval_dir))
        if f.startswith("conv-") and f.endswith(".json")
    ]
    if not files:
        raise ValueError(f"no result files in {eval_dir} (run run.py first)")
    return files


def _write_scores(all_results: dict[str, dict[str, dict]], scores_path: str) -> dict:
    """Recompute aggregate from the accumulated scored records and write scores.json."""
    flat = [rec for recs in all_results.values() for rec in recs.values()]
    report = {
        "samples": list(all_results.keys()),
        **_aggregate(flat),
        "results": {s: _ordered_scores(recs) for s, recs in all_results.items()},
    }
    jsonio.atomic_write_json(scores_path, report)
    return report


def score_eval(
    *,
    sample: str | None = None,
    workers: int = config.EVAL_SCORE_WORKERS,
    restart: bool = False,
    eval_dir: str | None = None,
) -> dict:
    """Score the selected result file(s) incrementally; write EVAL_SCORES_PATH.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    eval_dir = os.path.abspath(eval_dir or config.EVAL_DIR)
    scores_path = os.path.join(eval_dir, "scores.json")
    files = _result_files(sample, eval_dir)
    expected_manifest = _ensure_score_manifest(eval_dir, restart=restart, sample=sample)
    logger.info("scoring %d result file(s), workers=%d", len(files), workers)

    existing = (
        {} if restart and sample is None
        else (jsonio.read_json(scores_path, default={}) or {})
    )
    all_results: dict[str, dict[str, dict]] = {
        sample_id: {
            _score_key(record, position): record
            for position, record in enumerate(records)
            if isinstance(record, dict)
        }
        for sample_id, records in (existing.get("results", {}) or {}).items()
    }

    source_records = {}
    for path in files:
        records = jsonio.read_json(path, default=None)
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise ValueError("Answer output must be an array of records")
        if any(_category_int(row.get("category")) != ADVERSARIAL_CATEGORY
               and (row.get("retrieval_only") is True or not str(row.get("response") or "").strip())
               for row in records):
            raise ValueError("Answer output contains unanswered retrieval-only records; run the answer stage first")
        units = [_score_unit(row, index) for index, row in enumerate(records)]
        if len(units) != len(set(units)):
            raise ValueError("Answer output contains duplicate QA identities")
        source_records[path] = records
    if not (restart and sample is None):
        for cid, rows in all_results.items():
            if restart and cid == sample:
                continue
            if any(row.get("artifact_sha256") != _scored_digest(row) for row in rows.values()):
                raise ValueError("Prior score content changed or lacks integrity metadata; use full --restart")
    if restart and sample is None:
        for name in ("scores.json", "progress_score.json", _SCORE_MANIFEST):
            Path(eval_dir, name).unlink(missing_ok=True)
    checkpoint = Checkpoint(os.path.join(eval_dir, "progress_score.json"), "score")
    if restart and sample is not None:
        all_results.pop(sample, None)
        _write_scores(all_results, scores_path)
        checkpoint.clear(sample)
    jsonio.atomic_write_json(os.path.join(eval_dir, _SCORE_MANIFEST), expected_manifest)

    for path in files:
        sample_id = os.path.splitext(os.path.basename(path))[0]
        records = source_records[path]
        if restart:
            checkpoint.clear(sample_id)
            all_results[sample_id] = {}
        scored_by_id = all_results.setdefault(sample_id, {})
        done = checkpoint.done_units(sample_id)
        scorable = [
            (_score_unit(record, position), position, record)
            for position, record in enumerate(records)
            if _category_int(record.get("category")) != ADVERSARIAL_CATEGORY
        ]
        todo = [item for item in scorable
                if item[0] not in done
                or scored_by_id.get(_score_key(item[2], item[1]), {}).get("source_record_sha256") != _digest(item[2])]
        sources_by_unit = {unit: record for unit, _, record in scorable}
        logger.info("[%s] %d scorable of %d records: %d done, %d to score",
                    sample_id, len(scorable), len(records), len(scorable) - len(todo), len(todo))

        def persist(unit: int, position: int, scored: dict) -> None:
            scored["source_record_sha256"] = _digest(sources_by_unit[unit])
            scored["artifact_sha256"] = _scored_digest(scored)
            scored_by_id[_score_key(scored, position)] = scored
            _write_scores(all_results, scores_path)
            checkpoint.mark(sample_id, unit)

        completed = len(scorable) - len(todo)
        _print_progress(f"[{sample_id}] scoring", completed, len(scorable))

        if workers <= 1:
            for unit, position, record in todo:
                persist(unit, position, score_record(record))
                completed += 1
                _print_progress(f"[{sample_id}] scoring", completed, len(scorable))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(score_record, record): (unit, position)
                    for unit, position, record in todo
                }
                for fut in as_completed(futures):
                    unit, position = futures[fut]
                    persist(unit, position, fut.result())
                    completed += 1
                    _print_progress(f"[{sample_id}] scoring", completed, len(scorable))

    report = _write_scores(all_results, scores_path)
    logger.info("wrote scores for %d question(s) -> %s",
                report["overall"]["count"], scores_path)
    return report


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Score LoCoMo QA answers.")
    parser.add_argument(
        "--sample", default=None,
        help="sample_id to score, e.g. conv-26; if omitted, all result files",
    )
    parser.add_argument(
        "--workers", type=int, default=config.EVAL_SCORE_WORKERS,
        help="concurrent judge requests",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="rescore questions already scored (ignore progress)",
    )
    parser.add_argument(
        "--eval-dir", default=config.EVAL_DIR,
        help="answer directory containing the run to score",
    )
    args = parser.parse_args()

    report = score_eval(
        sample=args.sample, workers=args.workers, restart=args.restart,
        eval_dir=args.eval_dir,
    )
    overall = report["overall"]
    logger.info(
        "DONE: %d question(s) | bleu=%.4f f1=%.4f llm=%.4f",
        overall["count"], overall["bleu"], overall["f1"], overall["llm"],
    )


if __name__ == "__main__":
    main()
