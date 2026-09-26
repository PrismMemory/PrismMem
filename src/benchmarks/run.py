"""Stage-by-stage benchmark runner. `prepare` and `summarize` never call models.

Example: python -m benchmarks.run locomo all --input /path/to/locomo10.json
Use --backend smoke ONLY to check plumbing: smoke outputs never report benchmark scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

from benchmarks.adapters import BENCHMARKS, get_adapter
from benchmarks.artifacts import (
    checked_record,
    digest,
    file_digest,
    implementation_digest,
    load_record,
    read_json,
    record_path,
    save_record,
    write_json,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
STAGES = ("prepare", "build", "retrieve", "answer", "score", "summarize")
PUBLIC_QUESTION_FIELDS = {
    "id",
    "conversation_id",
    "question",
    "question_date",
    "choices",
    "context",
    "retrieval_query",
}
ENVIRONMENT_KEYS = {
    "PRISM_BM25_K1", "PRISM_BM25_B", "PRISM_BM25_LEMMATIZATION",
    "PRISM_BM25_AUTO_DOWNLOAD", "PRISM_RRF_K", "PRISM_RRF_CANDIDATE_TOP_N",
    "PRISM_MID_RERANK_TOP_N", "PRISM_CORE_RERANK_TOP_N",
    "PRISM_OTHER_RERANK_TOP_N", "PRISM_LONG_RERANK_MIN_SCORE",
    "PRISM_RERANK_BATCH_SIZE", "PRISM_EXTRACTION_ENABLE_THINKING",
    "PRISM_QA_ENABLE_THINKING",
}


def public_question(question: dict) -> dict:
    """Explicit boundary: gold, gold category, evidence and metadata cannot reach a model."""
    return {
        key: copy.deepcopy(value)
        for key, value in question.items()
        if key in PUBLIC_QUESTION_FIELDS
    }


def load_profile(name: str, path: Path | None = None) -> dict:
    profile = read_json(path or PACKAGE / "configs" / f"{name}.json")
    if not isinstance(profile, dict):
        raise ValueError("A benchmark config must be a JSON object")
    allowed = {
        "benchmark",
        "configuration_kind",
        "models",
        "environment",
        "prepare_options",
        "answer",
        "judge",
        "description",
        "build",
        "dataset_sha256",
    }
    if set(profile) - allowed:
        raise ValueError("Unsupported config fields; keep credentials in the local .env")
    if profile.get("benchmark") != name:
        raise ValueError("Config benchmark does not match the requested benchmark")
    for field in ("environment", "prepare_options", "models", "answer", "judge", "build"):
        if not isinstance(profile.get(field, {}), dict):
            raise ValueError(f"Config {field} must be a JSON object")
    for role in ("answer", "judge"):
        settings = profile.get(role, {})
        if set(settings) - {"temperature", "max_tokens", "projection"}:
            raise ValueError(f"Unsupported {role} decoding settings")
        tokens = settings.get("max_tokens", 2048)
        temperature = settings.get("temperature")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"{role} max_tokens must be a positive integer")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError(f"{role} temperature must be finite and between 0 and 2")
    unknown = set(profile.get("environment", {})) - ENVIRONMENT_KEYS
    if unknown:
        raise ValueError(f"Unsupported config environment keys: {', '.join(sorted(unknown))}")
    if set(profile.get("models", {})) != {"extraction", "embedding", "rerank", "answer", "judge"}:
        raise ValueError("Config must name extraction, embedding, rerank, answer and judge models")
    if any(
        not isinstance(value, str) or not value.strip()
        for role, value in profile["models"].items()
        if not (name == "personamem" and role == "judge" and value is None)
    ):
        raise ValueError("Every configured model must have a non-empty name")
    expected_projection = {
        "locomo": "locomo_embedding",
        "personamem": "personamem_synthetic_chronological",
        "beam": "beam_mid_dialogue_state",
    }.get(name)
    projection = profile.get("answer", {}).get("projection")
    if projection is not None and projection != expected_projection:
        raise ValueError("Unsupported answer projection for this benchmark")
    build = profile.get("build", {})
    allowed_build = {"memory_types", "mid_prompt", "participant_mode", "mid_workers", "long_workers"}
    if set(build) - allowed_build:
        raise ValueError("Unsupported memory construction settings")
    types = build.get("memory_types", ["core", "episodic", "knowledge"])
    if (
        not isinstance(types, list) or not types
        or any(not isinstance(kind, str) for kind in types)
        or len(set(types)) != len(types)
        or set(types) - {"core", "episodic", "knowledge"}
    ):
        raise ValueError("build.memory_types must name unique supported types")
    if build.get("mid_prompt", "mid_extraction") not in {"mid_extraction", "mid_extraction_beam"}:
        raise ValueError("Unsupported Mid extraction prompt")
    if build.get("participant_mode", "legacy_literal") not in {"legacy_literal", "explicit"}:
        raise ValueError("Unsupported participant attribution mode")
    for field in ("mid_workers", "long_workers"):
        value = build.get(field, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"build.{field} must be a positive integer")
    return profile


def validate_prepared(dataset: dict) -> None:
    conversations, questions = dataset.get("conversations"), dataset.get("questions")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("The adapter produced no conversation histories")
    if not isinstance(questions, list) or not questions:
        raise ValueError("The adapter produced no evaluation questions")
    ids = [sample.get("sample_id") for sample in conversations]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Conversation sample_id values must be non-empty and unique")
    question_ids = []
    for question in questions:
        if not isinstance(question.get("id"), str) or not question["id"]:
            raise ValueError("Each question needs a non-empty string id")
        if question.get("conversation_id") not in ids:
            raise ValueError("A question refers to a missing conversation")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError("Question text must be non-empty")
        question_ids.append(question["id"])
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question IDs must be unique")
    forbidden = {
        "answer",
        "gold",
        "gold_answer",
        "correct_choice",
        "qa",
        "questions",
        "answer_session_ids",
        "has_answer",
        "rubric",
    }
    for sample in conversations:
        if forbidden.intersection(sample) or forbidden.intersection(sample.get("conversation", {})):
            raise ValueError("Evaluation labels must not enter the memory corpus")
        for key, turns in sample["conversation"].items():
            if key.startswith("session_") and isinstance(turns, list):
                if any(forbidden.intersection(turn) for turn in turns):
                    raise ValueError("Evaluation labels must not enter conversation turns")


def prepare_run(args: argparse.Namespace) -> dict:
    if not args.input:
        raise ValueError("prepare/all requires --input; see README.md for each dataset")
    profile = load_profile(args.benchmark, args.config)
    options = dict(profile.get("prepare_options", {}))
    for key in ("limit_questions", "limit_conversations"):
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value
    profile["prepare_options"] = options
    dataset = get_adapter(args.benchmark).prepare(args.input, options=options)
    validate_prepared(dataset)
    backend = args.backend or "live"
    is_synthetic = bool(
        dataset.get("metadata", {}).get("synthetic_smoke", False)
        or dataset.get("metadata", {}).get("dataset_kind") == "synthetic_smoke"
    )
    expected_source = profile.get("dataset_sha256")
    if (
        expected_source
        and not is_synthetic
        and dataset.get("metadata", {}).get("source", {}).get("sha256") != expected_source
    ):
        raise ValueError(
            "Input does not match the official dataset pinned by this profile; "
            "use the pinned dataset or a separately named custom profile"
        )
    if backend == "smoke" and not is_synthetic:
        raise ValueError("The smoke backend only accepts explicitly synthetic fixtures")
    manifest = {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "backend": backend,
        "synthetic_smoke": is_synthetic or backend == "smoke",
        "subset_requested": any(
            options.get(key) is not None for key in ("limit_questions", "limit_conversations")
        ),
        "config_sha256": digest(profile),
        "implementation_sha256": implementation_digest(SOURCE_ROOT),
        "corpus_sha256": digest(dataset["conversations"]),
        "questions_sha256": digest(dataset["questions"]),
        "metadata_sha256": digest(dataset.get("metadata", {})),
        "conversation_count": len(dataset["conversations"]),
        "question_count": len(dataset["questions"]),
    }
    output = args.output
    existing = output / "manifest.json"
    if existing.exists():
        if read_json(existing) != manifest:
            raise ValueError("Dataset/config/code/backend changed; choose a new --output directory")
        load_run(args)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty and has no matching run manifest")
    write_json(output / "prepared" / "corpus.json", dataset["conversations"])
    write_json(output / "prepared" / "questions.json", dataset["questions"])
    write_json(output / "prepared" / "metadata.json", dataset.get("metadata", {}))
    write_json(output / "config.json", profile)
    write_json(existing, manifest)
    print(
        f"prepare: {manifest['conversation_count']} histories, {manifest['question_count']} questions"
    )
    return manifest


def load_run(args: argparse.Namespace) -> tuple[dict, dict, list, list]:
    output = args.output
    manifest = read_json(output / "manifest.json")
    profile = read_json(output / "config.json")
    if manifest.get("benchmark") != args.benchmark:
        raise ValueError("Output directory belongs to another benchmark")
    if args.backend and args.backend != manifest["backend"]:
        raise ValueError("Cannot mix smoke and live artifacts; choose a new output directory")
    if args.config:
        candidate = load_profile(args.benchmark, args.config)
        candidate["prepare_options"] = {
            **profile.get("prepare_options", {}),
            **candidate.get("prepare_options", {}),
        }
        for key in ("limit_questions", "limit_conversations"):
            if getattr(args, key, None) is not None:
                candidate["prepare_options"][key] = getattr(args, key)
        if digest(candidate) != digest(profile):
            raise ValueError("Config differs from the prepared run; choose a new output directory")
    if digest(profile) != manifest["config_sha256"]:
        raise ValueError("Stored config changed after prepare")
    if implementation_digest(SOURCE_ROOT) != manifest["implementation_sha256"]:
        raise ValueError("Code or prompts changed after prepare; use a new output directory")
    corpus = read_json(output / "prepared" / "corpus.json")
    questions = read_json(output / "prepared" / "questions.json")
    metadata = read_json(output / "prepared" / "metadata.json")
    for name, value in (("corpus", corpus), ("questions", questions), ("metadata", metadata)):
        if digest(value) != manifest[f"{name}_sha256"]:
            raise ValueError(f"Prepared {name} changed after prepare")
    return manifest, profile, corpus, questions


def memory_digest(output: Path, cid: str) -> str:
    scope = output / "memory" / hashlib.sha256(cid.encode()).hexdigest()[:20]
    files = {
        path.relative_to(scope).as_posix(): file_digest(path)
        for path in sorted(scope.rglob("*.json"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(scope).parts)
    }
    if not files:
        raise ValueError(f"No memory artifacts found for conversation {cid}; run build first")
    return digest(files)


def make_backend(manifest: dict, profile: dict, output: Path):
    from benchmarks.runtime import LiveBackend, SmokeBackend

    backend_class = SmokeBackend if manifest["backend"] == "smoke" else LiveBackend
    return backend_class(profile, output)


def require_record(output: Path, stage: str, record_id: str) -> dict:
    path = record_path(output, stage, record_id)
    if not path.exists():
        raise ValueError(f"Missing {stage} record for {record_id}; run {stage} first")
    record = checked_record(path)
    if record.get("id") != record_id:
        raise ValueError(f"Invalid {stage} record id")
    return record


def checked_memory(
    output: Path, question: dict, manifest: dict, corpus_by_id: dict, cache: dict
) -> str:
    cid = question["conversation_id"]
    if cid not in cache:
        fingerprint = digest({"sample": corpus_by_id[cid], "config": manifest["config_sha256"]})
        built = load_record(output, "build", cid, fingerprint)
        if built is None:
            raise ValueError(f"Missing build record for {cid}; run build first")
        current = memory_digest(output, cid)
        if built["memory_sha256"] != current:
            raise ValueError("Built memories changed; use a new output directory")
        cache[cid] = current
    return cache[cid]


def checked_retrieval(
    output: Path, question: dict, manifest: dict, corpus_by_id: dict, cache: dict
) -> dict:
    memory = checked_memory(output, question, manifest, corpus_by_id, cache)
    fingerprint = digest(
        {
            "question": public_question(question),
            "memory": memory,
            "config": manifest["config_sha256"],
        }
    )
    record = load_record(output, "retrieve", question["id"], fingerprint)
    if record is None:
        raise ValueError(f"Missing retrieve record for {question['id']}; run retrieve first")
    return record


def checked_answer(
    output: Path, question: dict, manifest: dict, corpus_by_id: dict, cache: dict
) -> dict:
    from benchmarks.evaluation import answer_messages

    retrieval = checked_retrieval(output, question, manifest, corpus_by_id, cache)
    messages = answer_messages(
        manifest["benchmark"], public_question(question), retrieval["memories"]
    )
    fingerprint = digest(
        {"retrieval": retrieval, "messages": messages, "config": manifest["config_sha256"]}
    )
    record = load_record(output, "answer", question["id"], fingerprint)
    if record is None:
        raise ValueError(f"Missing answer record for {question['id']}; run answer first")
    return record


def run_stage(args: argparse.Namespace, stage: str) -> dict | None:
    manifest, profile, corpus, questions = load_run(args)
    output = args.output
    if stage in {"summarize", "check"}:
        return summarize(args, manifest, questions, require_complete=not args.allow_partial)
    backend = None

    def runtime():
        nonlocal backend
        if backend is None:
            backend = make_backend(manifest, profile, output)
        return backend

    if stage == "build":
        for sample in corpus:
            cid = sample["sample_id"]
            fingerprint = digest({"sample": sample, "config": manifest["config_sha256"]})
            previous = load_record(output, "build", cid, fingerprint)
            if previous:
                if previous["memory_sha256"] != memory_digest(output, cid):
                    raise ValueError("Built memories changed; use a new output directory")
                continue
            result = runtime().build(sample, output / "prepared" / "corpus.json")
            save_record(
                output,
                "build",
                cid,
                fingerprint,
                {"stats": result, "memory_sha256": memory_digest(output, cid)},
            )
        print(f"build: {len(corpus)} histories complete")
        return None

    from benchmarks.evaluation import answer_messages, parse_answer, score_prediction

    validated_memories: dict[str, str] = {}
    corpus_by_id = {sample["sample_id"]: sample for sample in corpus}
    for index, question in enumerate(questions, 1):
        qid = question["id"]
        if stage == "retrieve":
            cid = question["conversation_id"]
            checked_memory(output, question, manifest, corpus_by_id, validated_memories)
            visible = public_question(question)
            fingerprint = digest(
                {
                    "question": visible,
                    "memory": validated_memories[cid],
                    "config": manifest["config_sha256"],
                }
            )
            if not load_record(output, stage, qid, fingerprint):
                result = runtime().retrieve(visible)
                save_record(output, stage, qid, fingerprint, result)
        elif stage == "answer":
            retrieval = checked_retrieval(
                output, question, manifest, corpus_by_id, validated_memories
            )
            messages = answer_messages(
                args.benchmark, public_question(question), retrieval["memories"]
            )
            fingerprint = digest(
                {"retrieval": retrieval, "messages": messages, "config": manifest["config_sha256"]}
            )
            if not load_record(output, stage, qid, fingerprint):
                response = runtime().complete("answer", messages)
                prediction = (
                    response["response"]
                    if manifest["backend"] == "smoke"
                    else parse_answer(args.benchmark, response["response"])
                )
                save_record(
                    output,
                    stage,
                    qid,
                    fingerprint,
                    {
                        **response,
                        "prediction": prediction,
                        "model": profile["models"]["answer"],
                        "messages_sha256": digest(messages),
                    },
                )
        elif stage == "score":
            answer = checked_answer(output, question, manifest, corpus_by_id, validated_memories)
            fingerprint = digest(
                {"answer": answer, "question": question, "config": manifest["config_sha256"]}
            )
            if not load_record(output, stage, qid, fingerprint):
                usage = []

                def judge(messages: list[dict], usage_log=usage) -> str:
                    response = runtime().complete("judge", messages)
                    usage_log.append(response.get("usage", {}))
                    return response["response"]

                result = (
                    {
                        "score": None,
                        "metric": "smoke_execution_only",
                        "details": {"not_a_benchmark_result": True},
                    }
                    if manifest["backend"] == "smoke"
                    else score_prediction(
                        args.benchmark,
                        question,
                        answer["response"]
                        if args.benchmark in {"locomo", "personamem"}
                        else answer["prediction"],
                        judge=judge,
                    )
                )
                score = result.get("score")
                if score is None and manifest["backend"] != "smoke":
                    raise ValueError(
                        "Live scorer returned no score; this question cannot count as completed"
                    )
                if score is not None and (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError("Scorer returned an invalid score; expected finite [0, 1]")
                save_record(output, stage, qid, fingerprint, {**result, "judge_usage": usage})
        else:
            raise ValueError(f"Unknown stage: {stage}")
        if index % 50 == 0 or index == len(questions):
            print(f"{stage}: {index}/{len(questions)} questions complete", flush=True)
    return None


def summarize(
    args: argparse.Namespace,
    manifest: dict,
    questions: list[dict],
    *,
    require_complete: bool = True,
) -> dict:
    output = args.output
    completed = []
    missing = []
    categories: dict[str, list[float]] = defaultdict(list)
    expected_categories: dict[str, int] = defaultdict(int)
    corpus_by_id = {
        sample["sample_id"]: sample for sample in read_json(output / "prepared" / "corpus.json")
    }
    validated_memories: dict[str, str] = {}
    expected_files = {record_path(output, "score", question["id"]).name for question in questions}
    unexpected = [
        path.name
        for path in (output / "score").glob("*.json")
        if not path.name.startswith(".") and path.name not in expected_files
    ]
    if unexpected:
        raise ValueError("Unknown question artifacts found in score directory")
    for question in questions:
        qid = question["id"]
        category = str(question.get("category", "unknown"))
        expected_categories[category] += 1
        path = record_path(output, "score", qid)
        if not path.exists():
            missing.append(qid)
            continue
        answer = checked_answer(output, question, manifest, corpus_by_id, validated_memories)
        expected_digest = digest(
            {"answer": answer, "question": question, "config": manifest["config_sha256"]}
        )
        score = load_record(output, "score", qid, expected_digest)
        if score["score"] is None and manifest["backend"] != "smoke":
            raise ValueError("A live score is missing; cannot count it as completed")
        completed.append({"id": qid, "category": category, **score})
        if score["score"] is not None:
            value = score["score"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Invalid score artifact")
            categories[category].append(value)
    if missing and require_complete:
        raise ValueError(
            f"Incomplete evaluation: {len(missing)}/{len(questions)} scores missing; "
            "resume score, or summarize --allow-partial"
        )
    synthetic = manifest["synthetic_smoke"]
    metadata = read_json(output / "prepared" / "metadata.json")
    values = [row["score"] for row in completed if row["score"] is not None]
    metrics = sorted({row["metric"] for row in completed})
    if len(metrics) > 1:
        raise ValueError("Mixed scoring protocols in one run")
    summary = {
        "benchmark": args.benchmark,
        "backend": manifest["backend"],
        "status": (
            "synthetic_smoke"
            if synthetic
            else "partial"
            if missing
            else "complete_subset"
            if manifest.get("subset_requested")
            else "complete"
        ),
        "subset_requested": manifest.get("subset_requested", False),
        "source_counts": metadata.get("source_counts", {}),
        "filters": metadata.get("filters", {}),
        "expected_questions": len(questions),
        "completed_questions": len(completed),
        "missing_question_ids": missing,
        "metric": metrics[0] if metrics else None,
        "mean_score": None if synthetic or not values else sum(values) / len(values),
        "score_percent": None if synthetic or not values else 100 * sum(values) / len(values),
        "per_category": {
            key: {
                "expected": count,
                "scored": len(categories[key]),
                "mean_score": (
                    None
                    if synthetic or not categories[key]
                    else sum(categories[key]) / len(categories[key])
                ),
            }
            for key, count in sorted(expected_categories.items())
        },
        "config_sha256": manifest["config_sha256"],
        "implementation_sha256": manifest["implementation_sha256"],
        "note": (
            "Synthetic plumbing check; no benchmark accuracy was measured."
            if synthetic
            else "Measured with the saved benchmark configuration."
        ),
    }
    if args.benchmark == "longmemeval":
        summary["category_macro_mean"] = (
            None
            if synthetic or not values
            else sum(sum(rows) / len(rows) for rows in categories.values() if rows)
            / sum(bool(rows) for rows in categories.values())
        )
        for label, is_abstention in (("abstention", True), ("answerable", False)):
            selected = [
                row["score"]
                for row in completed
                if str(row["id"]).endswith("_abs") == is_abstention and row["score"] is not None
            ]
            summary[label] = {
                "expected": sum(str(q["id"]).endswith("_abs") == is_abstention for q in questions),
                "scored": len(selected),
                "mean_score": None if synthetic or not selected else sum(selected) / len(selected),
            }
    write_json(output / "summary.json", summary)
    write_json(output / "per_question_scores.json", completed)
    score_text = (
        "not measured" if summary["score_percent"] is None else f"{summary['score_percent']:.4f}%"
    )
    lines = [
        f"# {args.benchmark} run",
        "",
        summary["note"],
        "",
        f"- Status: {summary['status']}",
        f"- Completed: {len(completed)} / {len(questions)}",
        f"- Mean score: {score_text}",
        f"- Metric: {summary['metric']}",
        "",
        "See summary.json for category counts and the configuration/source fingerprints.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "benchmark",
                    "status",
                    "completed_questions",
                    "expected_questions",
                    "score_percent",
                )
            },
            ensure_ascii=False,
        )
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("stage", choices=(*STAGES, "all", "check"))
    parser.add_argument("--input", type=Path, help="benchmark source file or directory")
    parser.add_argument("--output", type=Path, help="run directory; default runs/<benchmark>")
    parser.add_argument("--config", type=Path, help="JSON profile; defaults to benchmarks/configs")
    parser.add_argument(
        "--backend",
        choices=("live", "smoke"),
        default=None,
        help="live calls configured models; smoke requires a synthetic fixture",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="smoke/pilot subset only; never shortens selected histories",
    )
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="summarize an incomplete run, explicitly labeled partial",
    )
    args = parser.parse_args(argv)
    args.output = (args.output or Path.cwd() / "runs" / args.benchmark).expanduser().resolve()
    try:
        if args.stage not in {"prepare", "all"} and any(
            value is not None
            for value in (args.input, args.limit_questions, args.limit_conversations)
        ):
            raise ValueError(
                "--input and subset limits apply only to prepare/all; other stages use the saved dataset"
            )
        if args.stage in {"prepare", "all"}:
            prepare_run(args)
        if args.stage == "all":
            for stage in STAGES[1:]:
                run_stage(args, stage)
        elif args.stage != "prepare":
            run_stage(args, args.stage)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
