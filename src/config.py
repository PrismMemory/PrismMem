"""Environment-only configuration for the embedding benchmark pipelines."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv is convenient, not required for library imports.
    pass

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not float("-inf") < value < float("inf"):
        raise ValueError(f"{name} must be finite")
    return value


def _str(name: str, default: str | None = None) -> str | None:
    """Treat unset, empty, and whitespace-only template entries identically."""
    return os.getenv(name, "").strip() or default


DATA_DIR = _str("PRISM_MEMORY_DIR", str(PACKAGE_ROOT / "data"))
EVAL_DIR = _str("PRISM_EVAL_DIR", str(PACKAGE_ROOT / "outputs" / "eval"))

# Model roles and their environment overrides.
EXTRACTION_MODEL = _str("PRISM_EXTRACTION_MODEL", "glm-5.2")
PRISM_MODEL = _str("PRISM_MODEL", "text-embedding-v4")
RERANK_MODEL = _str("PRISM_RERANK_MODEL", "qwen3-rerank")
QA_ANSWER_MODEL = _str("PRISM_ANSWER_MODEL", "gpt-4o-mini")
EVAL_JUDGE_MODEL = _str("PRISM_JUDGE_MODEL", "gpt-4.1-mini")

# Every OpenAI-compatible route requires an explicit URL and key. Independent
# answer/judge pairs override the shared QA pair as a whole, never field by field.
EXTRACTION_BASE_URL = _str("PRISM_EXTRACTION_BASE_URL")
EXTRACTION_API_KEY = _str("PRISM_EXTRACTION_API_KEY")
PRISM_BASE_URL = _str("PRISM_BASE_URL")
PRISM_API_KEY = _str("PRISM_API_KEY")
RERANK_API_KEY = _str("PRISM_RERANK_API_KEY")
# A blank native rerank URL deliberately selects the DashScope SDK route.
RERANK_BASE_URL = _str("PRISM_RERANK_BASE_URL")
QA_BASE_URL = _str("PRISM_QA_BASE_URL")
QA_API_KEY = _str("PRISM_QA_API_KEY")
ANSWER_BASE_URL = _str("PRISM_ANSWER_BASE_URL")
ANSWER_API_KEY = _str("PRISM_ANSWER_API_KEY")
JUDGE_BASE_URL = _str("PRISM_JUDGE_BASE_URL")
JUDGE_API_KEY = _str("PRISM_JUDGE_API_KEY")

EXTRACTION_ENABLE_THINKING = _flag("PRISM_EXTRACTION_ENABLE_THINKING", False)
QA_ENABLE_THINKING = _flag("PRISM_QA_ENABLE_THINKING", False)

# Remote-call policy.
LLM_MAX_TOKENS = _int("PRISM_LLM_MAX_TOKENS", 20000)
LLM_TIMEOUT_SECONDS = _int("PRISM_LLM_TIMEOUT_SECONDS", 400)
LLM_MAX_RETRIES = _int("PRISM_LLM_MAX_RETRIES", 5)
LLM_RETRY_BACKOFF_SECONDS = _float("PRISM_LLM_RETRY_BACKOFF_SECONDS", 2.0)
# Endpoint protection for long high-concurrency runs. Neither value changes what the
# models see; they only pace requests so throttled batches finish instead of failing.
LLM_MIN_REQUEST_INTERVAL_SECONDS = _float(
    "PRISM_LLM_MIN_REQUEST_INTERVAL_SECONDS", 0.0
)
LLM_RATE_LIMIT_COOLDOWN_SECONDS = _float(
    "PRISM_LLM_RATE_LIMIT_COOLDOWN_SECONDS", 10.0
)
PRISM_BATCH_SIZE = _int("PRISM_BATCH_SIZE", 10)
PRISM_TIMEOUT_SECONDS = _int("PRISM_TIMEOUT_SECONDS", 30)
RERANK_MAX_RETRIES = _int("PRISM_RERANK_MAX_RETRIES", 3)
RERANK_RETRY_BACKOFF_SECONDS = _float("PRISM_RERANK_RETRY_BACKOFF_SECONDS", 1.0)
RERANK_TIMEOUT_SECONDS = _float("PRISM_RERANK_TIMEOUT_SECONDS", 300.0)
# Connection pool for the dedicated rerank route only; sized to the worker count so a
# high-concurrency run reuses TLS connections instead of discarding them.
RERANK_HTTP_POOL_MAXSIZE = _int("PRISM_RERANK_HTTP_POOL_MAXSIZE", 32)

# Retrieval defaults with environment overrides for each parameter.
BM25_K1 = _float("PRISM_BM25_K1", 1.5)
BM25_B = _float("PRISM_BM25_B", 0.75)
BM25_LEMMATIZATION_ENABLED = _flag("PRISM_BM25_LEMMATIZATION", True)
BM25_LEMMATIZATION_AUTO_DOWNLOAD = _flag("PRISM_BM25_AUTO_DOWNLOAD", True)
BM25_LEMMATIZATION_MODEL = "en_core_web_sm"
RRF_K = _int("PRISM_RRF_K", 60)
RRF_CANDIDATE_TOP_N = _int("PRISM_RRF_CANDIDATE_TOP_N", 70)
MID_RERANK_TOP_N = _int("PRISM_MID_RERANK_TOP_N", 10)
CORE_RERANK_TOP_N = _int("PRISM_CORE_RERANK_TOP_N", 60)
OTHER_RERANK_TOP_N = _int("PRISM_OTHER_RERANK_TOP_N", 50)
LONG_RERANK_MIN_SCORE = _float("PRISM_LONG_RERANK_MIN_SCORE", 0.5)
RERANK_BATCH_SIZE = _int("PRISM_RERANK_BATCH_SIZE", 100)

# Evaluation defaults.
EVAL_QA_WORKERS = _int("PRISM_QA_WORKERS", 20)
EVAL_SCORE_WORKERS = _int("PRISM_SCORE_WORKERS", 20)
EVAL_JUDGE_MAX_RETRIES = _int("PRISM_JUDGE_MAX_RETRIES", 3)
