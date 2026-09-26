"""Explicit model-role routing for portable service configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

import config


@dataclass(frozen=True)
class Endpoint:
    base_url: str | None
    api_key: str | None


@dataclass(frozen=True)
class Provider:
    protocol: str
    endpoint: Endpoint
    extra_body: dict = field(default_factory=dict)


def _qa_endpoint(base_url: str | None, api_key: str | None) -> Endpoint:
    # A partial explicit pair remains partial so validation rejects it. It must
    # never borrow the missing half from another service's shared credential.
    if base_url or api_key:
        return Endpoint(base_url, api_key)
    return Endpoint(config.QA_BASE_URL, config.QA_API_KEY)


def validate_endpoint(endpoint: Endpoint, *, require_url: bool = True) -> None:
    """Reject incomplete service configuration before constructing an SDK client."""
    if not endpoint.api_key or not endpoint.api_key.strip():
        raise ValueError("An API key is required for the selected model role")
    url = endpoint.base_url
    if not url:
        if require_url:
            raise ValueError(
                "An explicit base URL is required for the selected model role"
            )
        return
    try:
        parsed = urlsplit(url)
        valid = parsed.scheme in {"https", "http"} and bool(parsed.hostname)
        valid = valid and parsed.username is None and parsed.password is None
        valid = valid and not parsed.query and not parsed.fragment
        # Accessing port validates malformed numeric or out-of-range ports.
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "Model base URL must be an absolute HTTP(S) URL without embedded credentials"
        )


EXTRACTION = Provider(
    "openai",
    Endpoint(config.EXTRACTION_BASE_URL, config.EXTRACTION_API_KEY),
    {"enable_thinking": config.EXTRACTION_ENABLE_THINKING},
)
EMBEDDING = Provider(
    "openai_embedding",
    Endpoint(config.PRISM_BASE_URL, config.PRISM_API_KEY),
)
RERANK = Provider(
    "dashscope_rerank",
    Endpoint(config.RERANK_BASE_URL, config.RERANK_API_KEY),
)
_QA_BODY = {"enable_thinking": True} if config.QA_ENABLE_THINKING else {}
QA = Provider("openai", Endpoint(config.QA_BASE_URL, config.QA_API_KEY), dict(_QA_BODY))
ANSWER = Provider(
    "openai",
    _qa_endpoint(config.ANSWER_BASE_URL, config.ANSWER_API_KEY),
    dict(_QA_BODY),
)
JUDGE = Provider(
    "openai", _qa_endpoint(config.JUDGE_BASE_URL, config.JUDGE_API_KEY), dict(_QA_BODY)
)
_ROLES = {
    "extraction": (config.EXTRACTION_MODEL, EXTRACTION),
    "embedding": (config.PRISM_MODEL, EMBEDDING),
    "rerank": (config.RERANK_MODEL, RERANK),
    "answer": (config.QA_ANSWER_MODEL, ANSWER),
    "judge": (config.EVAL_JUDGE_MODEL, JUDGE),
}


def provider_for(model: str, *, role: str | None = None) -> Provider:
    """Resolve a configured role, rejecting ambiguous model names or missing routes.

    Explicit roles let answer and judge share a model name while using different
    endpoints. Calls that only name a model remain supported when unambiguous.
    """
    if role is not None:
        if role not in _ROLES:
            raise ValueError("Unknown model role")
        expected_model, selected = _ROLES[role]
        if model != expected_model:
            raise ValueError("Requested model does not match the configured role")
    else:
        matches = [provider for name, provider in _ROLES.values() if name == model]
        if not matches:
            raise ValueError("No provider configured for the requested model")
        selected = matches[0]
        if any(provider != selected for provider in matches[1:]):
            raise ValueError(
                "Model name has multiple configured routes; specify its role"
            )
    validate_endpoint(
        selected.endpoint, require_url=selected.protocol != "dashscope_rerank"
    )
    return selected
