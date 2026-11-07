"""Chat-model construction with an explicit per-request timeout.

Without one, a hung OpenAI request waits for the SDK default (about 10 minutes) and the run
looks stuck. The model is built through the same provider-profile helper `create_deep_agent`
uses for model strings, so defaults such as the Responses API are kept.
"""
from __future__ import annotations

import os

from deepagents.profiles.provider.provider_profiles import apply_provider_profile
from langchain.chat_models import init_chat_model

DEFAULT_REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 1  # one retry on timeout/transient error; the SDK default is 2


def request_timeout_seconds() -> float:
    """OPENAI_REQUEST_TIMEOUT_SECONDS from env (default 90)."""
    raw = os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_REQUEST_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_REQUEST_TIMEOUT_SECONDS


def build_model(spec: str):
    """`spec` like "openai:gpt-6.1-terra" -> chat model with a request timeout and one retry."""
    kwargs = apply_provider_profile(spec, {"timeout": request_timeout_seconds(), "max_retries": MAX_RETRIES})
    return init_chat_model(spec, **kwargs)
