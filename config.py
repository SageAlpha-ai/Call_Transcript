"""
Central configuration from environment variables. Fails fast if required keys are missing.
"""
from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

REQUIRED_ENV_NAMES: List[str] = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_GPT5_DEPLOYMENT",
    "AZURE_SPEECH_API_KEY",
    "AZURE_SPEECH_API_REGION",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER_NAME",
]


def apply_openai_deployment_alias() -> None:
    legacy = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if legacy and not (os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "").strip():
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = legacy.strip()


def get_allowed_origins() -> List[str]:
    raw = (os.getenv("ALLOWED_ORIGINS") or "").strip()
    if not raw:
        return ["http://localhost:3000"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def environment_ready() -> bool:
    """True if all required env vars are set (ignores ALLOW_INCOMPLETE_ENV)."""
    apply_openai_deployment_alias()
    for name in REQUIRED_ENV_NAMES:
        if not (os.getenv(name) or "").strip():
            return False
    return True


def get_app_port() -> int:
    return int(os.getenv("PORT", "8000"))


def get_max_request_body_bytes() -> int:
    return int(os.getenv("MAX_REQUEST_BODY_BYTES", str(512 * 1024)))


def get_max_transcript_chars() -> int:
    return int(os.getenv("MAX_TRANSCRIPT_INPUT_CHARS", "200000"))


def get_rate_limit_per_minute() -> int:
    return int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


def validate_required_environment() -> None:
    apply_openai_deployment_alias()
    if os.getenv("ALLOW_INCOMPLETE_ENV", "").lower() in ("1", "true", "yes"):
        logger.warning(
            "ALLOW_INCOMPLETE_ENV is set — skipping strict environment validation (not for production)"
        )
        return
    missing = [name for name in REQUIRED_ENV_NAMES if not (os.getenv(name) or "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
