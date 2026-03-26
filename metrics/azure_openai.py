from __future__ import annotations

import json
import logging
import os
from typing import Any

from .retry_utils import retry_sync


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_azure_openai_client() -> Any:
    from openai import AzureOpenAI

    endpoint = _get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_key = _get_required_env("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "(not set)")

    logging.info("Azure OpenAI config: endpoint=%s deployment=%s api_key=%s",
                 endpoint, deployment, "set" if api_key else "MISSING")

    if not endpoint.startswith("https://"):
        raise RuntimeError(f"AZURE_OPENAI_ENDPOINT must start with https:// — got: {endpoint}")
    if ".openai.azure.com" not in endpoint and ".cognitiveservices.azure.com" not in endpoint:
        raise RuntimeError(
            f"AZURE_OPENAI_ENDPOINT does not look like a valid Azure OpenAI URL: {endpoint}. "
            "Expected format: https://<resource-name>.openai.azure.com"
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-12-01-preview",
    )


def get_azure_openai_deployment() -> str:
    return _get_required_env("AZURE_OPENAI_DEPLOYMENT_NAME")


def create_chat_completion(client: Any, **kwargs: Any) -> Any:
    return retry_sync(
        lambda: client.chat.completions.create(**kwargs),
        operation="Azure OpenAI chat.completions.create",
    )


def parse_json_content(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception as exc:
        raise RuntimeError(f"Azure OpenAI response was not valid JSON: {content}") from exc
