"""
Azure OpenAI (GPT-5 deployment) helper for post-analysis next-action suggestions.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from metrics.azure_openai import create_chat_completion, get_azure_openai_client, parse_json_content
from metrics.unified_analysis import MAX_TRANSCRIPT_CHARS

logger = logging.getLogger(__name__)

NEXT_MOVE_SCHEMA_EXAMPLE = json.dumps(
    {"next_move": "string", "reasoning": "string"},
    indent=0,
)

NEXT_MOVE_PROMPT = (
    "You are an expert advisor for customer service and sales call handling. "
    "Given a call transcript and optional summary bullets and intent labels, "
    "recommend exactly one concrete next action for the agent or organization.\n\n"
    "Requirements:\n"
    "- Ground the recommendation only in the provided content; do not invent facts, "
    "customer details, or commitments.\n"
    "- The next action must be specific and executable (follow-up, escalation, "
    "documentation, callback, etc.).\n"
    "- Respond with a single JSON object only, no markdown or code fences.\n"
    "- Keys must be exactly: next_move (string), reasoning (string). "
    "reasoning should briefly justify the choice.\n\n"
    f"Schema: {NEXT_MOVE_SCHEMA_EXAMPLE}"
)

FALLBACK_MODEL_FAILURE: Dict[str, str] = {
    "next_move": "Unable to determine next action",
    "reasoning": "Model response failed",
}

NEXT_MOVE_TEMPERATURE = 0.25


def _deployment_name() -> Optional[str]:
    name = os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT", "").strip()
    return name or None


def _normalize_payload(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return dict(FALLBACK_MODEL_FAILURE)
    nm = raw.get("next_move")
    rs = raw.get("reasoning")
    if not isinstance(nm, str):
        nm = FALLBACK_MODEL_FAILURE["next_move"]
    if not isinstance(rs, str):
        rs = FALLBACK_MODEL_FAILURE["reasoning"]
    nm = nm.strip() or FALLBACK_MODEL_FAILURE["next_move"]
    rs = rs.strip() or FALLBACK_MODEL_FAILURE["reasoning"]
    return {"next_move": nm, "reasoning": rs}


def suggest_next_move_with_openai(
    transcript_text: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Call Azure OpenAI (deployment from AZURE_OPENAI_GPT5_DEPLOYMENT) and return
    {"next_move": str, "reasoning": str}. On any failure, returns FALLBACK_MODEL_FAILURE.
    """
    deployment = _deployment_name()
    if not deployment:
        logger.warning(
            "suggest_next_move: AZURE_OPENAI_GPT5_DEPLOYMENT is not set; skipping API call"
        )
        return dict(FALLBACK_MODEL_FAILURE)

    text = (transcript_text or "").strip()
    if not text:
        logger.warning("suggest_next_move: empty transcript after strip")
        return dict(FALLBACK_MODEL_FAILURE)

    trimmed = text[:MAX_TRANSCRIPT_CHARS]
    summ = summary if summary else []
    ints = intents if intents else []
    if not isinstance(summ, list):
        summ = []
    if not isinstance(ints, list):
        ints = []

    user_parts = [
        "=== TRANSCRIPT ===\n" + trimmed,
    ]
    if summ:
        user_parts.append(
            "=== SUMMARY (bullet points) ===\n"
            + "\n".join(f"- {s}" for s in summ if isinstance(s, str) and s.strip())
        )
    if ints:
        user_parts.append(
            "=== INTENTS ===\n"
            + ", ".join(str(i) for i in ints if i is not None)
        )
    user_content = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": NEXT_MOVE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "next_move OpenAI call start deployment=%s transcript_chars=%d summary_items=%d intent_items=%d",
        deployment,
        len(trimmed),
        len(summ),
        len(ints),
    )
    t0 = time.perf_counter()

    try:
        client = get_azure_openai_client()
        completion = create_chat_completion(
            client,
            model=deployment,
            messages=messages,
            temperature=NEXT_MOVE_TEMPERATURE,
            top_p=1,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - t0
        logger.info("next_move OpenAI call success latency_s=%.3f", elapsed)

        content = completion.choices[0].message.content
        if not content:
            logger.error("next_move: empty message content from model")
            return dict(FALLBACK_MODEL_FAILURE)

        parsed = parse_json_content(content)
        return _normalize_payload(parsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "next_move OpenAI call failed after %.3fs: %s", elapsed, exc
        )
        return dict(FALLBACK_MODEL_FAILURE)
