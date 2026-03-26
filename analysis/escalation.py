"""
Azure OpenAI (GPT-5 deployment) helper for escalation detection and action items.
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

ESCALATION_SCHEMA_EXAMPLE = json.dumps(
    {
        "escalation_required": True,
        "priority": "high",
        "reason": "short explanation",
        "action_items": ["task1", "task2"],
    },
    indent=0,
)

ESCALATION_PROMPT = (
    "You are an expert customer support QA system.\n"
    "Analyze the transcript (and optional summary and intents) and determine if escalation is required.\n\n"
    "Return a single JSON object only, no markdown or code fences, with exactly these keys:\n"
    '- "escalation_required": boolean\n'
    '- "priority": one of "low", "medium", "high"\n'
    '- "reason": short string explaining the assessment\n'
    '- "action_items": array of concise, practical follow-up task strings\n\n'
    "Rules:\n"
    "- Escalation is appropriate for: angry or distressed customer, repeated unresolved issue, "
    "refund/compensation demands, legal or compliance risk, threats, or explicit supervisor requests.\n"
    "- If escalation is not required, set escalation_required to false, priority to low, "
    "and action_items to useful routine follow-ups or [].\n"
    "- Ground assessments only in the provided text; do not invent facts.\n"
    f"- Schema example: {ESCALATION_SCHEMA_EXAMPLE}"
)

FALLBACK_ESCALATION: Dict[str, Any] = {
    "escalation_required": False,
    "priority": "low",
    "reason": "Unable to determine",
    "action_items": [],
}

ESCALATION_TEMPERATURE = 0.25
_VALID_PRIORITY = frozenset({"low", "medium", "high"})


def _deployment_name() -> Optional[str]:
    name = os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT", "").strip()
    return name or None


def _normalize_priority(value: Any) -> str:
    if isinstance(value, str) and value.lower() in _VALID_PRIORITY:
        return value.lower()
    return "low"


def _normalize_payload(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return dict(FALLBACK_ESCALATION)

    esc = raw.get("escalation_required")
    if isinstance(esc, str):
        esc = esc.lower() in ("true", "1", "yes")
    elif not isinstance(esc, bool):
        esc = bool(esc) if esc is not None else False

    reason = raw.get("reason")
    if not isinstance(reason, str):
        reason = FALLBACK_ESCALATION["reason"]
    reason = reason.strip() or FALLBACK_ESCALATION["reason"]

    items_raw = raw.get("action_items")
    action_items: List[str] = []
    if isinstance(items_raw, list):
        for x in items_raw:
            if isinstance(x, str) and x.strip():
                action_items.append(x.strip())
    elif isinstance(items_raw, str) and items_raw.strip():
        action_items.append(items_raw.strip())

    return {
        "escalation_required": esc,
        "priority": _normalize_priority(raw.get("priority")),
        "reason": reason,
        "action_items": action_items,
    }


def detect_escalation_with_openai(
    transcript_text: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Call Azure OpenAI (AZURE_OPENAI_GPT5_DEPLOYMENT). On failure, returns FALLBACK_ESCALATION.
    """
    deployment = _deployment_name()
    if not deployment:
        logger.warning(
            "detect_escalation: AZURE_OPENAI_GPT5_DEPLOYMENT is not set; skipping API call"
        )
        return dict(FALLBACK_ESCALATION)

    text = (transcript_text or "").strip()
    if not text:
        logger.warning("detect_escalation: empty transcript after strip")
        return dict(FALLBACK_ESCALATION)

    trimmed = text[:MAX_TRANSCRIPT_CHARS]
    summ = summary if summary else []
    ints = intents if intents else []
    if not isinstance(summ, list):
        summ = []
    if not isinstance(ints, list):
        ints = []

    user_parts = ["=== TRANSCRIPT ===\n" + trimmed]
    if summ:
        user_parts.append(
            "=== SUMMARY (bullet points) ===\n"
            + "\n".join(f"- {s}" for s in summ if isinstance(s, str) and s.strip())
        )
    if ints:
        user_parts.append(
            "=== INTENTS ===\n" + ", ".join(str(i) for i in ints if i is not None)
        )
    user_content = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": ESCALATION_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "escalation OpenAI call start deployment=%s transcript_chars=%d summary_items=%d intent_items=%d",
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
            temperature=ESCALATION_TEMPERATURE,
            top_p=1,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - t0
        logger.info("escalation OpenAI call success latency_s=%.3f", elapsed)

        content = completion.choices[0].message.content
        if not content:
            logger.error("detect_escalation: empty message content from model")
            return dict(FALLBACK_ESCALATION)

        parsed = parse_json_content(content)
        return _normalize_payload(parsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "escalation OpenAI call failed after %.3fs: %s", elapsed, exc
        )
        return dict(FALLBACK_ESCALATION)
