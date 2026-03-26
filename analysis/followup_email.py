"""
Azure OpenAI (GPT-5 deployment) helper for follow-up email generation.
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

FOLLOWUP_SCHEMA_EXAMPLE = json.dumps(
    {"subject": "email subject", "email_body": "full email text"},
    indent=0,
)

FOLLOWUP_EMAIL_PROMPT = (
    "You are a professional customer support assistant.\n"
    "Write a concise and polite follow-up email based on the call.\n\n"
    "Return a single JSON object only, no markdown or code fences, with exactly:\n"
    '- "subject": string — clear email subject line\n'
    '- "email_body": string — full email including greeting and sign-off\n\n'
    "Rules:\n"
    "- Be polite, professional, and clear.\n"
    "- Summarize key discussion points from the provided content only.\n"
    "- Include agreed or suggested next steps when applicable; align with any suggested "
    "next_move if provided.\n"
    "- Do not invent customer names, account details, dates, or promises not supported by the text.\n"
    "- Keep the email concise.\n"
    f"- Schema example: {FOLLOWUP_SCHEMA_EXAMPLE}"
)

FALLBACK_FOLLOWUP_EMAIL: Dict[str, str] = {
    "subject": "Follow-up on your recent call",
    "email_body": "Thank you for your time. We will follow up shortly.",
}

FOLLOWUP_TEMPERATURE = 0.35


def _deployment_name() -> Optional[str]:
    name = os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT", "").strip()
    return name or None


def _normalize_payload(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return dict(FALLBACK_FOLLOWUP_EMAIL)
    subj = raw.get("subject")
    body = raw.get("email_body")
    if raw.get("body") and not body:
        body = raw.get("body")
    if not isinstance(subj, str):
        subj = FALLBACK_FOLLOWUP_EMAIL["subject"]
    if not isinstance(body, str):
        body = FALLBACK_FOLLOWUP_EMAIL["email_body"]
    subj = subj.strip() or FALLBACK_FOLLOWUP_EMAIL["subject"]
    body = body.strip() or FALLBACK_FOLLOWUP_EMAIL["email_body"]
    return {"subject": subj, "email_body": body}


def generate_followup_email_with_openai(
    transcript_text: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
    next_move: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Azure OpenAI (AZURE_OPENAI_GPT5_DEPLOYMENT). On failure, returns FALLBACK_FOLLOWUP_EMAIL.
    """
    deployment = _deployment_name()
    if not deployment:
        logger.warning(
            "followup_email: AZURE_OPENAI_GPT5_DEPLOYMENT is not set; skipping API call"
        )
        return dict(FALLBACK_FOLLOWUP_EMAIL)

    text = (transcript_text or "").strip()
    if not text:
        logger.warning("followup_email: empty transcript after strip")
        return dict(FALLBACK_FOLLOWUP_EMAIL)

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
    nm = (next_move or "").strip()
    if nm:
        user_parts.append("=== SUGGESTED NEXT MOVE (incorporate if relevant) ===\n" + nm)
    user_content = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": FOLLOWUP_EMAIL_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "followup_email OpenAI call start deployment=%s transcript_chars=%d summary_items=%d "
        "intent_items=%d has_next_move=%s",
        deployment,
        len(trimmed),
        len(summ),
        len(ints),
        bool(nm),
    )
    t0 = time.perf_counter()

    try:
        client = get_azure_openai_client()
        completion = create_chat_completion(
            client,
            model=deployment,
            messages=messages,
            temperature=FOLLOWUP_TEMPERATURE,
            top_p=1,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - t0
        logger.info("followup_email OpenAI call success latency_s=%.3f", elapsed)

        content = completion.choices[0].message.content
        if not content:
            logger.error("followup_email: empty message content from model")
            return dict(FALLBACK_FOLLOWUP_EMAIL)

        parsed = parse_json_content(content)
        return _normalize_payload(parsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "followup_email OpenAI call failed after %.3fs: %s", elapsed, exc
        )
        return dict(FALLBACK_FOLLOWUP_EMAIL)
