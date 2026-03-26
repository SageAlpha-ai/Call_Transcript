"""
Single Azure OpenAI call: full call analysis (sentiment, intents, summary, next move,
escalation, follow-up email, agent score).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from metrics.azure_openai import create_chat_completion, get_azure_openai_client, parse_json_content
from metrics.unified_analysis import MAX_TRANSCRIPT_CHARS

logger = logging.getLogger(__name__)

_AGENT_CATS: Tuple[str, ...] = (
    "empathy",
    "clarity",
    "professionalism",
    "problem_resolution",
)
_VALID_PRIORITY = frozenset({"low", "medium", "high"})

UNIFIED_FULL_SCHEMA: Dict[str, Any] = {
    "sentiment_score": 0.0,
    "utterance_sentiments": [{"index": 0, "sentiment_score": 0.0}],
    "intents": ["intent"],
    "summary": ["point"],
    "next_move": "action",
    "reasoning": "brief justification for next_move",
    "escalation": {
        "escalation_required": False,
        "priority": "low",
        "reason": "",
        "action_items": ["task"],
    },
    "followup_email": {"subject": "", "email_body": ""},
    "agent_score": {
        "overall_score": 0.0,
        "category_scores": {k: 0 for k in _AGENT_CATS},
        "strengths": ["text"],
        "improvements": ["text"],
    },
}

UNIFIED_FULL_PROMPT = (
    "You are an expert call analysis AI.\n"
    "Given the full transcript and numbered utterances, return ONE JSON object with ALL of the following keys.\n"
    "Do not omit keys. Use only information supported by the transcript; do not invent facts.\n\n"
    "Keys:\n"
    '- sentiment_score: float from -1 (very negative) to 1 (very positive) for the whole call\n'
    '- utterance_sentiments: array of {{"index": int (0-based), "sentiment_score": float (-1 to 1)}} '
    "for every utterance index\n"
    '- intents: 1–5 snake_case intent strings\n'
    '- summary: up to 5 short bullet strings\n'
    '- next_move: one concrete recommended next action\n'
    '- reasoning: one short string explaining next_move\n'
    '- escalation: {escalation_required (bool), priority (low|medium|high), reason (string), '
    "action_items (array of short strings)}\n"
    '- followup_email: {subject (string), email_body (string)} professional follow-up to the customer\n'
    "- agent_score: {overall_score (float 0-10), category_scores (empathy, clarity, professionalism, "
    "problem_resolution as int 0-10), strengths (array of strings), improvements (array of strings)}\n\n"
    "Rules:\n"
    "- Return ONLY valid JSON, no markdown.\n"
    "- Align utterance_sentiments indices with the numbered utterance list.\n"
    "- Keep text concise.\n"
    f"- Schema shape reference: {json.dumps(UNIFIED_FULL_SCHEMA)}"
)


def _deployment_name() -> Optional[str]:
    name = os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT", "").strip()
    return name or None


def _clamp_sentiment(value: Any) -> float:
    try:
        v = float(value)
        return max(-1.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.0


def _clamp_float_10(value: Any) -> float:
    try:
        v = float(value)
        return max(0.0, min(10.0, v))
    except (TypeError, ValueError):
        return 0.0


def _clamp_int_10(value: Any) -> int:
    try:
        v = int(round(float(value)))
        return max(0, min(10, v))
    except (TypeError, ValueError):
        return 0


def _normalize_priority(value: Any) -> str:
    if isinstance(value, str) and value.lower() in _VALID_PRIORITY:
        return value.lower()
    return "low"


def _str_list(value: Any, max_items: int = 8) -> List[str]:
    out: List[str] = []
    if not isinstance(value, list):
        return out
    for x in value:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        if len(out) >= max_items:
            break
    return out


def _merge_utterance_sentiments(
    raw_list: Any,
    utterances: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    n = len(utterances)
    utt_map: Dict[int, float] = {}
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, dict) and "index" in item:
                try:
                    idx = int(item["index"])
                    utt_map[idx] = _clamp_sentiment(item.get("sentiment_score"))
                except (TypeError, ValueError):
                    continue
    filled: List[Dict[str, Any]] = []
    for i in range(n):
        text = ""
        if i < len(utterances):
            text = str(utterances[i].get("text", "") or "")
        filled.append({
            "text": text,
            "sentiment_score": utt_map.get(i, 0.0),
        })
    return filled


def _normalize_escalation(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "escalation_required": False,
            "priority": "low",
            "reason": "Unable to determine",
            "action_items": [],
        }
    esc = raw.get("escalation_required")
    if isinstance(esc, str):
        esc = esc.lower() in ("true", "1", "yes")
    elif not isinstance(esc, bool):
        esc = bool(esc) if esc is not None else False
    reason = raw.get("reason")
    if not isinstance(reason, str):
        reason = "Unable to determine"
    reason = reason.strip() or "Unable to determine"
    items = _str_list(raw.get("action_items"), max_items=12)
    return {
        "escalation_required": esc,
        "priority": _normalize_priority(raw.get("priority")),
        "reason": reason,
        "action_items": items,
    }


def _normalize_followup(raw: Any) -> Dict[str, str]:
    fb = {"subject": "Follow-up on your recent call", "email_body": "Thank you for your time. We will follow up shortly."}
    if not isinstance(raw, dict):
        return dict(fb)
    subj = raw.get("subject")
    body = raw.get("email_body")
    if raw.get("body") and not body:
        body = raw.get("body")
    if not isinstance(subj, str):
        subj = fb["subject"]
    if not isinstance(body, str):
        body = fb["email_body"]
    subj = subj.strip() or fb["subject"]
    body = body.strip() or fb["email_body"]
    return {"subject": subj, "email_body": body}


def _normalize_agent_score(raw: Any) -> Dict[str, Any]:
    base = {
        "overall_score": 0.0,
        "category_scores": {k: 0 for k in _AGENT_CATS},
        "strengths": [],
        "improvements": [],
    }
    if not isinstance(raw, dict):
        return base
    overall = _clamp_float_10(raw.get("overall_score"))
    cats_in = raw.get("category_scores")
    cats = {k: 0 for k in _AGENT_CATS}
    if isinstance(cats_in, dict):
        for k in _AGENT_CATS:
            if k in cats_in:
                cats[k] = _clamp_int_10(cats_in[k])
    return {
        "overall_score": overall,
        "category_scores": cats,
        "strengths": _str_list(raw.get("strengths")),
        "improvements": _str_list(raw.get("improvements")),
    }


def _fallback_response(utterances: List[Dict[str, Any]]) -> Dict[str, Any]:
    utt_block = _merge_utterance_sentiments([], utterances)
    return {
        "sentiment_score": 0.0,
        "utterance_sentiments": utt_block,
        "intents": [],
        "summary": [],
        "next_move": "Unable to determine next action",
        "reasoning": "Analysis unavailable",
        "escalation": _normalize_escalation(None),
        "followup_email": _normalize_followup(None),
        "agent_score": _normalize_agent_score(None),
    }


def _normalize_full_parsed(
    parsed: Dict[str, Any],
    utterances: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sentiment = _clamp_sentiment(parsed.get("sentiment_score"))
    intents = _str_list(parsed.get("intents"), max_items=6)
    summary = _str_list(parsed.get("summary"), max_items=8)
    utt_sents = _merge_utterance_sentiments(parsed.get("utterance_sentiments"), utterances)
    nm = parsed.get("next_move")
    if not isinstance(nm, str):
        nm = ""
    nm = nm.strip() or "Unable to determine next action"
    reason = parsed.get("reasoning") or parsed.get("next_move_reasoning")
    if not isinstance(reason, str):
        reason = ""
    reason = reason.strip()
    return {
        "sentiment_score": sentiment,
        "utterance_sentiments": utt_sents,
        "intents": intents,
        "summary": summary,
        "next_move": nm,
        "reasoning": reason or "—",
        "escalation": _normalize_escalation(parsed.get("escalation")),
        "followup_email": _normalize_followup(parsed.get("followup_email")),
        "agent_score": _normalize_agent_score(parsed.get("agent_score")),
    }


def _default_utterances(transcript_text: str) -> List[Dict[str, Any]]:
    parts = [s.strip() for s in transcript_text.split(".") if s.strip()]
    if parts:
        return [{"text": p} for p in parts]
    t = transcript_text.strip()
    return [{"text": t}] if t else [{"text": ""}]


def analyze_all_with_openai(
    transcript_text: str,
    utterances: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Single GPT call (AZURE_OPENAI_GPT5_DEPLOYMENT). Returns normalized dict with all sections.
    """
    text = (transcript_text or "").strip()
    if not text:
        logger.warning("unified_full: empty transcript")
        u = utterances if utterances else [{"text": ""}]
        return _fallback_response(u)

    utts = utterances if utterances else _default_utterances(text)
    if not utts:
        utts = [{"text": text}]

    deployment = _deployment_name()
    if not deployment:
        logger.warning("unified_full: AZURE_OPENAI_GPT5_DEPLOYMENT not set")
        return _fallback_response(utts)

    trimmed = text[:MAX_TRANSCRIPT_CHARS]
    utterance_lines = "\n".join(
        f"[{i}] {u.get('text', '')}" for i, u in enumerate(utts)
    )
    utterance_block = utterance_lines[:MAX_TRANSCRIPT_CHARS]
    user_content = (
        f"=== FULL TRANSCRIPT ===\n{trimmed}\n\n"
        f"=== UTTERANCES (numbered) ===\n{utterance_block}"
    )

    messages = [
        {"role": "system", "content": UNIFIED_FULL_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "unified_full OpenAI call start deployment=%s utterances=%d transcript_chars=%d",
        deployment,
        len(utts),
        len(trimmed),
    )
    t0 = time.perf_counter()

    try:
        client = get_azure_openai_client()
        completion = create_chat_completion(
            client,
            model=deployment,
            messages=messages,
            temperature=0,
            top_p=1,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - t0
        usage = getattr(completion, "usage", None)
        if usage is not None:
            logger.info(
                "unified_full OpenAI success latency_s=%.3f prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                elapsed,
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "completion_tokens", None),
                getattr(usage, "total_tokens", None),
            )
        else:
            logger.info("unified_full OpenAI success latency_s=%.3f (no usage metadata)", elapsed)

        content = completion.choices[0].message.content
        if not content:
            logger.error("unified_full: empty model content")
            return _fallback_response(utts)

        parsed_raw = parse_json_content(content)
        if not isinstance(parsed_raw, dict):
            return _fallback_response(utts)
        return _normalize_full_parsed(parsed_raw, utts)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception("unified_full OpenAI failed after %.3fs: %s", elapsed, exc)
        return _fallback_response(utts)
