"""
Azure OpenAI (GPT-5 deployment) helper for support agent performance scoring.
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

_CATEGORY_KEYS: Tuple[str, ...] = (
    "empathy",
    "clarity",
    "professionalism",
    "problem_resolution",
)

AGENT_SCORING_SCHEMA = json.dumps(
    {
        "overall_score": 8.5,
        "category_scores": {k: 8 for k in _CATEGORY_KEYS},
        "strengths": ["point1"],
        "improvements": ["point1"],
    },
    indent=0,
)

AGENT_SCORING_PROMPT = (
    "You are an expert call quality analyst.\n"
    "Evaluate the support agent's performance based on the transcript (and optional summary/intents).\n\n"
    "Return a single JSON object only, no markdown or code fences, with exactly:\n"
    '- "overall_score": number from 0 to 10 (float allowed, one decimal is fine)\n'
    '- "category_scores": object with integer 0-10 scores for keys: '
    "empathy, clarity, professionalism, problem_resolution\n"
    '- "strengths": array of short strings (what the agent did well)\n'
    '- "improvements": array of short strings (actionable coaching points)\n\n'
    "Rules:\n"
    "- Be objective and consistent; every score must be traceable to evidence in the text.\n"
    "- If agent lines are not clearly distinguishable, score conservatively and say so in improvements.\n"
    "- Keep lists concise (up to 5 items each).\n"
    f"- Schema shape: {AGENT_SCORING_SCHEMA}"
)

FALLBACK_AGENT_SCORE: Dict[str, Any] = {
    "overall_score": 0.0,
    "category_scores": {k: 0 for k in _CATEGORY_KEYS},
    "strengths": [],
    "improvements": [],
}

AGENT_SCORE_TEMPERATURE = 0.25


def _deployment_name() -> Optional[str]:
    name = os.getenv("AZURE_OPENAI_GPT5_DEPLOYMENT", "").strip()
    return name or None


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


def _normalize_str_list(value: Any, max_items: int = 8) -> List[str]:
    out: List[str] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        if len(out) >= max_items:
            break
    return out


def _normalize_payload(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "overall_score": float(FALLBACK_AGENT_SCORE["overall_score"]),
            "category_scores": dict(FALLBACK_AGENT_SCORE["category_scores"]),
            "strengths": [],
            "improvements": [],
        }

    overall = _clamp_float_10(raw.get("overall_score"))
    cats_in = raw.get("category_scores")
    category_scores: Dict[str, int] = {k: 0 for k in _CATEGORY_KEYS}
    if isinstance(cats_in, dict):
        for k in _CATEGORY_KEYS:
            if k in cats_in:
                category_scores[k] = _clamp_int_10(cats_in[k])

    return {
        "overall_score": overall,
        "category_scores": category_scores,
        "strengths": _normalize_str_list(raw.get("strengths")),
        "improvements": _normalize_str_list(raw.get("improvements")),
    }


def score_agent_performance_with_openai(
    transcript_text: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Call Azure OpenAI (AZURE_OPENAI_GPT5_DEPLOYMENT). On failure, returns FALLBACK_AGENT_SCORE.
    """
    deployment = _deployment_name()
    if not deployment:
        logger.warning(
            "agent_scoring: AZURE_OPENAI_GPT5_DEPLOYMENT is not set; skipping API call"
        )
        return {
            "overall_score": float(FALLBACK_AGENT_SCORE["overall_score"]),
            "category_scores": dict(FALLBACK_AGENT_SCORE["category_scores"]),
            "strengths": [],
            "improvements": [],
        }

    text = (transcript_text or "").strip()
    if not text:
        logger.warning("agent_scoring: empty transcript after strip")
        return {
            "overall_score": float(FALLBACK_AGENT_SCORE["overall_score"]),
            "category_scores": dict(FALLBACK_AGENT_SCORE["category_scores"]),
            "strengths": [],
            "improvements": [],
        }

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
        {"role": "system", "content": AGENT_SCORING_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "agent_score OpenAI call start deployment=%s transcript_chars=%d summary_items=%d intent_items=%d",
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
            temperature=AGENT_SCORE_TEMPERATURE,
            top_p=1,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - t0
        logger.info("agent_score OpenAI call success latency_s=%.3f", elapsed)

        content = completion.choices[0].message.content
        if not content:
            logger.error("agent_scoring: empty message content from model")
            return {
                "overall_score": float(FALLBACK_AGENT_SCORE["overall_score"]),
                "category_scores": dict(FALLBACK_AGENT_SCORE["category_scores"]),
                "strengths": [],
                "improvements": [],
            }

        parsed = parse_json_content(content)
        return _normalize_payload(parsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "agent_score OpenAI call failed after %.3fs: %s", elapsed, exc
        )
        return {
            "overall_score": float(FALLBACK_AGENT_SCORE["overall_score"]),
            "category_scores": dict(FALLBACK_AGENT_SCORE["category_scores"]),
            "strengths": [],
            "improvements": [],
        }
