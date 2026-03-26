import json
import logging
import time
from typing import Any, Dict, List

from .azure_openai import (
    create_chat_completion,
    get_azure_openai_client,
    get_azure_openai_deployment,
    parse_json_content,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants (example shapes embedded in prompts + validation defaults)
# ---------------------------------------------------------------------------

TRANSCRIPT_ANALYSIS_SCHEMA = {
    "sentiment_score": 0.0,
    "utterance_sentiments": [{"index": 0, "sentiment_score": 0.0}],
    "intents": ["intent_example"],
    "summary": ["point 1", "point 2"],
}

UNIFIED_ANALYSIS_SCHEMA = TRANSCRIPT_ANALYSIS_SCHEMA

SUMMARY_SCHEMA = {
    "summary": ["concise bullet point"],
}

INTENT_SCHEMA = {
    "intents": ["snake_case_intent"],
}

SENTIMENT_SCHEMA = {
    "sentiment_score": 0.0,
    "utterance_sentiments": [{"index": 0, "sentiment_score": 0.0}],
}

NEXT_ACTION_SCHEMA = {
    "next_move": "recommended action",
    "reasoning": "brief justification",
}

ESCALATION_SCHEMA = {
    "escalation_required": False,
    "priority": "low",
    "reason": "",
    "action_items": ["task"],
}

EMAIL_SCHEMA = {
    "subject": "email subject",
    "email_body": "full email body",
}

AGENT_PERFORMANCE_SCHEMA = {
    "overall_score": 0.0,
    "category_scores": {
        "empathy": 0,
        "clarity": 0,
        "professionalism": 0,
        "problem_resolution": 0,
    },
    "strengths": ["point"],
    "improvements": ["point"],
}

# ---------------------------------------------------------------------------
# Task-specific system prompts
# ---------------------------------------------------------------------------

TRANSCRIPT_SYSTEM_PROMPT = (
    "You are an expert call-transcript analyst.\n\n"
    "Task:\n"
    "Analyze the provided transcript and numbered utterances. Produce a complete structured assessment "
    "covering overall and per-utterance sentiment, caller intents, and a factual summary.\n\n"
    "Return a JSON object with:\n\n"
    "1. sentiment_score — float from -1 (very negative) to 1 (very positive) for the entire call\n"
    "2. utterance_sentiments — array with one object per utterance; each object has index (int, 0-based, "
    "matching the utterance number) and sentiment_score (float, -1 to 1)\n"
    "3. intents — array of 1 to 5 concise snake_case labels describing the caller's business purpose "
    "(e.g. billing_dispute, refund_request), not vague words like \"help\" alone\n"
    "4. summary — array of up to 5 concise strings: key issues, decisions, commitments, and outcomes "
    "explicitly stated in the call\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON (no markdown, no explanations)\n"
    "* Do not hallucinate or add information not present in transcript\n"
    "* Be concise and precise\n"
    "* If the transcript is long, prioritize the most material facts\n"
    f"* Follow this schema exactly: {json.dumps(TRANSCRIPT_ANALYSIS_SCHEMA)}"
)

SUMMARY_SYSTEM_PROMPT = (
    "You are an expert call analyst.\n\n"
    "Task:\n"
    "Analyze the conversation transcript and produce a concise, high-quality summary.\n\n"
    "Return a JSON object with:\n\n"
    "* summary: a list of up to 5 bullet points\n\n"
    "Guidelines:\n\n"
    "* Focus on key issues, customer concerns, and outcomes\n"
    "* Highlight any problems, requests, or resolutions\n"
    "* Include important decisions or next steps\n"
    "* Avoid generic statements like 'the customer called'\n"
    "* Do not repeat information\n"
    "* Keep each bullet short but meaningful\n"
    "* Prefer business-relevant insights over narration\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON\n"
    "* No markdown or explanations\n"
    f"* Follow this schema exactly: {json.dumps(SUMMARY_SCHEMA)}"
)

INTENT_SYSTEM_PROMPT = (
    "You are an expert customer-intent taxonomist.\n\n"
    "Task:\n"
    "Infer the caller's business intent from the transcript and numbered utterances. Labels must "
    "encode operational meaning (product, policy, or outcome), not generic chat descriptors.\n\n"
    "Return a JSON object with:\n\n"
    "1. intents — array of 1 to 5 snake_case strings (e.g. order_status_inquiry, account_closure_request); "
    "avoid empty or overly generic labels\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON (no markdown, no explanations)\n"
    "* Do not hallucinate or add information not present in transcript\n"
    "* Be concise and precise\n"
    f"* Follow this schema exactly: {json.dumps(INTENT_SCHEMA)}"
)

SENTIMENT_SYSTEM_PROMPT = (
    "You are an expert conversational sentiment analyst.\n\n"
    "Task:\n"
    "Score emotional tone for the full call and for each numbered utterance, using only cues from the text.\n\n"
    "Return a JSON object with:\n\n"
    "1. sentiment_score — float from -1 (very negative) to 1 (very positive) for the entire transcript\n"
    "2. utterance_sentiments — array with one object per utterance; each object has index (int, 0-based) "
    "and sentiment_score (float, -1 to 1)\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON (no markdown, no explanations)\n"
    "* Do not hallucinate or add information not present in transcript\n"
    "* Be concise and precise\n"
    f"* Follow this schema exactly: {json.dumps(SENTIMENT_SCHEMA)}"
)

NEXT_ACTION_SYSTEM_PROMPT = (
    "You are an expert customer-operations strategist.\n\n"
    "Task:\n"
    "Determine the single best next organizational or agent action to resolve or advance the situation, "
    "based solely on the transcript and numbered utterances. The action must be executable and specific "
    "(who does what), not vague advice.\n\n"
    "Return a JSON object with:\n\n"
    "1. next_move — one string naming a concrete, actionable next step tied to transcript evidence\n"
    "2. reasoning — short string citing why this step is warranted\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON (no markdown, no explanations)\n"
    "* Do not hallucinate or add information not present in transcript\n"
    "* Be concise and precise\n"
    f"* Follow this schema exactly: {json.dumps(NEXT_ACTION_SCHEMA)}"
)

ESCALATION_SYSTEM_PROMPT = (
    "You are an expert support QA and risk analyst.\n\n"
    "Task:\n"
    "Decide whether the call warrants escalation (e.g. severe dissatisfaction, legal/compliance risk, "
    "repeated failure, explicit supervisor demand) and specify priority and follow-up tasks.\n\n"
    "Return a JSON object with:\n\n"
    "1. escalation_required — boolean\n"
    "2. priority — exactly one of: low, medium, high\n"
    "3. reason — short factual explanation grounded in the transcript\n"
    "4. action_items — array of concise, practical follow-up tasks (empty if none)\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON (no markdown, no explanations)\n"
    "* Do not hallucinate or add information not present in transcript\n"
    "* Be concise and precise\n"
    f"* Follow this schema exactly: {json.dumps(ESCALATION_SCHEMA)}"
)

EMAIL_SYSTEM_PROMPT = (
    "You are an expert customer communications writer.\n\n"
    "Task:\n"
    "Draft a professional, human-sounding follow-up email to the customer that reflects what was actually "
    "discussed, acknowledges their situation, and states clear resolution or next steps without inventing "
    "names, dates, amounts, or promises not supported by the transcript.\n\n"
    "Return a JSON object with:\n\n"
    "1. subject — clear, professional subject line\n"
    "2. email_body — full email including greeting and sign-off; natural tone, not robotic\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON (no markdown, no explanations)\n"
    "* Do not hallucinate or add information not present in transcript\n"
    "* Be concise and precise\n"
    f"* Follow this schema exactly: {json.dumps(EMAIL_SCHEMA)}"
)

AGENT_PERFORMANCE_SYSTEM_PROMPT = (
    "You are an expert call-quality and coaching analyst.\n\n"
    "Task:\n"
    "Evaluate the support agent's performance using only observable behaviors and statements in the "
    "transcript and numbered utterances. Tie every score and bullet to specific evidence; reject generic "
    "praise or criticism.\n\n"
    "Return a JSON object with:\n\n"
    "1. overall_score — float from 0 to 10 summarizing agent performance\n"
    "2. category_scores — object with integer 0–10 scores for empathy, clarity, professionalism, "
    "problem_resolution\n"
    "3. strengths — array of short strings, each referencing what the agent did well with implicit "
    "grounding in the call\n"
    "4. improvements — array of short, actionable coaching points grounded in the call\n\n"
    "Rules:\n\n"
    "* Return ONLY valid JSON (no markdown, no explanations)\n"
    "* Do not hallucinate or add information not present in transcript\n"
    "* Be concise and precise\n"
    f"* Follow this schema exactly: {json.dumps(AGENT_PERFORMANCE_SCHEMA)}"
)

PROMPT_MAP: Dict[str, str] = {
    "transcript_analysis": TRANSCRIPT_SYSTEM_PROMPT,
    "summary": SUMMARY_SYSTEM_PROMPT,
    "intent": INTENT_SYSTEM_PROMPT,
    "sentiment": SENTIMENT_SYSTEM_PROMPT,
    "next_action": NEXT_ACTION_SYSTEM_PROMPT,
    "escalation": ESCALATION_SYSTEM_PROMPT,
    "email": EMAIL_SYSTEM_PROMPT,
    "agent_performance": AGENT_PERFORMANCE_SYSTEM_PROMPT,
}

MAX_TRANSCRIPT_CHARS = 12000


def _fill_utterance_sentiments(
    result: Dict[str, Any],
    utterance_count: int,
) -> None:
    raw_utt = result.get("utterance_sentiments")
    if not isinstance(raw_utt, list):
        raw_utt = []

    utt_map: Dict[int, float] = {}
    for item in raw_utt:
        if isinstance(item, dict) and "index" in item:
            try:
                utt_map[int(item["index"])] = float(item.get("sentiment_score", 0.0))
            except (TypeError, ValueError):
                continue

    filled: List[Dict[str, Any]] = []
    for i in range(utterance_count):
        filled.append({
            "index": i,
            "sentiment_score": utt_map.get(i, 0.0),
        })
    result["utterance_sentiments"] = filled


def _validate_transcript_analysis(result: Dict[str, Any], utterance_count: int) -> None:
    """Full unified analysis shape (backward compatible)."""
    if "sentiment_score" not in result:
        result["sentiment_score"] = 0.0
    if not isinstance(result.get("intents"), list):
        result["intents"] = []
    if not isinstance(result.get("summary"), list):
        result["summary"] = []
    _fill_utterance_sentiments(result, utterance_count)


def _validate_sentiment_task(result: Dict[str, Any], utterance_count: int) -> None:
    if "sentiment_score" not in result:
        result["sentiment_score"] = 0.0
    _fill_utterance_sentiments(result, utterance_count)


def _validate_summary_task(result: Dict[str, Any], utterance_count: int) -> None:
    del utterance_count
    if not isinstance(result.get("summary"), list):
        result["summary"] = []


def _validate_intent_task(result: Dict[str, Any], utterance_count: int) -> None:
    del utterance_count
    if not isinstance(result.get("intents"), list):
        result["intents"] = []


def _validate_next_action_task(result: Dict[str, Any], utterance_count: int) -> None:
    del utterance_count
    if not isinstance(result.get("next_move"), str):
        result["next_move"] = ""
    if not isinstance(result.get("reasoning"), str):
        result["reasoning"] = ""


def _validate_escalation_task(result: Dict[str, Any], utterance_count: int) -> None:
    del utterance_count
    esc = result.get("escalation_required")
    if isinstance(esc, str):
        esc = esc.lower() in ("true", "1", "yes")
    elif not isinstance(esc, bool):
        esc = bool(esc) if esc is not None else False
    result["escalation_required"] = esc
    p = result.get("priority")
    if not isinstance(p, str) or p.lower() not in ("low", "medium", "high"):
        result["priority"] = "low"
    else:
        result["priority"] = p.lower()
    if not isinstance(result.get("reason"), str):
        result["reason"] = ""
    if not isinstance(result.get("action_items"), list):
        result["action_items"] = []


def _validate_email_task(result: Dict[str, Any], utterance_count: int) -> None:
    del utterance_count
    if not isinstance(result.get("subject"), str):
        result["subject"] = ""
    if not isinstance(result.get("email_body"), str):
        result["email_body"] = ""


def _validate_agent_performance_task(result: Dict[str, Any], utterance_count: int) -> None:
    del utterance_count
    if "overall_score" not in result:
        result["overall_score"] = 0.0
    cats = result.get("category_scores")
    keys = ("empathy", "clarity", "professionalism", "problem_resolution")
    if not isinstance(cats, dict):
        cats = {}
    for k in keys:
        if k not in cats:
            cats[k] = 0
        try:
            cats[k] = max(0, min(10, int(round(float(cats[k])))))
        except (TypeError, ValueError):
            cats[k] = 0
    result["category_scores"] = {k: cats[k] for k in keys}
    if not isinstance(result.get("strengths"), list):
        result["strengths"] = []
    if not isinstance(result.get("improvements"), list):
        result["improvements"] = []


_TASK_VALIDATORS = {
    "transcript_analysis": _validate_transcript_analysis,
    "summary": _validate_summary_task,
    "intent": _validate_intent_task,
    "sentiment": _validate_sentiment_task,
    "next_action": _validate_next_action_task,
    "escalation": _validate_escalation_task,
    "email": _validate_email_task,
    "agent_performance": _validate_agent_performance_task,
}


def _validate_result_for_task(
    result: Dict[str, Any],
    task_name: str,
    utterance_count: int,
) -> None:
    validator = _TASK_VALIDATORS.get(task_name)
    if validator is None:
        return
    validator(result, utterance_count)


def run_analysis_task(
    transcript_text: str,
    utterances: List[Dict[str, Any]],
    task_name: str,
) -> Dict[str, Any]:
    """
    Run a single Azure OpenAI JSON task over transcript + numbered utterances.
    Use task_name from PROMPT_MAP keys (e.g. "transcript_analysis").
    """
    if task_name not in PROMPT_MAP:
        raise ValueError(
            f"Unknown task_name {task_name!r}; expected one of {tuple(PROMPT_MAP)}"
        )

    system_prompt = PROMPT_MAP[task_name]
    client = get_azure_openai_client()
    deployment = get_azure_openai_deployment()

    trimmed = transcript_text[:MAX_TRANSCRIPT_CHARS]

    utterance_lines = "\n".join(
        f"[{i}] {u.get('text', '')}" for i, u in enumerate(utterances)
    )
    utterance_block = utterance_lines[:MAX_TRANSCRIPT_CHARS]

    if task_name == "summary":
        user_content = (
            "Use only the transcript content below. Ignore formatting artifacts; focus on spoken substance.\n\n"
            f"=== FULL TRANSCRIPT ===\n{trimmed}\n"
        )
        if utterance_block.strip():
            user_content += (
                "\n=== UTTERANCES (numbered, optional context) ===\n"
                f"{utterance_block}\n"
            )
    else:
        user_content = (
            f"=== FULL TRANSCRIPT ===\n{trimmed}\n\n"
            f"=== UTTERANCES (numbered) ===\n{utterance_block}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "OpenAI task=%s started (%d utterances, %d chars)",
        task_name,
        len(utterances),
        len(trimmed),
    )
    t0 = time.perf_counter()

    completion = create_chat_completion(
        client,
        model=deployment,
        messages=messages,
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "OpenAI task=%s completed in %.2f seconds",
        task_name,
        elapsed,
    )

    result = parse_json_content(completion.choices[0].message.content)
    if not isinstance(result, dict):
        raise RuntimeError(f"Model returned non-object JSON for task={task_name}")

    _validate_result_for_task(result, task_name, len(utterances))
    return result


def analyze_transcript_with_openai(
    transcript_text: str,
    utterances: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Single Azure OpenAI call: full transcript analysis (sentiment, utterances, intents, summary).
    Equivalent to run_analysis_task(..., task_name="transcript_analysis").
    """
    return run_analysis_task(transcript_text, utterances, "transcript_analysis")
