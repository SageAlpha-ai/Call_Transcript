import os
import json
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


_JSON_HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm")


# ---------------------------------------------------------------------------
# Azure helpers
# ---------------------------------------------------------------------------

def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_blob_service_client() -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(
        get_required_env("AZURE_STORAGE_CONNECTION_STRING")
    )


@st.cache_data(ttl=120)
def list_audio_blobs() -> List[str]:
    container = get_required_env("AZURE_STORAGE_CONTAINER_NAME")
    client = get_blob_service_client().get_container_client(container)
    return sorted(
        b.name for b in client.list_blobs()
        if b.name.lower().endswith(AUDIO_EXTENSIONS)
    )


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def api_transcribe(audio_blob_path: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{API_BASE}/transcribe",
        json={"audio_blob_path": audio_blob_path},
        headers=_JSON_HEADERS,
        timeout=300,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Transcribe failed ({resp.status_code}): {_err(resp)}")
    return resp.json()


def api_analyze(transcript: str, utterances: Optional[List[Dict]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"transcript": transcript}
    if utterances:
        body["utterances"] = utterances
    resp = requests.post(
        f"{API_BASE}/analyze", json=body, headers=_JSON_HEADERS, timeout=300
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Analyze failed ({resp.status_code}): {_err(resp)}")
    return resp.json()


def api_analyze_all(transcript: str, utterances: Optional[List[Dict]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"transcript": transcript}
    if utterances:
        body["utterances"] = utterances
    resp = requests.post(
        f"{API_BASE}/analyze-all", json=body, headers=_JSON_HEADERS, timeout=300
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Analyze-all failed ({resp.status_code}): {_err(resp)}")
    return resp.json()


def api_next_move(
    transcript: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"transcript": transcript}
    if summary is not None:
        body["summary"] = summary
    if intents is not None:
        body["intents"] = intents
    resp = requests.post(
        f"{API_BASE}/next-move", json=body, headers=_JSON_HEADERS, timeout=120
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Next move failed ({resp.status_code}): {_err(resp)}")
    return resp.json()


def api_escalation(
    transcript: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"transcript": transcript}
    if summary is not None:
        body["summary"] = summary
    if intents is not None:
        body["intents"] = intents
    resp = requests.post(
        f"{API_BASE}/escalation", json=body, headers=_JSON_HEADERS, timeout=120
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Escalation failed ({resp.status_code}): {_err(resp)}")
    return resp.json()


def api_followup_email(
    transcript: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
    next_move: Optional[str] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"transcript": transcript}
    if summary is not None:
        body["summary"] = summary
    if intents is not None:
        body["intents"] = intents
    if next_move:
        body["next_move"] = next_move
    resp = requests.post(
        f"{API_BASE}/followup-email", json=body, headers=_JSON_HEADERS, timeout=120
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Follow-up email failed ({resp.status_code}): {_err(resp)}")
    return resp.json()


def api_agent_score(
    transcript: str,
    summary: Optional[List[str]] = None,
    intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"transcript": transcript}
    if summary is not None:
        body["summary"] = summary
    if intents is not None:
        body["intents"] = intents
    resp = requests.post(
        f"{API_BASE}/agent-score", json=body, headers=_JSON_HEADERS, timeout=120
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Agent score failed ({resp.status_code}): {_err(resp)}")
    return resp.json()


def _err(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error") or data.get("message") or data.get("detail")
            det = data.get("details")
            if err and det:
                return f"{err}: {det}"
            if err:
                return str(err)
        return json.dumps(data)
    except Exception:
        return resp.text


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init() -> None:
    defaults: Dict[str, Any] = {
        "selected_file": None,
        "transcribed_file": None,
        "transcript": None,
        "utterances": None,
        "analyzed_file": None,
        "analysis": None,
        "show_summary": False,
        "show_intent": False,
        "show_sentiment": False,
        "next_move": None,
        "next_move_sig": None,
        "escalation": None,
        "escalation_sig": None,
        "followup_email": None,
        "followup_email_sig": None,
        "agent_score": None,
        "agent_score_sig": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _on_file_change() -> None:
    """Called when the user picks a different recording."""
    st.session_state["transcript"] = None
    st.session_state["transcribed_file"] = None
    st.session_state["utterances"] = None
    st.session_state["analysis"] = None
    st.session_state["analyzed_file"] = None
    st.session_state["show_summary"] = False
    st.session_state["show_intent"] = False
    st.session_state["show_sentiment"] = False
    st.session_state["next_move"] = None
    st.session_state["next_move_sig"] = None
    st.session_state["escalation"] = None
    st.session_state["escalation_sig"] = None
    st.session_state["followup_email"] = None
    st.session_state["followup_email_sig"] = None
    st.session_state["agent_score"] = None
    st.session_state["agent_score_sig"] = None


# ---------------------------------------------------------------------------
# Button callbacks — run ONCE, never re-fire on Streamlit reruns
# ---------------------------------------------------------------------------

def _do_transcribe() -> None:
    """Callback for Get Transcript button."""
    blob = st.session_state.get("selected_file")
    if not blob:
        return

    if st.session_state.get("transcribed_file") == blob:
        return

    try:
        res = api_transcribe(blob)
        st.session_state["transcript"] = res.get("transcript", "")
        st.session_state["utterances"] = res.get("utterances")
        st.session_state["transcribed_file"] = blob
        st.session_state["analysis"] = None
        st.session_state["analyzed_file"] = None
        st.session_state["show_summary"] = False
        st.session_state["show_intent"] = False
        st.session_state["show_sentiment"] = False
        st.session_state["next_move"] = None
        st.session_state["next_move_sig"] = None
        st.session_state["escalation"] = None
        st.session_state["escalation_sig"] = None
        st.session_state["followup_email"] = None
        st.session_state["followup_email_sig"] = None
        st.session_state["agent_score"] = None
        st.session_state["agent_score_sig"] = None
    except Exception as exc:
        st.session_state["_last_error"] = f"Transcription failed: {exc}"


def _do_analyze(show_key: str) -> None:
    """Callback for any analysis button. Calls API once, then caches."""
    transcript = st.session_state.get("transcript")
    if not transcript:
        return

    current_file = st.session_state.get("transcribed_file")
    if st.session_state.get("analyzed_file") != current_file:
        try:
            result = api_analyze(transcript, st.session_state.get("utterances"))
            st.session_state["analysis"] = result
            st.session_state["analyzed_file"] = current_file
        except Exception as exc:
            st.session_state["_last_error"] = f"Analysis failed: {exc}"
            return

    st.session_state[show_key] = True


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Voice Analysis Pipeline", layout="wide")
    _init()

    st.title("Voice Analysis Pipeline")

    # ---- Sidebar: recording picker ----
    with st.sidebar:
        st.header("Recordings")
        try:
            audio_files = list_audio_blobs()
        except Exception as exc:
            st.error(f"Cannot list recordings: {exc}")
            return

        if not audio_files:
            st.warning("No audio files found in the blob container.")
            return

        prev = st.session_state["selected_file"]
        idx = audio_files.index(prev) if prev in audio_files else 0
        st.selectbox(
            "Select call recording",
            audio_files,
            index=idx,
            key="selected_file",
            on_change=_on_file_change,
        )

        st.caption("Blob path:")
        st.code(st.session_state["selected_file"], language="text")

    # ---- Show any error from a callback ----
    last_err = st.session_state.pop("_last_error", None)
    if last_err:
        st.error(last_err)

    # ==================================================================
    # STEP 1 — Get Transcript
    # ==================================================================
    already_transcribed = (
        st.session_state.get("transcribed_file") == st.session_state.get("selected_file")
        and st.session_state.get("transcript")
    )

    if already_transcribed:
        st.success("Transcript loaded (cached)")
    else:
        st.button(
            "Get Transcript",
            use_container_width=True,
            on_click=_do_transcribe,
        )

    # ---- Show transcript ----
    if st.session_state["transcript"]:
        st.subheader("Transcript")
        st.text_area(
            "Full transcript text",
            value=st.session_state["transcript"],
            height=200,
            key="ta_transcript",
        )

        st.checkbox("⚡ Fast Mode (Single Call)", key="fast_mode")

        # ==============================================================
        # STEP 2 — Action buttons (only after transcript ready)
        # ==============================================================
        st.divider()
        if not st.session_state.get("fast_mode"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.button(
                    "Summarize",
                    use_container_width=True,
                    on_click=_do_analyze,
                    args=("show_summary",),
                )
            with col2:
                st.button(
                    "Detect Intent",
                    use_container_width=True,
                    on_click=_do_analyze,
                    args=("show_intent",),
                )
            with col3:
                st.button(
                    "Sentiment",
                    use_container_width=True,
                    on_click=_do_analyze,
                    args=("show_sentiment",),
                )
        else:
            fa_busy = st.session_state.get("_analyze_all_busy", False)
            if st.button(
                "Run full analysis (single call)",
                use_container_width=True,
                disabled=fa_busy,
                key="btn_analyze_all",
            ):
                st.session_state["_analyze_all_busy"] = True

            if st.session_state.pop("_analyze_all_busy", False):
                with st.spinner("Running unified analysis (one API call)…"):
                    try:
                        raw = api_analyze_all(
                            st.session_state["transcript"],
                            st.session_state.get("utterances"),
                        )
                        if raw.get("status") != "success":
                            raise RuntimeError(raw.get("message", "unexpected response"))
                        summary_list = raw.get("summary") if isinstance(raw.get("summary"), list) else []
                        intents_list = raw.get("intents") if isinstance(raw.get("intents"), list) else []
                        st.session_state["analysis"] = {
                            "sentiment_score": raw.get("sentiment_score", 0.0),
                            "utterance_sentiments": raw.get("utterance_sentiments")
                            if isinstance(raw.get("utterance_sentiments"), list)
                            else [],
                            "intents": intents_list,
                            "summary": summary_list,
                        }
                        st.session_state["analyzed_file"] = st.session_state.get("transcribed_file")
                        st.session_state["show_summary"] = True
                        st.session_state["show_intent"] = True
                        st.session_state["show_sentiment"] = True
                        sig = hash(
                            (
                                st.session_state.get("transcript") or "",
                                tuple(summary_list),
                                tuple(intents_list),
                            )
                        )
                        nm = (raw.get("next_move") or "").strip()
                        st.session_state["next_move"] = {
                            "next_move": raw.get("next_move", ""),
                            "reasoning": str(raw.get("reasoning", "")),
                        }
                        st.session_state["next_move_sig"] = sig
                        esc = raw.get("escalation")
                        if not isinstance(esc, dict):
                            esc = {}
                        st.session_state["escalation"] = {
                            "escalation_required": bool(esc.get("escalation_required")),
                            "priority": str(esc.get("priority", "low")),
                            "reason": str(esc.get("reason", "")),
                            "action_items": esc.get("action_items")
                            if isinstance(esc.get("action_items"), list)
                            else [],
                        }
                        st.session_state["escalation_sig"] = sig
                        fe = raw.get("followup_email")
                        if not isinstance(fe, dict):
                            fe = {}
                        st.session_state["followup_email"] = {
                            "subject": str(fe.get("subject", "")),
                            "email_body": str(fe.get("email_body", "")),
                        }
                        st.session_state["followup_email_sig"] = hash(
                            (
                                st.session_state.get("transcript") or "",
                                tuple(summary_list),
                                tuple(intents_list),
                                nm,
                            )
                        )
                        ag = raw.get("agent_score")
                        if not isinstance(ag, dict):
                            ag = {}
                        cats = ag.get("category_scores")
                        if not isinstance(cats, dict):
                            cats = {}
                        st.session_state["agent_score"] = {
                            "overall_score": float(ag.get("overall_score", 0)),
                            "category_scores": {
                                "empathy": int(cats.get("empathy", 0)),
                                "clarity": int(cats.get("clarity", 0)),
                                "professionalism": int(cats.get("professionalism", 0)),
                                "problem_resolution": int(cats.get("problem_resolution", 0)),
                            },
                            "strengths": ag.get("strengths")
                            if isinstance(ag.get("strengths"), list)
                            else [],
                            "improvements": ag.get("improvements")
                            if isinstance(ag.get("improvements"), list)
                            else [],
                        }
                        st.session_state["agent_score_sig"] = sig
                    except Exception as exc:
                        st.session_state["_last_error"] = f"Unified analysis failed: {exc}"
                st.rerun()

        # ==============================================================
        # STEP 3 — Display results
        # ==============================================================
        analysis: Optional[Dict[str, Any]] = st.session_state.get("analysis")
        if not analysis:
            return

        st.divider()

        if st.session_state["show_summary"]:
            st.subheader("Summary")
            summary = analysis.get("summary", [])
            if isinstance(summary, list) and summary:
                for point in summary:
                    st.markdown(f"- {point}")
            else:
                st.info("No summary available.")

        if st.session_state["show_intent"]:
            st.subheader("Intent")
            intents = analysis.get("intents", [])
            if intents:
                for intent in intents:
                    st.markdown(f"- `{intent}`")
            else:
                st.info("No intents detected.")

        if st.session_state["show_sentiment"]:
            st.subheader("Sentiment")
            score = analysis.get("sentiment_score")
            if score is not None:
                label = "Positive" if score > 0.2 else "Negative" if score < -0.2 else "Neutral"
                st.metric("Overall Sentiment", f"{score:+.2f}", label)

            utt_sents = analysis.get("utterance_sentiments", [])
            if utt_sents:
                with st.expander("Per-utterance sentiment", expanded=False):
                    for item in utt_sents:
                        s = item.get("sentiment_score", 0)
                        dot = "+" if s > 0.2 else "-" if s < -0.2 else "~"
                        st.markdown(f"**[{dot} {s:+.2f}]** {item.get('text', '')}")

        st.divider()
        summary_list = analysis.get("summary") if isinstance(analysis.get("summary"), list) else []
        intents_list = analysis.get("intents") if isinstance(analysis.get("intents"), list) else []
        sig = hash(
            (
                st.session_state.get("transcript") or "",
                tuple(summary_list),
                tuple(intents_list),
            )
        )
        nm_busy = st.session_state.get("_next_move_busy", False)
        if st.button(
            "Next Move",
            use_container_width=True,
            disabled=nm_busy,
            key="btn_next_move",
        ):
            st.session_state["_next_move_busy"] = True

        if st.session_state.pop("_next_move_busy", False):
            with st.spinner("Getting next-move suggestion…"):
                try:
                    res = api_next_move(
                        st.session_state["transcript"],
                        summary=summary_list,
                        intents=intents_list,
                    )
                    st.session_state["next_move"] = {
                        "next_move": res.get("next_move", ""),
                        "reasoning": res.get("reasoning", ""),
                    }
                    st.session_state["next_move_sig"] = sig
                except Exception as exc:
                    st.session_state["_last_error"] = f"Next move failed: {exc}"
            st.rerun()

        cached = st.session_state.get("next_move")
        cached_sig = st.session_state.get("next_move_sig")
        if cached and cached_sig == sig:
            st.subheader("Suggested Next Action")
            st.markdown(f"### {cached.get('next_move', '')}")
            st.caption(cached.get("reasoning", ""))

        st.divider()
        st.subheader("Escalation & Actions")
        esc_busy = st.session_state.get("_escalation_busy", False)
        if st.button(
            "Check Escalation",
            use_container_width=True,
            disabled=esc_busy,
            key="btn_escalation",
        ):
            st.session_state["_escalation_busy"] = True

        if st.session_state.pop("_escalation_busy", False):
            with st.spinner("Checking escalation and extracting actions…"):
                try:
                    er = api_escalation(
                        st.session_state["transcript"],
                        summary=summary_list,
                        intents=intents_list,
                    )
                    st.session_state["escalation"] = {
                        "escalation_required": bool(er.get("escalation_required")),
                        "priority": str(er.get("priority", "low")),
                        "reason": str(er.get("reason", "")),
                        "action_items": er.get("action_items")
                        if isinstance(er.get("action_items"), list)
                        else [],
                    }
                    st.session_state["escalation_sig"] = sig
                except Exception as exc:
                    st.session_state["_last_error"] = f"Escalation check failed: {exc}"
            st.rerun()

        esc_cached = st.session_state.get("escalation")
        esc_sig = st.session_state.get("escalation_sig")
        if esc_cached and esc_sig == sig:
            if esc_cached.get("escalation_required"):
                st.markdown(":red[**Escalation required**]")
            else:
                st.markdown(":green[**No escalation required**]")
            pri = (esc_cached.get("priority") or "low").lower()
            if pri == "high":
                st.markdown("**Priority:** :red[high]")
            elif pri == "medium":
                st.markdown("**Priority:** :orange[medium]")
            else:
                st.markdown("**Priority:** :green[low]")
            st.markdown(f"**Reason:** {esc_cached.get('reason', '')}")
            items = esc_cached.get("action_items") or []
            if items:
                st.markdown("**Action items**")
                for t in items:
                    st.markdown(f"- {t}")
            else:
                st.caption("No action items.")

        nm_cached = st.session_state.get("next_move")
        nm_sig_ok = st.session_state.get("next_move_sig") == sig
        next_move_for_email = ""
        if nm_cached and nm_sig_ok:
            next_move_for_email = (nm_cached.get("next_move") or "").strip()

        email_sig = hash(
            (
                st.session_state.get("transcript") or "",
                tuple(summary_list),
                tuple(intents_list),
                next_move_for_email,
            )
        )
        st.divider()
        st.subheader("Follow-up Email")
        fe_busy = st.session_state.get("_followup_email_busy", False)
        if st.button(
            "Generate Email",
            use_container_width=True,
            disabled=fe_busy,
            key="btn_followup_email",
        ):
            st.session_state["_followup_email_busy"] = True

        if st.session_state.pop("_followup_email_busy", False):
            with st.spinner("Generating follow-up email…"):
                try:
                    fe = api_followup_email(
                        st.session_state["transcript"],
                        summary=summary_list,
                        intents=intents_list,
                        next_move=next_move_for_email or None,
                    )
                    st.session_state["followup_email"] = {
                        "subject": str(fe.get("subject", "")),
                        "email_body": str(fe.get("email_body", "")),
                    }
                    st.session_state["followup_email_sig"] = email_sig
                except Exception as exc:
                    st.session_state["_last_error"] = f"Follow-up email failed: {exc}"
            st.rerun()

        fe_cached = st.session_state.get("followup_email")
        fe_sig = st.session_state.get("followup_email_sig")
        if fe_cached and fe_sig == email_sig:
            sk = f"fu_{email_sig}"
            subj_key = f"fu_subj_{sk}"
            body_key = f"fu_body_{sk}"
            st.markdown("**Subject**")
            st.text_input(
                "subject_field",
                value=fe_cached.get("subject", ""),
                key=subj_key,
                label_visibility="collapsed",
                placeholder="Email subject",
            )
            st.markdown("**Email body**")
            st.text_area(
                "body_field",
                value=fe_cached.get("email_body", ""),
                height=280,
                key=body_key,
                label_visibility="collapsed",
                placeholder="Email body",
            )
            body_widget = st.session_state.get(body_key, fe_cached.get("email_body", ""))
            subj_widget = st.session_state.get(subj_key, fe_cached.get("subject", ""))
            combined = f"Subject: {subj_widget}\n\n{body_widget}"
            st.download_button(
                label="Download email (.txt)",
                data=combined,
                file_name="followup_email.txt",
                mime="text/plain",
                key=f"fu_dl_{sk}",
                use_container_width=True,
            )
            try:
                import pyperclip  # type: ignore[import-untyped]

                if st.button("Copy to clipboard", key=f"fu_copy_{sk}", use_container_width=True):
                    pyperclip.copy(combined)
                    st.toast("Copied to clipboard")
            except ImportError:
                pass

        st.divider()
        st.subheader("Agent Performance")
        as_busy = st.session_state.get("_agent_score_busy", False)
        if st.button(
            "Evaluate Agent",
            use_container_width=True,
            disabled=as_busy,
            key="btn_agent_score",
        ):
            st.session_state["_agent_score_busy"] = True

        if st.session_state.pop("_agent_score_busy", False):
            with st.spinner("Evaluating agent performance…"):
                try:
                    ar = api_agent_score(
                        st.session_state["transcript"],
                        summary=summary_list,
                        intents=intents_list,
                    )
                    cats = ar.get("category_scores")
                    if not isinstance(cats, dict):
                        cats = {}
                    st.session_state["agent_score"] = {
                        "overall_score": float(ar.get("overall_score", 0)),
                        "category_scores": {
                            "empathy": int(cats.get("empathy", 0)),
                            "clarity": int(cats.get("clarity", 0)),
                            "professionalism": int(cats.get("professionalism", 0)),
                            "problem_resolution": int(cats.get("problem_resolution", 0)),
                        },
                        "strengths": ar.get("strengths")
                        if isinstance(ar.get("strengths"), list)
                        else [],
                        "improvements": ar.get("improvements")
                        if isinstance(ar.get("improvements"), list)
                        else [],
                    }
                    st.session_state["agent_score_sig"] = sig
                except Exception as exc:
                    st.session_state["_last_error"] = f"Agent scoring failed: {exc}"
            st.rerun()

        as_cached = st.session_state.get("agent_score")
        as_sig = st.session_state.get("agent_score_sig")
        if as_cached and as_sig == sig:
            ov = float(as_cached.get("overall_score", 0))
            if ov >= 8:
                st.markdown(f"### Overall score: :green[{ov:.1f}] / 10")
            elif ov >= 5:
                st.markdown(f"### Overall score: :orange[{ov:.1f}] / 10")
            else:
                st.markdown(f"### Overall score: :red[{ov:.1f}] / 10")
            st.progress(min(max(ov / 10.0, 0.0), 1.0))
            st.markdown("**Category breakdown**")
            cat_labels = {
                "empathy": "Empathy",
                "clarity": "Clarity",
                "professionalism": "Professionalism",
                "problem_resolution": "Problem resolution",
            }
            cs = as_cached.get("category_scores") or {}
            for key, label in cat_labels.items():
                v = int(cs.get(key, 0))
                v = max(0, min(10, v))
                ccol = "green" if v >= 8 else "orange" if v >= 5 else "red"
                st.markdown(f"{label}: :{ccol}[{v}/10]")
                st.progress(v / 10.0)
            strengths = as_cached.get("strengths") or []
            st.markdown("**Strengths**")
            if strengths:
                for s in strengths:
                    if isinstance(s, str) and s.strip():
                        st.markdown(f"- :green[{s.strip()}]")
            else:
                st.caption("None listed.")
            improvements = as_cached.get("improvements") or []
            st.markdown("**Improvements**")
            if improvements:
                for s in improvements:
                    if isinstance(s, str) and s.strip():
                        st.markdown(f"- :red[{s.strip()}]")
            else:
                st.caption("None listed.")

    else:
        st.info("Select a recording and click **Get Transcript** to start.")


if __name__ == "__main__":
    main()
