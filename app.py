import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

from middleware import (
    RequestTimeoutMiddleware,
    configure_structured_logging,
    max_body_middleware,
    request_logging_middleware,
)
from security import rate_limit_middleware

configure_structured_logging()

from analysis import main as run_pipeline
from analysis import transcribe_audio, analyze_transcript
from analysis.agent_scoring import score_agent_performance_with_openai
from analysis.escalation import detect_escalation_with_openai
from analysis.followup_email import generate_followup_email_with_openai
from analysis.next_move import suggest_next_move_with_openai
from analysis.unified_full_analysis import analyze_all_with_openai
from config import get_allowed_origins, get_max_transcript_chars, validate_required_environment

logger = logging.getLogger(__name__)

DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
_MAX_TRANSCRIPT = get_max_transcript_chars()
_MAX_LIST_ITEMS = 500
_MAX_UTTERANCES = 5000


def _stable_id(audio_blob_path: str) -> str:
    return hashlib.sha256(audio_blob_path.encode()).hexdigest()[:16]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    validate_required_environment()
    logger.info(
        "Environment loaded: AZURE_STORAGE_CONNECTION_STRING is %s",
        "set" if os.getenv("AZURE_STORAGE_CONNECTION_STRING") else "missing",
    )
    yield


app = FastAPI(title="Voice Analytics Pipeline API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
app.add_middleware(RequestTimeoutMiddleware)


@app.middleware("http")
async def _security_stack(request: Request, call_next):
    async def _inner(req: Request):
        return await call_next(req)

    async def _log(req: Request):
        return await request_logging_middleware(req, _inner)

    async def _body(req: Request):
        return await max_body_middleware(req, _log)

    return await rate_limit_middleware(request, _body)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "details": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": msg, "details": None},
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    logging.exception("Unhandled error request_id=%s", rid)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": None},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not environment_ready():
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    try:
        r = httpx.get(
            f"{endpoint}/openai/deployments",
            headers={"api-key": key},
            params={"api-version": "2024-12-01-preview"},
            timeout=8.0,
        )
        if r.status_code >= 400:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready"}


@app.get("/ping")
def ping():
    return PlainTextResponse("pong", media_type="text/plain")


class _ListCapMixin(BaseModel):
    @field_validator("summary", "intents", mode="before", check_fields=False)
    @classmethod
    def _cap_lists(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            return v
        if len(v) > _MAX_LIST_ITEMS:
            raise ValueError(f"List exceeds maximum length ({_MAX_LIST_ITEMS})")
        return v


class TranscribeRequest(BaseModel):
    audio_blob_path: str = Field(..., min_length=1, max_length=2048)
    id: str | None = Field(None, max_length=256)
    force: bool = False


class AnalyzeRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=_MAX_TRANSCRIPT)
    utterances: list[dict] | None = None

    @field_validator("utterances", mode="before")
    @classmethod
    def _cap_utterances(cls, v):
        if v is None:
            return v
        if len(v) > _MAX_UTTERANCES:
            raise ValueError(f"utterances exceeds maximum length ({_MAX_UTTERANCES})")
        return v


class NextMoveRequest(_ListCapMixin):
    transcript: str = Field(..., min_length=1, max_length=_MAX_TRANSCRIPT)
    summary: list[str] | None = None
    intents: list[str] | None = None


class EscalationRequest(_ListCapMixin):
    transcript: str = Field(..., min_length=1, max_length=_MAX_TRANSCRIPT)
    summary: list[str] | None = None
    intents: list[str] | None = None


class FollowupEmailRequest(_ListCapMixin):
    transcript: str = Field(..., min_length=1, max_length=_MAX_TRANSCRIPT)
    summary: list[str] | None = None
    intents: list[str] | None = None
    next_move: str | None = Field(None, max_length=8000)


class AgentScoreRequest(_ListCapMixin):
    transcript: str = Field(..., min_length=1, max_length=_MAX_TRANSCRIPT)
    summary: list[str] | None = None
    intents: list[str] | None = None


class PipelineRequest(BaseModel):
    audio_blob_path: str = Field(..., min_length=1, max_length=2048)
    id: str | None = Field(None, max_length=256)
    force: bool = False
    callback_url: str | None = Field(None, max_length=2048)


@app.post("/transcribe")
def transcribe(request: TranscribeRequest):
    try:
        payload = {
            "audio_blob_path": request.audio_blob_path,
            "id": request.id or _stable_id(request.audio_blob_path),
            "force": request.force,
        }
        logging.info(
            "POST /transcribe — blob: %s  id: %s",
            payload["audio_blob_path"],
            payload["id"],
        )
        result = transcribe_audio(payload)
        logging.info("POST /transcribe completed for id=%s", payload["id"])
        return {
            "status": "success",
            "id": payload["id"],
            **result,
        }
    except ValueError as exc:
        logging.error("Transcription validation failed: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "details": "validation"},
        )
    except Exception as exc:
        step = getattr(exc, "step", "transcription")
        _log_error("Transcription", step, exc)
        return _error_response(step, exc)


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        logging.info("POST /analyze — transcript length: %d chars", len(request.transcript))
        result = analyze_transcript(request.transcript, request.utterances)
        logging.info("POST /analyze completed")
        return {
            "status": "success",
            **result,
        }
    except ValueError as exc:
        logging.error("Analysis validation failed: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "details": "validation"},
        )
    except Exception as exc:
        step = getattr(exc, "step", "analysis")
        _log_error("Analysis", step, exc)
        return _error_response(step, exc)


@app.post("/analyze-all")
def analyze_all(request: AnalyzeRequest):
    transcript = (request.transcript or "").strip()
    if not transcript:
        logging.warning("POST /analyze-all rejected: empty transcript")
        return JSONResponse(
            status_code=400,
            content={"error": "transcript is required and cannot be empty", "details": None},
        )

    logging.info(
        "POST /analyze-all — transcript_chars=%d utterances=%s",
        len(transcript),
        len(request.utterances) if request.utterances else "default",
    )
    try:
        t0 = time.perf_counter()
        result = analyze_all_with_openai(transcript, request.utterances)
        elapsed = time.perf_counter() - t0
        logging.info("POST /analyze-all completed latency_s=%.3f", elapsed)
        return {"status": "success", **result}
    except Exception as exc:
        logging.exception("POST /analyze-all failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "analyze_all failed", "details": str(exc)},
        )


@app.post("/next-move")
def next_move(request: NextMoveRequest):
    transcript = (request.transcript or "").strip()
    if not transcript:
        return JSONResponse(
            status_code=400,
            content={"error": "transcript is required and cannot be empty", "details": None},
        )

    summary = request.summary if request.summary is not None else []
    intents = request.intents if request.intents is not None else []
    logging.info(
        "POST /next-move — transcript_chars=%d summary=%d intents=%d",
        len(transcript),
        len(summary),
        len(intents),
    )

    try:
        result = suggest_next_move_with_openai(
            transcript_text=transcript,
            summary=list(summary),
            intents=list(intents),
        )
        logging.info("POST /next-move completed")
        return {
            "next_move": result["next_move"],
            "reasoning": result["reasoning"],
        }
    except Exception as exc:
        logging.exception("POST /next-move failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "next_move failed", "details": str(exc)},
        )


@app.post("/escalation")
def escalation(request: EscalationRequest):
    transcript = (request.transcript or "").strip()
    if not transcript:
        return JSONResponse(
            status_code=400,
            content={"error": "transcript is required and cannot be empty", "details": None},
        )

    summary = request.summary if request.summary is not None else []
    intents = request.intents if request.intents is not None else []
    logging.info(
        "POST /escalation — transcript_chars=%d summary=%d intents=%d",
        len(transcript),
        len(summary),
        len(intents),
    )

    try:
        result = detect_escalation_with_openai(
            transcript_text=transcript,
            summary=list(summary),
            intents=list(intents),
        )
        logging.info("POST /escalation completed")
        return {
            "escalation_required": result["escalation_required"],
            "priority": result["priority"],
            "reason": result["reason"],
            "action_items": result["action_items"],
        }
    except Exception as exc:
        logging.exception("POST /escalation failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "escalation failed", "details": str(exc)},
        )


@app.post("/followup-email")
def followup_email(request: FollowupEmailRequest):
    transcript = (request.transcript or "").strip()
    if not transcript:
        return JSONResponse(
            status_code=400,
            content={"error": "transcript is required and cannot be empty", "details": None},
        )

    summary = request.summary if request.summary is not None else []
    intents = request.intents if request.intents is not None else []
    nm = (request.next_move or "").strip() if request.next_move else None
    logging.info(
        "POST /followup-email — transcript_chars=%d summary=%d intents=%d next_move=%s",
        len(transcript),
        len(summary),
        len(intents),
        "set" if nm else "none",
    )

    try:
        result = generate_followup_email_with_openai(
            transcript_text=transcript,
            summary=list(summary),
            intents=list(intents),
            next_move=nm,
        )
        logging.info("POST /followup-email completed")
        return {
            "subject": result["subject"],
            "email_body": result["email_body"],
        }
    except Exception as exc:
        logging.exception("POST /followup-email failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "followup_email failed", "details": str(exc)},
        )


@app.post("/agent-score")
def agent_score(request: AgentScoreRequest):
    transcript = (request.transcript or "").strip()
    if not transcript:
        return JSONResponse(
            status_code=400,
            content={"error": "transcript is required and cannot be empty", "details": None},
        )

    summary = request.summary if request.summary is not None else []
    intents = request.intents if request.intents is not None else []
    logging.info(
        "POST /agent-score — transcript_chars=%d summary=%d intents=%d",
        len(transcript),
        len(summary),
        len(intents),
    )

    try:
        t0 = time.perf_counter()
        result = score_agent_performance_with_openai(
            transcript_text=transcript,
            summary=list(summary),
            intents=list(intents),
        )
        elapsed = time.perf_counter() - t0
        logging.info("POST /agent-score completed latency_s=%.3f", elapsed)
        return {
            "overall_score": result["overall_score"],
            "category_scores": result["category_scores"],
            "strengths": result["strengths"],
            "improvements": result["improvements"],
        }
    except Exception as exc:
        logging.exception("POST /agent-score failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "agent_score failed", "details": str(exc)},
        )


@app.post("/pipeline")
def pipeline(request: PipelineRequest):
    try:
        payload = {
            "audio_blob_path": request.audio_blob_path,
            "id": request.id or _stable_id(request.audio_blob_path),
            "force": request.force,
            "callback_url": request.callback_url,
        }
        logging.info(
            "POST /pipeline — blob: %s  id: %s",
            payload["audio_blob_path"],
            payload["id"],
        )
        result_blob_url = run_pipeline(payload)
        logging.info("POST /pipeline completed for id=%s", payload["id"])
        return {
            "status": "success",
            "transcript_url": result_blob_url,
            "id": payload["id"],
        }
    except ValueError as exc:
        logging.error("Pipeline validation failed: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": str(exc), "details": "validation"},
        )
    except Exception as exc:
        step = getattr(exc, "step", "unknown")
        _log_error("Pipeline", step, exc)
        return _error_response(step, exc)


def _log_error(label: str, step: str, exc: Exception) -> None:
    logging.error("%s failed at step=%s: %s", label, step, exc, exc_info=True)


def _error_response(step: str, exc: Exception) -> JSONResponse:
    if DEBUG_MODE:
        logging.debug("step=%s detail=%s", step, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Request failed",
            "details": str(exc),
        },
    )
