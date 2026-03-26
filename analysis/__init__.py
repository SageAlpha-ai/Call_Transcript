import json
import logging
import requests
import os

from typing import Any, Dict, List
from dotenv import load_dotenv

from transcribe import transcribe_url, generalize_transcript
from metrics import analyze_transcript_with_openai
from utils import upload_blob_stream, get_wordcloud_data

load_dotenv()
logging.basicConfig(level=logging.INFO)
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
ALLOWED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm")

REQUIRED_ENV_VARS = [
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER_NAME",
    "AZURE_SPEECH_API_KEY",
    "AZURE_SPEECH_API_REGION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
]

TRANSCRIBE_ENV_VARS = [
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_CONTAINER_NAME",
    "AZURE_SPEECH_API_KEY",
    "AZURE_SPEECH_API_REGION",
]

ANALYSIS_ENV_VARS = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
]


class PipelineStepError(RuntimeError):
    def __init__(self, step: str, message: str):
        super().__init__(message)
        self.step = step


def validate_required_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise ValueError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )


def _get_blob_service_client() -> Any:
    from azure.storage.blob import BlobServiceClient

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(conn_str)


def validate_input_payload(payload: Dict[str, Any]) -> None:
    audio_blob_path = payload.get("audio_blob_path")
    if not audio_blob_path:
        raise ValueError("Missing required input field: audio_blob_path")

    logging.info("Received audio blob path: %s", audio_blob_path)

    if not audio_blob_path.lower().endswith(ALLOWED_AUDIO_EXTENSIONS):
        raise ValueError(
            "Unsupported audio file extension. Allowed: "
            + ", ".join(ALLOWED_AUDIO_EXTENSIONS)
        )

    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
    blob_client = _get_blob_service_client().get_blob_client(
        container=container_name, blob=audio_blob_path
    )
    if not blob_client.exists():
        raise ValueError(
            f"Audio blob does not exist in container '{container_name}': {audio_blob_path}"
        )
    logging.info("Validated blob exists in container '%s'", container_name)


def generate_wordcloud_data(transcript_output: Dict[str, Any]) -> Dict[str, Any]:
    """Generate wordcloud data from transcript text."""
    if transcript_output.get("json_response", {}).get("wordcloud"):
        logging.info(f"Using cached wordcloud for _id: {transcript_output['_id']}")
        return transcript_output

    transcript_output["json_response"].update(
        {"wordcloud": get_wordcloud_data(transcript_output.get("text"))}
    )
    return transcript_output


def run_unified_analysis(transcript_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single Azure OpenAI call replacing the previous 4 separate calls
    (overall sentiment, per-utterance sentiment, intent, summary).
    """
    json_resp = transcript_output.get("json_response", {})
    has_all_cached = (
        json_resp.get("sentiment_score") is not None
        and json_resp.get("intents")
        and json_resp.get("summary")
        and all(
            u.get("sentiment_score") is not None
            for u in json_resp.get("utterances", [])
        )
    )
    if has_all_cached:
        logging.info(
            "All analysis fields cached for _id: %s — skipping OpenAI call",
            transcript_output["_id"],
        )
        return transcript_output

    transcript_text = transcript_output.get("text", "")
    utterances = json_resp.get("utterances", [])

    result = analyze_transcript_with_openai(transcript_text, utterances)

    json_resp["sentiment_score"] = result["sentiment_score"]
    json_resp["intents"] = result["intents"]
    json_resp["summary"] = result["summary"]

    utt_sentiments = {
        item["index"]: item["sentiment_score"]
        for item in result.get("utterance_sentiments", [])
    }
    for i, utterance in enumerate(utterances):
        if utterance.get("sentiment_score") is None:
            utterance["sentiment_score"] = utt_sentiments.get(i, 0.0)

    transcript_output["json_response"] = json_resp
    return transcript_output


def transcribe_audio(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 1 — Transcription only (no OpenAI).
    Downloads blob, converts audio, runs Azure Speech, returns transcript.
    """
    missing = [v for v in TRANSCRIBE_ENV_VARS if not os.getenv(v)]
    if missing:
        raise ValueError("Missing env var(s): " + ", ".join(missing))

    validate_input_payload(payload)

    audio_blob_path = payload.get("audio_blob_path")
    _id = payload.get("id")
    force = payload.get("force", False)

    transcribe_payload = {
        "audio_blob_path": audio_blob_path,
        "_id": _id,
        "force": force,
    }

    logging.info("Stage 1: Blob download + audio conversion + speech-to-text start")
    transcript_output = transcribe_url(transcribe_payload)
    logging.info("Stage 1: Speech-to-text completed")

    transcript_output = generalize_transcript(transcript_output)
    logging.info("Transcription completed — returning transcript")

    json_resp = transcript_output.get("json_response", {})
    return {
        "transcript": transcript_output.get("text", ""),
        "utterances": json_resp.get("utterances", []),
    }


def analyze_transcript(
    transcript: str,
    utterances: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Stage 2 — Analysis only (single OpenAI call).
    Takes plain transcript text, returns sentiment + intent + summary.
    """
    missing = [v for v in ANALYSIS_ENV_VARS if not os.getenv(v)]
    if missing:
        raise ValueError("Missing env var(s): " + ", ".join(missing))

    if not transcript or not transcript.strip():
        raise ValueError("transcript must be a non-empty string")

    if utterances is None:
        utterances = [{"text": s.strip()} for s in transcript.split(".") if s.strip()]

    logging.info("Analysis started — single OpenAI call")
    result = analyze_transcript_with_openai(transcript, utterances)
    logging.info("Analysis completed")

    utt_sentiments = result.get("utterance_sentiments", [])
    for item in utt_sentiments:
        idx = item.get("index", 0)
        if idx < len(utterances):
            utterances[idx]["sentiment_score"] = item.get("sentiment_score", 0.0)

    return {
        "sentiment_score": result.get("sentiment_score", 0.0),
        "utterance_sentiments": [
            {
                "text": utterances[i]["text"] if i < len(utterances) else "",
                "sentiment_score": item.get("sentiment_score", 0.0),
            }
            for i, item in enumerate(utt_sentiments)
        ],
        "intents": result.get("intents", []),
        "summary": result.get("summary", []),
    }


def store_updated_transcript(transcript_output: Dict[str, Any]) -> Dict[str, Any]:
    updated_transcript_json = json.dumps(
        transcript_output.get("json_response"), indent=2
    )
    blob_url = upload_blob_stream(
        updated_transcript_json,
        f"{transcript_output.get('_id')}_transcript.json",
        "transcripts",
    )
    return blob_url


def call_callback_url(blob_url: str, callback_url: str) -> bool:
    response = requests.post(callback_url, json={"transcript_url": blob_url})
    return response.content


def main(payload: Dict[str, Any]):
    try:
        validate_required_env()
        validate_input_payload(payload)
        logging.info("Starting pipeline")
        logging.info("Payload: %s", payload)
    except Exception as exc:
        logging.error("Pipeline validation failed: %s", exc, exc_info=DEBUG_MODE)
        raise PipelineStepError("validation", str(exc)) from exc

    audio_blob_path = payload.get("audio_blob_path")
    _id = payload.get("id")
    force = payload.get("force", False)
    callback_url = payload.get("callback_url")

    logging.info(f"Recevied event payload: {payload}")

    transcribe_payload = {
        "audio_blob_path": audio_blob_path,
        "_id": _id,
        "force": force,
        "callback_url": callback_url,
    }

    logging.info(f"Starting analysis with payload: {transcribe_payload}")

    try:
        logging.info("Blob download + audio conversion + speech-to-text start")
        transcript_output = transcribe_url(transcribe_payload)
        logging.info("Speech-to-text completed")
    except Exception as exc:
        logging.error("Speech-to-text failed: %s", exc, exc_info=DEBUG_MODE)
        raise PipelineStepError("speech_to_text", str(exc)) from exc

    try:
        transcript_output = generalize_transcript(transcript_output)
        output = transcript_output.get("json_response", {}).keys()
        logging.info("Generalizing transcript done: %s", output)
    except Exception as exc:
        logging.error("Transcript generalization failed: %s", exc, exc_info=DEBUG_MODE)
        raise PipelineStepError("generalize_transcript", str(exc)) from exc

    try:
        logging.info("Unified Azure OpenAI analysis call start (sentiment + intent + summary)")
        transcript_output = run_unified_analysis(transcript_output)
        json_resp = transcript_output.get("json_response", {})
        logging.info(
            "Unified analysis done — sentiment=%.2f, intents=%s, summary_points=%d",
            json_resp.get("sentiment_score", 0),
            json_resp.get("intents", []),
            len(json_resp.get("summary", [])),
        )
    except Exception as exc:
        logging.error("Unified OpenAI analysis failed: %s", exc, exc_info=DEBUG_MODE)
        raise PipelineStepError("unified_analysis", str(exc)) from exc

    try:
        transcript_output = generate_wordcloud_data(transcript_output)
        logging.info("Wordcloud data generation done")
    except Exception as exc:
        logging.error("Wordcloud generation failed: %s", exc, exc_info=DEBUG_MODE)
        raise PipelineStepError("wordcloud", str(exc)) from exc

    try:
        blob_url = store_updated_transcript(transcript_output)
        logging.info("Final output stored in blob: %s", blob_url)
    except Exception as exc:
        logging.error("Final storage failed: %s", exc, exc_info=DEBUG_MODE)
        raise PipelineStepError("store_output", str(exc)) from exc

    if callback_url:
        try:
            logging.info("Calling callback URL: %s with Blob: %s", callback_url, blob_url)
            callback_res = call_callback_url(blob_url, callback_url)
            logging.info("Callback call complete: %s", callback_res)
        except Exception as exc:
            logging.error("Callback failed: %s", exc, exc_info=DEBUG_MODE)
            raise PipelineStepError("callback", str(exc)) from exc

    return blob_url
