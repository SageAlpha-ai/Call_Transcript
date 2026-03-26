import os
import json
import logging
import tempfile
import uuid

from utils import (
    check_blob_exists,
    download_blob_to_stream,
    download_blob_to_file,
    get_blob_url,
    convert_to_wav,
)
from .azure import transcribe_audio as azure_transcribe_audio


def transcribe_url(payload: dict) -> dict:
    _id = payload.get("_id")
    # if force refresh then transcribe again
    force = payload.get("force", False)

    # if transcript already exists then return it
    folder_name, file_name = "transcripts", f"{_id}_transcript.json"
    if not force and check_blob_exists(folder_name, file_name):
        blob_url = get_blob_url(folder_name, file_name)
        logging.info(f"Using cached transcript: {blob_url}")
        json_response = json.loads(download_blob_to_stream(folder_name, file_name))
        return {
            "_id": _id,
            "blob_url": blob_url,
            "text": json_response.get("text"),
            "json_response": json_response,
        }

    if force:
        logging.info(f"Force transcribing audio with payload: {payload}")

    audio_blob_path = payload.get("audio_blob_path")
    if not audio_blob_path:
        raise RuntimeError(
            "Missing 'audio_blob_path' in payload. Azure-native mode only supports Azure Blob input."
        )

    source_name = os.path.basename(audio_blob_path) or f"{_id}.wav"
    unique_prefix = uuid.uuid4().hex[:8]
    local_source_path = os.path.join(tempfile.gettempdir(), f"{unique_prefix}_{source_name}")
    logging.info("Request %s: temp file path %s", _id, local_source_path)
    try:
        logging.info("Blob download start: %s", audio_blob_path)
        audio_path = download_blob_to_file(audio_blob_path, local_source_path)
        logging.info("Blob download complete: %s", audio_path)
    except Exception as exc:
        logging.error("Blob download failed: %s", exc)
        raise

    try:
        logging.info("Audio conversion start for file: %s", audio_path)
        wav_audio_output = convert_to_wav(audio_path)
        logging.info("Audio conversion output: %s", wav_audio_output)
    except Exception as exc:
        logging.error("Audio conversion failed: %s", exc, exc_info=True)
        raise
    if (
        wav_audio_output.get("success")
        and (wav_audio_path := wav_audio_output.get("output"))
        and (wav_audio_exists := os.path.exists(wav_audio_path))
    ):
        logging.info(
            f"Convert to wav done for Payload: {payload} with wav output: {wav_audio_output} "
            f"and path exists: {wav_audio_exists}"
        )
    else:
        raise RuntimeError(
            f"Conversion to wav failed for Payload: {payload} with wav output: {wav_audio_output}"
        )

    logging.info(
        "Pre-transcription check path=%s isfile=%s",
        wav_audio_path,
        os.path.isfile(wav_audio_path),
    )
    if not os.path.isfile(wav_audio_path):
        raise RuntimeError(f"WAV for transcription does not exist: {wav_audio_path}")
    try:
        wav_size = os.path.getsize(wav_audio_path)
    except OSError as exc:
        logging.error("Pre-transcription stat failed path=%s err=%s", wav_audio_path, exc)
        raise RuntimeError(f"Cannot read WAV file size: {wav_audio_path}") from exc
    logging.info(
        "Pre-transcription WAV path=%s size_bytes=%d",
        wav_audio_path,
        wav_size,
    )
    if wav_size <= 0:
        raise RuntimeError(f"WAV file is empty, cannot transcribe: {wav_audio_path}")

    logging.info("Starting transcription with Azure Speech for payload: %s", payload)
    try:
        result = azure_transcribe_audio(wav_audio_path)
        logging.info("Azure Speech transcription complete")
        return {**payload, **result}
    except Exception as exc:
        logging.exception(
            "Azure Speech transcription failed path=%s payload_id=%s: %s",
            wav_audio_path,
            _id,
            exc,
        )
        raise
