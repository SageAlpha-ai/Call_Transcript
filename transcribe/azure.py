import logging
import os
import time
from types import ModuleType

from dotenv import load_dotenv

from metrics.retry_utils import retry_sync

load_dotenv()

_DEFAULT_LANG = "en-US"


def _speech_sdk() -> ModuleType:
    import azure.cognitiveservices.speech as speechsdk

    return speechsdk


def transcribe_audio(audio_path: str, **kwargs):
    def _run_once():
        sdk = _speech_sdk()
        audio_ext = os.path.splitext(audio_path)[1].lower()
        logging.info("Speech step input file: %s (extension: %s)", audio_path, audio_ext)
        if audio_ext != ".wav":
            raise RuntimeError("Only WAV format supported. Please convert audio before processing.")

        speech_key = os.getenv("AZURE_SPEECH_API_KEY")
        speech_region = os.getenv("AZURE_SPEECH_API_REGION")
        logging.info(
            "Speech env check: AZURE_SPEECH_API_KEY=%s AZURE_SPEECH_API_REGION=%s",
            "set" if speech_key else "missing",
            "set" if speech_region else "missing",
        )
        if not speech_key:
            raise RuntimeError("Missing required environment variable: AZURE_SPEECH_API_KEY")
        if not speech_region:
            raise RuntimeError("Missing required environment variable: AZURE_SPEECH_API_REGION")

        try:
            lang = (kwargs.get("lang") or "").strip() or None
            if not lang:
                lang = _detect_language_once(sdk, speech_key, speech_region, audio_path)

            speech_config = _speech_config_with_language(sdk, speech_key, speech_region, lang)
            effective_lang = speech_config.speech_recognition_language
            logging.info("Azure Speech transcription using language=%s", effective_lang)

            audio_config = sdk.audio.AudioConfig(filename=audio_path)
            conversation_transcriber = sdk.transcription.ConversationTranscriber(
                speech_config=speech_config, audio_config=audio_config
            )
            done = False

            def stop_cb(evt):
                logging.debug("CLOSING on {}".format(evt))
                conversation_transcriber.stop_transcribing_async()
                nonlocal done
                done = True

            all_results = []

            def handle_transcribed_event(evt):
                all_results.append(evt.result)

            conversation_transcriber.transcribed.connect(handle_transcribed_event)
            conversation_transcriber.session_started.connect(
                lambda evt: logging.debug("SESSION STARTED: {}".format(evt))
            )
            conversation_transcriber.session_stopped.connect(
                lambda evt: logging.debug("SESSION STOPPED {}".format(evt))
            )
            conversation_transcriber.canceled.connect(
                lambda evt: logging.debug("CANCELED {}".format(evt))
            )

            conversation_transcriber.session_stopped.connect(stop_cb)
            conversation_transcriber.canceled.connect(stop_cb)

            conversation_transcriber.start_transcribing_async()
            while not done:
                time.sleep(0.5)

            utterances = []
            for result in all_results:
                speaker_label = getattr(result, "speaker_id", None) or "unknown"
                utterances.append(
                    {
                        "text": result.text or "",
                        "start": result.offset / 10000.0,
                        "end": (result.offset + result.duration) / 10000.0,
                        "speaker": speaker_label,
                    }
                )

            full_text = "\n".join((r.text or "").strip() for r in all_results if r and (r.text or "").strip())

            if not full_text.strip():
                logging.warning(
                    "Azure Speech ConversationTranscriber produced no text; trying continuous recognition"
                )
                full_text = _continuous_recognize(sdk, speech_key, speech_region, effective_lang, audio_path)
                if full_text.strip():
                    utterances = [
                        {
                            "text": full_text.strip(),
                            "start": 0.0,
                            "end": 0.0,
                            "speaker": "unknown",
                        }
                    ]

            full_text = full_text.strip()
            if not full_text:
                logging.warning("Azure Speech: transcript is empty after recognition")

            payload = {
                "transcript": full_text,
                "text": full_text,
                "json_response": {
                    "text": full_text,
                    "utterances": utterances,
                    "generalized": True,
                },
            }
            return payload
        except Exception as exc:
            logging.exception(
                "Azure Speech transcription failed for file=%s: %s",
                audio_path,
                exc,
            )
            raise

    return retry_sync(_run_once, operation="Azure Speech transcribe_audio")


def _speech_config_with_language(
    sdk: ModuleType, subscription: str, region: str, language: str
):
    cfg = sdk.SpeechConfig(subscription=subscription, region=region)
    lang = (language or "").strip() or _DEFAULT_LANG
    try:
        cfg.speech_recognition_language = lang
    except Exception as exc:
        logging.warning(
            "speech_recognition_language=%r rejected (%s); using %s",
            lang,
            exc,
            _DEFAULT_LANG,
        )
        cfg = sdk.SpeechConfig(subscription=subscription, region=region)
        cfg.speech_recognition_language = _DEFAULT_LANG
    return cfg


def _detect_language_once(sdk: ModuleType, subscription: str, region: str, audio_path: str) -> str:
    speech_config = sdk.SpeechConfig(subscription=subscription, region=region)
    auto_detect_source_language_config = sdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "en-IN", "hi-IN"]
    )
    audio_config = sdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = sdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config,
    )
    recognize_result = speech_recognizer.recognize_once()

    reason = recognize_result.reason
    sample_text = recognize_result.text or ""
    logging.info(
        "Azure Speech autodetect recognize_once: reason=%s text_len=%d",
        reason,
        len(sample_text),
    )
    if sample_text:
        logging.info("Azure Speech autodetect text (truncated): %s", sample_text[:240])

    if reason == sdk.ResultReason.RecognizedSpeech:
        adr = sdk.AutoDetectSourceLanguageResult(recognize_result)
        detected = (adr.language or "").strip()
        if detected:
            return detected
        logging.warning("Azure Speech autodetect: RecognizedSpeech but no language; using %s", _DEFAULT_LANG)
        return _DEFAULT_LANG

    if reason == sdk.ResultReason.NoMatch:
        nm = recognize_result.no_match_details
        if nm is not None:
            logging.warning("Azure Speech autodetect NoMatch: %s", nm.reason)
        else:
            logging.warning("Azure Speech autodetect NoMatch (no details)")
        return _DEFAULT_LANG

    if reason == sdk.ResultReason.Canceled:
        cancel = recognize_result.cancellation_details
        if cancel is not None:
            logging.error(
                "Azure Speech autodetect Canceled: reason=%s error_details=%s",
                cancel.reason,
                cancel.error_details,
            )
        else:
            logging.error("Azure Speech autodetect Canceled (no details)")
        return _DEFAULT_LANG

    logging.warning("Azure Speech autodetect unexpected reason=%s; using %s", reason, _DEFAULT_LANG)
    return _DEFAULT_LANG


def _continuous_recognize(
    sdk: ModuleType, subscription: str, region: str, language: str, audio_path: str
) -> str:
    speech_config = _speech_config_with_language(sdk, subscription, region, language)
    audio_config = sdk.audio.AudioConfig(filename=audio_path)
    recognizer = sdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    parts: list[str] = []
    done = False

    def on_recognized(evt):
        res = evt.result
        logging.info(
            "Azure Speech continuous recognized: reason=%s text_len=%d",
            res.reason,
            len(res.text or ""),
        )
        if res.reason == sdk.ResultReason.RecognizedSpeech and (res.text or "").strip():
            parts.append(res.text.strip())

    def on_canceled(evt):
        logging.error("Azure Speech continuous canceled: %s", evt)
        nonlocal done
        done = True

    def on_session_stopped(evt):
        logging.debug("Azure Speech continuous session_stopped: %s", evt)
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(on_session_stopped)

    recognizer.start_continuous_recognition_async().get()
    deadline = time.monotonic() + 7200.0
    while not done and time.monotonic() < deadline:
        time.sleep(0.25)
    try:
        recognizer.stop_continuous_recognition_async().get()
    except Exception as exc:
        logging.debug("Azure Speech continuous stop: %s", exc)

    return "\n".join(parts)
