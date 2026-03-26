import os
import logging
import contextlib
import wave
import subprocess
import tempfile

from pydub import AudioSegment

AZURE_SPEECH_WAV_CHANNELS = 1
AZURE_SPEECH_WAV_FRAMERATE = 16000
AZURE_SPEECH_WAV_SAMPLE_WIDTH_BYTES = 2


def get_wav_duration(wav):
    try:
        with contextlib.closing(wave.open(wav, "r")) as fd:
            frames = fd.getnframes()
            rate = fd.getframerate()
            return frames / float(rate)
    except wave.Error as e:
        logging.debug(f"error occured while finding duration through wave {e}")

    try:
        sound = AudioSegment.from_file(wav)
        return sound.duration_seconds
    except Exception as e:
        return 0


def clean_and_return(duration, wav_file, file):
    # remove original file
    try:
        if wav_file != file:
            os.remove(file)
    except Exception as e:
        pass
    return {"success": True, "duration": duration, "output": wav_file}


def get_num_channels(wav):
    latest_exception = None
    try:
        s = AudioSegment.from_wav(wav)
        return s.channels
    except Exception as e:
        latest_exception = e

    try:
        s = AudioSegment.from_mp3(wav)
        return s.channels
    except Exception as e:
        latest_exception = e

    raise latest_exception


def _validate_wav_for_azure_speech(wav_path: str) -> None:
    """Require mono 16 kHz 16-bit PCM WAV. Raises ValueError or wave.Error if invalid."""
    with wave.open(wav_path, "rb") as wf:
        ch = wf.getnchannels()
        rate = wf.getframerate()
        sw = wf.getsampwidth()
        comptype = wf.getcomptype()

    logging.info(
        "WAV properties path=%s channels=%d framerate=%d sample_width_bytes=%s comptype=%s",
        wav_path,
        ch,
        rate,
        sw,
        comptype,
    )

    if ch != AZURE_SPEECH_WAV_CHANNELS:
        raise ValueError(
            f"WAV must be mono ({AZURE_SPEECH_WAV_CHANNELS} channel); got {ch}"
        )
    if rate != AZURE_SPEECH_WAV_FRAMERATE:
        raise ValueError(
            f"WAV must be {AZURE_SPEECH_WAV_FRAMERATE} Hz; got {rate}"
        )
    if sw != AZURE_SPEECH_WAV_SAMPLE_WIDTH_BYTES:
        raise ValueError(
            f"WAV must be 16-bit PCM ({AZURE_SPEECH_WAV_SAMPLE_WIDTH_BYTES}-byte samples); got {sw}"
        )


def convert_to_wav(audio_path: str):
    temp_dir = tempfile.gettempdir()

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    input_file = audio_path
    logging.info("Audio conversion input path: %s", input_file)
    logging.info("Audio conversion temp directory: %s", temp_dir)

    try:
        input_size = os.path.getsize(input_file)
    except OSError as exc:
        logging.error("Audio conversion could not stat input: %s", exc)
        return {"success": False, "msg": f"Cannot read input file size: {exc}"}

    logging.info("Audio conversion input size_bytes: %d", input_size)
    if input_size <= 0:
        logging.error("Audio conversion rejected empty input file: %s", input_file)
        return {"success": False, "msg": "Input audio file is empty"}

    filename = os.path.basename(audio_path)
    base_file, _ext = os.path.splitext(filename)
    out_file = os.path.join(temp_dir, f"{base_file}_azure_speech.wav")
    logging.info("Audio conversion output wav path: %s", out_file)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-ac",
        str(AZURE_SPEECH_WAV_CHANNELS),
        "-ar",
        str(AZURE_SPEECH_WAV_FRAMERATE),
        "-sample_fmt",
        "s16",
        "-c:a",
        "pcm_s16le",
        out_file,
    ]
    logging.info("Audio conversion ffmpeg command: %s", " ".join(command))

    completed_process = subprocess.run(command, capture_output=True, text=True)
    if completed_process.returncode != 0:
        logging.error(
            "ffmpeg failed returncode=%s stderr=%s stdout=%s",
            completed_process.returncode,
            completed_process.stderr.strip(),
            completed_process.stdout.strip(),
        )
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
            except OSError:
                pass
        try:
            if os.path.exists(input_file):
                os.remove(input_file)
        except OSError:
            pass
        return {
            "success": False,
            "msg": (
                "ffmpeg could not produce Azure Speech WAV. "
                f"stderr: {completed_process.stderr.strip()}"
            ),
        }

    if not os.path.isfile(out_file):
        logging.error("ffmpeg reported success but output missing: %s", out_file)
        return {"success": False, "msg": "Conversion produced no output WAV file"}

    try:
        output_size = os.path.getsize(out_file)
    except OSError as exc:
        logging.error("Could not stat output wav: %s", exc)
        return {"success": False, "msg": f"Cannot read output WAV size: {exc}"}

    logging.info("Audio conversion output size_bytes: %d path=%s", output_size, out_file)
    if output_size <= 0:
        logging.error("Converted WAV is empty: %s", out_file)
        try:
            os.remove(out_file)
        except OSError:
            pass
        return {"success": False, "msg": "Converted WAV file is empty"}

    try:
        _validate_wav_for_azure_speech(out_file)
    except (ValueError, wave.Error, EOFError) as exc:
        logging.error(
            "WAV validation failed after conversion path=%s error=%s",
            out_file,
            exc,
            exc_info=True,
        )
        try:
            os.remove(out_file)
        except OSError:
            pass
        return {"success": False, "msg": f"Invalid WAV after conversion: {exc}"}

    duration = None
    return clean_and_return(duration, wav_file=out_file, file=input_file)
