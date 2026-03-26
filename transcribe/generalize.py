import json

from utils import upload_blob_stream


def generalize_transcript(transcript_output: dict) -> dict:
    transcript_json = transcript_output.get("json_response")
    transcript_text = transcript_output.get("text")
    force = transcript_output.get("force")

    # don't generalize, use cached data instead
    if isinstance(transcript_json, dict) and (
        transcript_json.get("generalized") and not force
    ):
        return transcript_output

    # generalized transcript for final use in UI
    generalized_transcript_json = {
        "utterances": transcript_json.get("utterances", []) if isinstance(transcript_json, dict) else [],
        "text": transcript_text,
        "wordcloud": [],
        "generalized": True,
    }

    transcript_output["json_response"] = generalized_transcript_json

    # upload generalized transcript for caching
    upload_blob_stream(
        json.dumps(generalized_transcript_json, indent=2),
        f"{transcript_output.get('_id')}_transcript.json",
        "transcripts",
    )

    return transcript_output
