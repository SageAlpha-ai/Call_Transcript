import json

from typing import List, Dict, Any
from .azure_openai import (
    create_chat_completion,
    get_azure_openai_client,
    get_azure_openai_deployment,
    parse_json_content,
)


def calc_sentiment(transcript_text: str) -> Dict[str, float]:
    """Calculate sentiment for transcript text using Azure OpenAI."""
    client = get_azure_openai_client()
    deployment = get_azure_openai_deployment()

    example_data_schema = {
        "sentiment_score": 0,
    }
    system_content = (
        "You are expert in analyzing and providing the following,\n"
        "1. Sentiment of given text on a scale from -1 to 1\n\n"
        "Trim the input text if it exceedes the model context length."
        f"Generate JSON in schema: {json.dumps(example_data_schema)}"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Analyze the following: {transcript_text}"},
    ]
    completion = create_chat_completion(
        client,
        model=deployment,
        messages=messages,
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
    )
    return parse_json_content(completion.choices[0].message.content)


def calc_sentence_sentiment(transcript_utterances: List[Dict[str, Any]]):
    """Calculate sentiment of each sentences by looping over the list and calling ChatGPT API"""

    for transcript_dict in transcript_utterances:
        if transcript_dict.get("sentiment_score") is not None:
            continue

        transcript_text = transcript_dict["text"]
        transcript_sentiment = calc_sentiment(transcript_text)
        if isinstance(transcript_sentiment, dict):
            transcript_sentiment = transcript_sentiment.get("sentiment_score")
            transcript_dict["sentiment_score"] = transcript_sentiment

    return transcript_utterances


def detect_intent(transcript_text: str) -> Dict[str, Any]:
    """Extract intent labels from transcript text using Azure OpenAI."""
    client = get_azure_openai_client()
    deployment = get_azure_openai_deployment()

    schema = {
        "intents": [],
    }
    system_content = (
        "You analyze customer call transcripts and return concise intent labels.\n"
        "Return 1 to 5 intents as short snake_case strings.\n"
        f"Generate JSON in schema: {json.dumps(schema)}"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Transcript: {transcript_text}"},
    ]
    completion = create_chat_completion(
        client,
        model=deployment,
        messages=messages,
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
    )
    return parse_json_content(completion.choices[0].message.content)
