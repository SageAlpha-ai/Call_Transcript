import json

from typing import List
from .azure_openai import (
    create_chat_completion,
    get_azure_openai_client,
    get_azure_openai_deployment,
    parse_json_content,
)


def get_transcript_summary(transcript_text: str) -> List[str]:
    """Generate transcript summary using Azure OpenAI."""
    client = get_azure_openai_client()
    deployment = get_azure_openai_deployment()

    example_data_schema = {
        "summary": [],
    }
    system_content = (
        "You are expert in analyzing and providing the following,\n"
        "1. Summary of the given conversation in 5 points\n\n"
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
