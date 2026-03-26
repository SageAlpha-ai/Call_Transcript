from __future__ import annotations

import os
import io
import logging

from typing import Any, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


## AZURE UTILS
def _get_blob_service_client() -> Any:
    from azure.storage.blob import BlobServiceClient

    if not AZURE_STORAGE_CONNECTION_STRING:
        raise RuntimeError(
            "Missing required environment variable: AZURE_STORAGE_CONNECTION_STRING"
        )
    if not AZURE_STORAGE_CONTAINER_NAME:
        raise RuntimeError(
            "Missing required environment variable: AZURE_STORAGE_CONTAINER_NAME"
        )
    logging.info(
        "Azure Storage env check: connection_string=%s container_name=%s",
        "set" if AZURE_STORAGE_CONNECTION_STRING else "missing",
        "set" if AZURE_STORAGE_CONTAINER_NAME else "missing",
    )
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


def upload_blob_stream(
    input_stream,
    file_name,
    folder_name,
    blob_service_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Uploads data stream to Azure blob storage using block-blob"""
    if not blob_service_client:
        blob_service_client = _get_blob_service_client()

    blob_client = blob_service_client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME,
        blob=f"{folder_name}/{file_name}",
    )
    blob_client.upload_blob(input_stream, blob_type="BlockBlob", overwrite=True)
    return blob_client.url


def check_blob_exists(folder_name: str, file_name: str) -> bool:
    """Returns True if a blob exists with the defined parameters, and returns False otherwise."""
    blob_service_client = _get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME,
        blob=f"{folder_name}/{file_name}",
    )
    return blob_client.exists()


def download_blob_to_stream(folder_name: str, file_name: str) -> io.BytesIO:
    """Downloads blob and returns a stream"""
    blob_service_client = _get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME, blob=f"{folder_name}/{file_name}"
    )

    downloader = blob_client.download_blob(max_concurrency=1)
    blob_text = downloader.readall()
    return blob_text


def get_blob_url(folder_name: str, file_name: str) -> io.BytesIO:
    """Downloads blob and returns a stream"""
    blob_service_client = _get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME, blob=f"{folder_name}/{file_name}"
    )
    return blob_client.url


def download_blob_to_file(blob_name: str, local_path: str) -> str:
    """Downloads a blob object path to a local file path."""
    blob_service_client = _get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=AZURE_STORAGE_CONTAINER_NAME, blob=blob_name
    )
    downloader = blob_client.download_blob(max_concurrency=1)
    with open(local_path, "wb") as fd:
        fd.write(downloader.readall())
    return local_path
