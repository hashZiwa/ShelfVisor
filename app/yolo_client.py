from __future__ import annotations

import os
import json
import base64
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_MODEL_ID = "book-spine-detection-2cci9/2"


def infer_book_spines(image_bytes: bytes) -> dict[str, Any]:
    _load_local_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY is not set.")

    api_url = os.environ.get("ROBOFLOW_API_URL", ROBOFLOW_API_URL).rstrip("/")
    model_id = os.environ.get("ROBOFLOW_MODEL_ID", ROBOFLOW_MODEL_ID)
    project_id, version = _split_model_id(model_id)
    query = urlencode({"api_key": api_key})
    url = f"{api_url}/{project_id}/{version}?{query}"
    payload = base64.b64encode(image_bytes)
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Roboflow API error {error.code}: {_redact_api_key(body, api_key)}") from error
    except URLError as error:
        raise RuntimeError(f"Roboflow API connection failed: {error.reason}") from error


def _split_model_id(model_id: str) -> tuple[str, str]:
    chunks = model_id.split("/")
    if len(chunks) != 2 or not chunks[0] or not chunks[1]:
        raise ValueError("ROBOFLOW_MODEL_ID must look like project/version.")
    return chunks[0], chunks[1]


def _redact_api_key(value: str, api_key: str) -> str:
    return value.replace(api_key, "***")


def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
