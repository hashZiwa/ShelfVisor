from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CREDENTIALS_DIR = ROOT / "credentials" / "google-vision"


def run_google_vision_ocr(image_bytes: bytes) -> dict[str, Any]:
    _load_local_env()
    client = _client(_credentials_path())

    try:
        from google.cloud import vision
    except ImportError as error:
        raise RuntimeError(
            "Google Vision OCR requires google-cloud-vision. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    response = client.document_text_detection(image=vision.Image(content=image_bytes))
    if response.error.message:
        raise RuntimeError(f"Google Vision OCR failed: {response.error.message}")

    annotations = []
    for annotation in response.text_annotations[1:]:
        box = _box_from_vertices(annotation.bounding_poly.vertices)
        annotations.append(
            {
                "text": annotation.description,
                "box": box,
            }
        )

    return {
        "fullText": response.full_text_annotation.text or "",
        "annotations": annotations,
    }


@lru_cache(maxsize=1)
def _client(credentials_path: str) -> Any:
    try:
        from google.cloud import vision
        from google.oauth2 import service_account
    except ImportError as error:
        raise RuntimeError(
            "Google Vision OCR requires google-cloud-vision. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    return vision.ImageAnnotatorClient(credentials=credentials)


def _credentials_path() -> str:
    configured_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Google Vision credentials not found: {path}")

    json_files = sorted(DEFAULT_CREDENTIALS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            "Google Vision service account JSON was not found. "
            "Place it under credentials/google-vision or set GOOGLE_APPLICATION_CREDENTIALS."
        )
    return str(json_files[0])


def _box_from_vertices(vertices: Any) -> list[int]:
    xs = [int(getattr(vertex, "x", 0) or 0) for vertex in vertices]
    ys = [int(getattr(vertex, "y", 0) or 0) for vertex in vertices]
    if not xs or not ys:
        return [0, 0, 0, 0]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def _load_local_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
