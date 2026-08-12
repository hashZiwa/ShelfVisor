from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CREDENTIALS_DIR = ROOT / "credentials" / "google-vision"


def run_ocr_inference(image_bytes: bytes) -> dict[str, Any]:
    _load_local_env()
    client = _client(_credentials_path())
    try:
        from google.cloud import vision
    except ImportError as error:
        raise RuntimeError("Google Vision OCR requires google-cloud-vision. Install project dependencies with `python -m pip install -r requirements.txt`.") from error
    response = client.document_text_detection(image=vision.Image(content=image_bytes))
    if response.error.message:
        raise RuntimeError(f"Google Vision OCR failed: {response.error.message}")
    annotations = _annotations_from_full_text(response.full_text_annotation)
    if not annotations:
        annotations = [{
            "text": annotation.description,
            "box": _box_from_vertices(annotation.bounding_poly.vertices),
            "confidence": _annotation_confidence(annotation),
        } for annotation in response.text_annotations[1:]]
    return {"fullText": response.full_text_annotation.text or "", "annotations": annotations}


def _annotations_from_full_text(full_text_annotation: Any) -> list[dict[str, Any]]:
    annotations = []
    for page in getattr(full_text_annotation, "pages", []):
        for block in getattr(page, "blocks", []):
            for paragraph in getattr(block, "paragraphs", []):
                for word in getattr(paragraph, "words", []):
                    text = "".join(getattr(symbol, "text", "") for symbol in getattr(word, "symbols", [])).strip()
                    if text:
                        annotations.append({
                            "text": text,
                            "box": _box_from_vertices(word.bounding_box.vertices),
                            "confidence": float(getattr(word, "confidence", 0.0) or 0.0),
                        })
    return annotations


def _annotation_confidence(annotation: Any) -> float:
    for name in ("confidence", "score"):
        value = getattr(annotation, name, None)
        if value is not None:
            return float(value)
    return 0.0


@lru_cache(maxsize=1)
def _client(credentials_path: str) -> Any:
    try:
        from google.cloud import vision
        from google.oauth2 import service_account
    except ImportError as error:
        raise RuntimeError("Google Vision OCR requires google-cloud-vision. Install project dependencies with `python -m pip install -r requirements.txt`.") from error
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    return vision.ImageAnnotatorClient(credentials=credentials)


def _credentials_path() -> str:
    configured = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if configured:
        path = Path(configured)
        path = path if path.is_absolute() else ROOT / path
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Google Vision credentials not found: {path}")
    json_files = sorted(DEFAULT_CREDENTIALS_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("Google Vision service account JSON was not found. Place it under credentials/google-vision or set GOOGLE_APPLICATION_CREDENTIALS.")
    return str(json_files[0])


def _box_from_vertices(vertices: Any) -> list[int]:
    xs = [int(getattr(vertex, "x", 0) or 0) for vertex in vertices]
    ys = [int(getattr(vertex, "y", 0) or 0) for vertex in vertices]
    if not xs or not ys:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [x1, y1, x2 - x1, y2 - y1]


def _load_local_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
