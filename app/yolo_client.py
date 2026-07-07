from __future__ import annotations

import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "yolo" / "yolo-model-v1.pt"
DEFAULT_CONFIDENCE = 0.15
DEFAULT_IMAGE_SIZE = 1024


def infer_book_spines(image_bytes: bytes) -> dict[str, Any]:
    _load_local_env()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    model_path = _model_path()
    model = _load_model(str(model_path))
    confidence = _float_env("LOCAL_YOLO_CONFIDENCE", DEFAULT_CONFIDENCE)
    image_size = int(_float_env("LOCAL_YOLO_IMAGE_SIZE", DEFAULT_IMAGE_SIZE))

    results = model.predict(
        source=np.asarray(image),
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )
    predictions = _predictions_from_results(results, model)

    return {
        "predictions": predictions,
        "model_id": _display_model_path(model_path),
        "source": "local_ultralytics",
        "image": {"width": image.width, "height": image.height},
    }


@lru_cache(maxsize=1)
def _load_model(model_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Local YOLO inference requires ultralytics. "
            "Install project dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Local YOLO model not found: {path}")
    return YOLO(str(path))


def _predictions_from_results(results: Any, model: Any) -> list[dict[str, Any]]:
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    xywh = boxes.xywh.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    predictions = []

    for index, (center_x, center_y, width, height) in enumerate(xywh):
        class_index = int(classes[index])
        confidence = float(confidences[index])
        x1 = float(center_x - width / 2)
        y1 = float(center_y - height / 2)
        x2 = float(center_x + width / 2)
        y2 = float(center_y + height / 2)
        predictions.append(
            {
                "x": float(center_x),
                "y": float(center_y),
                "width": float(width),
                "height": float(height),
                "confidence": confidence,
                "class": str(names.get(class_index, class_index)),
                "class_id": class_index,
                "points": [
                    {"x": x1, "y": y1},
                    {"x": x2, "y": y1},
                    {"x": x2, "y": y2},
                    {"x": x1, "y": y2},
                ],
            }
        )

    predictions.sort(key=lambda prediction: (prediction["x"], prediction["y"]))
    return predictions


def _model_path() -> Path:
    configured_path = os.environ.get("LOCAL_YOLO_MODEL_PATH")
    if not configured_path:
        return DEFAULT_MODEL_PATH

    path = Path(configured_path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _display_model_path(model_path: Path) -> str:
    try:
        return str(model_path.relative_to(ROOT))
    except ValueError:
        return str(model_path)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
