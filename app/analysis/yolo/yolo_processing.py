from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..analysis_models import DetectedRegion
from ..image_processing import clamp, clamp_float


DEFAULT_FILTER_OPTIONS = {"maxBoxWidthRatio": 0.33, "maxBoxAreaRatio": 0.10}


def normalize_filter_options(options: Mapping[str, Any] | None) -> dict[str, float]:
    normalized = dict(DEFAULT_FILTER_OPTIONS)
    if options:
        normalized.update(options)
    normalized["maxBoxWidthRatio"] = clamp_float(float(normalized["maxBoxWidthRatio"]), 0.01, 1.0)
    normalized["maxBoxAreaRatio"] = clamp_float(float(normalized["maxBoxAreaRatio"]), 0.01, 1.0)
    return normalized


def parse_yolo_predictions(result: Mapping[str, Any], image_size: tuple[int, int]) -> list[DetectedRegion]:
    width, height = image_size
    regions = []
    for prediction in result.get("predictions", []):
        region = _region_from_prediction(prediction, width, height)
        if region is not None:
            regions.append(region)
    regions.sort(key=lambda item: (item.box[0] + item.box[2] / 2, item.box[1]))
    return regions[:80]


def filter_yolo_regions(
    regions: Sequence[DetectedRegion],
    image_size: tuple[int, int],
    options: Mapping[str, float],
) -> list[DetectedRegion]:
    image_width, image_height = image_size
    max_width = image_width * options["maxBoxWidthRatio"]
    max_area = image_width * image_height * options["maxBoxAreaRatio"]
    return [region for region in regions if region.box[2] < max_width and region.box[2] * region.box[3] < max_area]


def _region_from_prediction(
    prediction: Mapping[str, Any],
    image_width: int,
    image_height: int,
) -> DetectedRegion | None:
    polygon = _polygon_from_prediction(prediction)
    if not polygon:
        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))
        width = float(prediction.get("width", 0))
        height = float(prediction.get("height", 0))
        if width <= 0 or height <= 0:
            return None
        polygon = [
            [round(x - width / 2), round(y - height / 2)],
            [round(x + width / 2), round(y - height / 2)],
            [round(x + width / 2), round(y + height / 2)],
            [round(x - width / 2), round(y + height / 2)],
        ]
    polygon = [[clamp(int(x), 0, image_width), clamp(int(y), 0, image_height)] for x, y in polygon]
    box = _box_from_polygon(polygon)
    if box[2] <= 0 or box[3] <= 0:
        return None
    return DetectedRegion(
        box=box,
        polygon=polygon,
        confidence=_optional_float(prediction.get("confidence")),
        class_name=prediction.get("class"),
        raw=dict(prediction),
    )


def _polygon_from_prediction(prediction: Mapping[str, Any]) -> list[list[int]]:
    points = prediction.get("points")
    if not isinstance(points, list) or len(points) < 3:
        return []
    polygon = []
    for point in points:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            return []
        polygon.append([round(float(point["x"])), round(float(point["y"]))])
    return polygon


def _box_from_polygon(polygon: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x1, y1, x2 - x1, y2 - y1


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
