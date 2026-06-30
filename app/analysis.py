from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .yolo_client import infer_book_spines
except ImportError:
    from yolo_client import infer_book_spines


@dataclass
class Spine:
    index: int
    x: int
    y: int
    width: int
    height: int
    call_number: str
    expected_rank: int
    status: str
    polygon: list[list[int]]
    confidence: float | None
    class_name: str | None


DEFAULT_REFINEMENT_OPTIONS = {
    "boxPaddingX": 0.12,
    "boxPaddingY": 0.00,
    "edgeWeight": 0.45,
    "colorWeight": 0.45,
    "houghWeight": 0.10,
    "searchZoneRatio": 0.34,
    "minSpineWidth": 18,
    "maxSkew": 0.22,
    "confidenceThreshold": 0.15,
}


def analyze_shelf_photo(
    image_bytes: bytes,
    include_debug: bool = False,
    refinement_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = _fit_image(image, max_side=1400)
    model_bytes = _image_to_jpeg_bytes(display_image)
    roboflow_result = infer_book_spines(model_bytes)
    options = _normalize_refinement_options(refinement_options)
    yolo_regions = _regions_from_roboflow(roboflow_result, display_image.size, options["confidenceThreshold"])
    regions = _refine_regions_with_opencv(display_image, yolo_regions, options)

    call_numbers = mock_ocr_call_numbers(len(regions))
    statuses = check_call_number_order(call_numbers)

    spines = [
        Spine(
            index=i + 1,
            x=region["box"][0],
            y=region["box"][1],
            width=region["box"][2],
            height=region["box"][3],
            call_number=call_numbers[i],
            expected_rank=statuses[i]["expected_rank"],
            status=statuses[i]["status"],
            polygon=region["polygon"],
            confidence=region["confidence"],
            class_name=region["class_name"],
        )
        for i, region in enumerate(regions)
    ]

    annotated = draw_annotation(display_image, spines)
    result = {
        "summary": {
            "bookCount": len(spines),
            "misplacedCount": len([spine for spine in spines if spine.status != "ok"]),
            "status": "needs_review" if any(spine.status != "ok" for spine in spines) else "ok",
            "detector": "roboflow_yolo",
            "refinement": "opencv",
        },
        "spines": [spine.__dict__ for spine in spines],
        "annotatedImage": image_to_data_url(annotated),
    }

    if include_debug:
        result["debug"] = _build_debug_payload(display_image, spines, yolo_regions, roboflow_result, options)

    return result


def detect_book_spines(image: Image.Image) -> list[tuple[int, int, int, int]]:
    model_bytes = _image_to_jpeg_bytes(image.convert("RGB"))
    roboflow_result = infer_book_spines(model_bytes)
    options = _normalize_refinement_options(None)
    regions = _regions_from_roboflow(roboflow_result, image.size, options["confidenceThreshold"])
    return [region["box"] for region in _refine_regions_with_opencv(image.convert("RGB"), regions, options)]


def mock_ocr_call_numbers(count: int) -> list[str]:
    labels = [f"811.{120 + i * 7} K{i + 1:02d}" for i in range(count)]
    if count >= 5:
        labels[2], labels[3] = labels[3], labels[2]
    return labels


def check_call_number_order(call_numbers: list[str]) -> list[dict[str, Any]]:
    sorted_labels = sorted(call_numbers, key=call_number_sort_key)
    expected_rank = {label: rank + 1 for rank, label in enumerate(sorted_labels)}

    return [
        {
            "label": label,
            "expected_rank": expected_rank[label],
            "status": "ok" if expected_rank[label] == index + 1 else "misplaced",
        }
        for index, label in enumerate(call_numbers)
    ]


def call_number_sort_key(value: str) -> tuple[Any, ...]:
    parts = re.findall(r"\d+\.\d+|\d+|[A-Za-z]+", value.upper())
    key: list[Any] = []
    for part in parts:
        if re.fullmatch(r"\d+\.\d+|\d+", part):
            key.append((0, float(part)))
        else:
            key.append((1, part))
    return tuple(key)


def draw_annotation(image: Image.Image, spines: list[Spine]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()

    for spine in spines:
        is_ok = spine.status == "ok"
        color = (43, 156, 94, 235) if is_ok else (220, 70, 65, 240)
        fill = (43, 156, 94, 40) if is_ok else (220, 70, 65, 65)
        polygon = [tuple(point) for point in spine.polygon]
        draw.polygon(polygon, outline=color, fill=fill)
        draw.line(polygon + [polygon[0]], fill=color, width=4)

        confidence = "" if spine.confidence is None else f" {spine.confidence:.2f}"
        label = f"{spine.index}. {spine.call_number}{confidence}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0] + 12
        label_height = label_box[3] - label_box[1] + 10
        label_x = max(0, min(spine.x + 4, output.width - label_width))
        label_y = max(0, spine.y - label_height - 4)
        if label_y < 4:
            label_y = spine.y + 4
        draw.rounded_rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            radius=4,
            fill=(17, 24, 39, 225),
        )
        draw.text((label_x + 6, label_y + 5), label, font=font, fill=(255, 255, 255, 255))

    return output


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _regions_from_roboflow(
    result: dict[str, Any],
    image_size: tuple[int, int],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    predictions = result.get("predictions", [])
    regions = []
    for prediction in predictions:
        region = _region_from_prediction(prediction, image_width, image_height)
        if region is not None and (region["confidence"] is None or region["confidence"] >= confidence_threshold):
            regions.append(region)

    regions.sort(key=lambda region: (region["box"][0] + region["box"][2] / 2, region["box"][1]))
    return regions[:80]


def _refine_regions_with_opencv(
    image: Image.Image,
    regions: list[dict[str, Any]],
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    rgb = np.asarray(image.convert("RGB"))
    refined = []
    for region in regions:
        refined.append(_refine_region_with_opencv(rgb, region, options))
    refined.sort(key=lambda item: (item["box"][0] + item["box"][2] / 2, item["box"][1]))
    return refined


def _refine_region_with_opencv(
    rgb: np.ndarray,
    region: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    image_height, image_width = rgb.shape[:2]
    x, y, width, height = region["box"]
    pad_x = max(3, int(width * options["boxPaddingX"]))
    pad_y = max(2, int(height * options["boxPaddingY"]))
    crop_x1 = _clamp(x - pad_x, 0, image_width - 1)
    crop_y1 = _clamp(y - pad_y, 0, image_height - 1)
    crop_x2 = _clamp(x + width + pad_x, crop_x1 + 1, image_width)
    crop_y2 = _clamp(y + height + pad_y, crop_y1 + 1, image_height)
    crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_height, crop_width = crop.shape[:2]
    if crop_width < 8 or crop_height < 20:
        return region

    score = _opencv_boundary_score(crop, options)
    hough_mask = _hough_column_mask(crop)
    if hough_mask.max() > 0:
        score = score + hough_mask * float(options["houghWeight"])

    left_line = _fit_side_line(score, side="left", options=options)
    right_line = _fit_side_line(score, side="right", options=options)
    if left_line is None or right_line is None:
        return region

    left_top, left_bottom = left_line
    right_top, right_bottom = right_line
    min_width = int(options["minSpineWidth"])
    if min(right_top - left_top, right_bottom - left_bottom) < min_width:
        return region

    polygon = [
        [_clamp(crop_x1 + left_top, 0, image_width), crop_y1],
        [_clamp(crop_x1 + right_top, 0, image_width), crop_y1],
        [_clamp(crop_x1 + right_bottom, 0, image_width), crop_y2],
        [_clamp(crop_x1 + left_bottom, 0, image_width), crop_y2],
    ]
    refined = dict(region)
    refined["polygon"] = polygon
    refined["box"] = _box_from_polygon(polygon)
    refined["refined"] = True
    return refined


def _opencv_boundary_score(crop: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    gray_score = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    color_score = np.zeros_like(gray_score)
    for channel in cv2.split(lab):
        color_score = np.maximum(color_score, np.abs(cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)))

    score = (
        _normalize_array(gray_score) * float(options["edgeWeight"])
        + _normalize_array(color_score) * float(options["colorWeight"])
    )
    return cv2.GaussianBlur(score, (5, 5), 0)


def _hough_column_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(16, crop.shape[0] // 7),
        minLineLength=max(18, int(crop.shape[0] * 0.35)),
        maxLineGap=max(6, crop.shape[0] // 18),
    )
    mask = np.zeros(edges.shape, dtype=np.float32)
    if lines is None:
        return mask

    for x1, y1, x2, y2 in lines[:, 0]:
        dy = y2 - y1
        if abs(dy) < 10:
            continue
        slope = (x2 - x1) / dy
        if abs(slope) > 0.5:
            continue
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1.0, 3)
    return cv2.GaussianBlur(mask, (7, 7), 0)


def _fit_side_line(
    score: np.ndarray,
    side: str,
    options: dict[str, Any],
) -> tuple[int, int] | None:
    height, width = score.shape
    zone = max(4, int(width * float(options["searchZoneRatio"])))
    if side == "left":
        x_offset = 0
        zone_slice = slice(0, zone)
    else:
        x_offset = width - zone
        zone_slice = slice(x_offset, width)

    points = []
    band_count = 7
    for index in range(band_count):
        y1 = int(index * height / band_count)
        y2 = int((index + 1) * height / band_count)
        band = score[y1:y2, zone_slice]
        if band.size == 0:
            continue
        column_score = band.mean(axis=0)
        x = int(np.argmax(column_score)) + x_offset
        y = (y1 + y2) // 2
        points.append((x, y))

    if len(points) < 2:
        return None

    ys = np.array([point[1] for point in points], dtype=np.float32)
    xs = np.array([point[0] for point in points], dtype=np.float32)
    slope, intercept = np.polyfit(ys, xs, 1)
    max_delta = float(options["maxSkew"]) * height
    top = float(intercept)
    bottom = float(slope * height + intercept)
    midpoint = (top + bottom) / 2
    top = _clamp_float(top, midpoint - max_delta / 2, midpoint + max_delta / 2)
    bottom = _clamp_float(bottom, midpoint - max_delta / 2, midpoint + max_delta / 2)
    return _clamp(int(round(top)), 0, width), _clamp(int(round(bottom)), 0, width)


def _region_from_prediction(prediction: dict[str, Any], image_width: int, image_height: int) -> dict[str, Any] | None:
    polygon = _polygon_from_prediction(prediction)
    if not polygon:
        x = float(prediction.get("x", 0))
        y = float(prediction.get("y", 0))
        width = float(prediction.get("width", 0))
        height = float(prediction.get("height", 0))
        if width <= 0 or height <= 0:
            return None
        x1 = int(round(x - width / 2))
        y1 = int(round(y - height / 2))
        x2 = int(round(x + width / 2))
        y2 = int(round(y + height / 2))
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    polygon = [[_clamp(point[0], 0, image_width), _clamp(point[1], 0, image_height)] for point in polygon]
    box = _box_from_polygon(polygon)
    if box[2] <= 0 or box[3] <= 0:
        return None

    return {
        "box": box,
        "polygon": polygon,
        "confidence": _to_optional_float(prediction.get("confidence")),
        "class_name": prediction.get("class"),
        "raw": prediction,
    }


def _polygon_from_prediction(prediction: dict[str, Any]) -> list[list[int]]:
    points = prediction.get("points")
    if not isinstance(points, list) or len(points) < 3:
        return []
    polygon = []
    for point in points:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            return []
        polygon.append([int(round(float(point["x"]))), int(round(float(point["y"])))])
    return polygon


def _box_from_polygon(polygon: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return x1, y1, x2 - x1, y2 - y1


def _build_debug_payload(
    image: Image.Image,
    spines: list[Spine],
    yolo_regions: list[dict[str, Any]],
    roboflow_result: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    predictions = roboflow_result.get("predictions", [])
    return {
        "usedFallback": False,
        "boundaryCount": len(predictions),
        "boxCount": len(spines),
        "model": roboflow_result.get("model_id") or "book-spine-detection-2cci9/2",
        "stages": [
            _debug_stage("Original", image),
            _debug_stage("YOLO predictions", _draw_regions(image, yolo_regions, outline=(245, 158, 11, 235))),
            _debug_stage("OpenCV refinement", draw_annotation(image, spines)),
        ],
        "rawPredictionCount": len(predictions),
        "refinementOptions": options,
    }


def _debug_stage(label: str, image: Image.Image) -> dict[str, str]:
    return {"label": label, "image": image_to_data_url(image)}


def _draw_regions(
    image: Image.Image,
    regions: list[dict[str, Any]],
    outline: tuple[int, int, int, int],
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()
    for index, region in enumerate(regions, start=1):
        polygon = [tuple(point) for point in region["polygon"]]
        draw.polygon(polygon, outline=outline, fill=(outline[0], outline[1], outline[2], 35))
        draw.line(polygon + [polygon[0]], fill=outline, width=4)
        x, y, _width, _height = region["box"]
        confidence = region.get("confidence")
        label = str(index) if confidence is None else f"{index} {confidence:.2f}"
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255, 255), font=font)
    return output


def _normalize_refinement_options(options: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(DEFAULT_REFINEMENT_OPTIONS)
    if options:
        normalized.update(options)
    normalized["boxPaddingX"] = _clamp_float(float(normalized["boxPaddingX"]), 0.0, 0.5)
    normalized["boxPaddingY"] = _clamp_float(float(normalized["boxPaddingY"]), 0.0, 0.3)
    normalized["edgeWeight"] = _clamp_float(float(normalized["edgeWeight"]), 0.0, 2.0)
    normalized["colorWeight"] = _clamp_float(float(normalized["colorWeight"]), 0.0, 2.0)
    normalized["houghWeight"] = _clamp_float(float(normalized["houghWeight"]), 0.0, 2.0)
    normalized["searchZoneRatio"] = _clamp_float(float(normalized["searchZoneRatio"]), 0.15, 0.49)
    normalized["minSpineWidth"] = int(_clamp(int(normalized["minSpineWidth"]), 4, 160))
    normalized["maxSkew"] = _clamp_float(float(normalized["maxSkew"]), 0.0, 0.8)
    normalized["confidenceThreshold"] = _clamp_float(float(normalized["confidenceThreshold"]), 0.0, 1.0)
    return normalized


def _normalize_array(array: np.ndarray) -> np.ndarray:
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _fit_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
