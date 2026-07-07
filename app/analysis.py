from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

try:
    from .yolo_client import infer_book_spines
except ImportError:
    from yolo_client import infer_book_spines


DEFAULT_FILTER_OPTIONS = {
    "maxBoxWidthRatio": 0.33,
    "maxBoxAreaRatio": 0.10,
}
OCR_SHEET_LABEL_WIDTH = 64
OCR_SHEET_PADDING = 16
OCR_SHEET_GAP = 18
OCR_SHEET_MIN_CROP_HEIGHT = 96
OCR_SHEET_MAX_CROP_WIDTH = 720


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


def analyze_shelf_photo(
    image_bytes: bytes,
    include_debug: bool = False,
    filter_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = _fit_image(image, max_side=1400)
    model_bytes = _image_to_jpeg_bytes(display_image)
    yolo_result = infer_book_spines(model_bytes)
    options = _normalize_filter_options(filter_options)
    yolo_regions = _regions_from_yolo_result(yolo_result, display_image.size)
    regions = _filter_regions_by_size(yolo_regions, display_image.size, options)

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
            "detector": "local_yolo",
            "refinement": "none",
        },
        "spines": [spine.__dict__ for spine in spines],
        "annotatedImage": image_to_data_url(annotated),
    }

    if include_debug:
        result["debug"] = _build_debug_payload(display_image, yolo_regions, regions, yolo_result, options)

    return result


def detect_book_spines(image: Image.Image) -> list[tuple[int, int, int, int]]:
    model_bytes = _image_to_jpeg_bytes(image.convert("RGB"))
    yolo_result = infer_book_spines(model_bytes)
    regions = _regions_from_yolo_result(yolo_result, image.size)
    return [region["box"] for region in _filter_regions_by_size(regions, image.size, _normalize_filter_options(None))]


def mock_ocr_call_numbers(count: int) -> list[str]:
    return [f"811.{120 + i * 7} K{i + 1:02d}" for i in range(count)]


def check_call_number_order(call_numbers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "expected_rank": index + 1,
            "status": "ok",
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


def _regions_from_yolo_result(
    result: dict[str, Any],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    predictions = result.get("predictions", [])
    regions = []
    for prediction in predictions:
        region = _region_from_prediction(prediction, image_width, image_height)
        if region is not None:
            regions.append(region)

    regions.sort(key=lambda region: (region["box"][0] + region["box"][2] / 2, region["box"][1]))
    return regions[:80]


def _filter_regions_by_size(
    regions: list[dict[str, Any]],
    image_size: tuple[int, int],
    options: dict[str, float],
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    max_box_width = image_width * options["maxBoxWidthRatio"]
    max_box_area = image_width * image_height * options["maxBoxAreaRatio"]
    return [
        region
        for region in regions
        if region["box"][2] < max_box_width and region["box"][2] * region["box"][3] < max_box_area
    ]


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
    yolo_regions: list[dict[str, Any]],
    filtered_regions: list[dict[str, Any]],
    yolo_result: dict[str, Any],
    filter_options: dict[str, float],
) -> dict[str, Any]:
    predictions = yolo_result.get("predictions", [])
    ocr_sheet, ocr_sheet_rows = _build_ocr_contact_sheet(image, filtered_regions)
    return {
        "usedFallback": False,
        "boundaryCount": len(predictions),
        "boxCount": len(filtered_regions),
        "filteredOutCount": len(yolo_regions) - len(filtered_regions),
        "filterOptions": filter_options,
        "ocrSheet": {
            "rowCount": len(ocr_sheet_rows),
            "rows": ocr_sheet_rows,
        },
        "model": yolo_result.get("model_id") or "models/yolo/yolo-model-v1.pt",
        "stages": [
            _debug_stage("Original", image),
            _debug_stage("YOLO predictions", _draw_regions(image, yolo_regions, outline=(245, 158, 11, 235))),
            _debug_stage("Size filter", _draw_regions(image, filtered_regions, outline=(43, 156, 94, 235))),
            _debug_stage("OCR contact sheet", ocr_sheet),
        ],
        "rawPredictionCount": len(predictions),
    }


def _debug_stage(label: str, image: Image.Image) -> dict[str, str]:
    return {"label": label, "image": image_to_data_url(image)}


def _normalize_filter_options(options: dict[str, Any] | None) -> dict[str, float]:
    normalized = dict(DEFAULT_FILTER_OPTIONS)
    if options:
        normalized.update(options)
    normalized["maxBoxWidthRatio"] = _clamp_float(float(normalized["maxBoxWidthRatio"]), 0.01, 1.0)
    normalized["maxBoxAreaRatio"] = _clamp_float(float(normalized["maxBoxAreaRatio"]), 0.01, 1.0)
    return normalized


def _build_ocr_contact_sheet(
    image: Image.Image,
    regions: list[dict[str, Any]],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    font = ImageFont.load_default()
    rows = []
    prepared_crops = []
    image_width, image_height = image.size

    for index, region in enumerate(regions, start=1):
        x, y, width, height = region["box"]
        x1 = _clamp(x, 0, image_width - 1)
        y1 = _clamp(y, 0, image_height - 1)
        x2 = _clamp(x + width, x1 + 1, image_width)
        y2 = _clamp(y + height, y1 + 1, image_height)
        crop = image.crop((x1, y1, x2, y2))
        scale = max(1.0, OCR_SHEET_MIN_CROP_HEIGHT / max(1, crop.height))
        if crop.width * scale > OCR_SHEET_MAX_CROP_WIDTH:
            scale = OCR_SHEET_MAX_CROP_WIDTH / max(1, crop.width)
        if scale != 1.0:
            crop = crop.resize(
                (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
                Image.Resampling.LANCZOS,
            )
        prepared_crops.append((index, region, crop, (x1, y1, x2 - x1, y2 - y1)))

    if not prepared_crops:
        empty = Image.new("RGB", (480, 160), "white")
        draw = ImageDraw.Draw(empty)
        draw.text((OCR_SHEET_PADDING, OCR_SHEET_PADDING), "No label boxes detected", fill=(17, 24, 39), font=font)
        return empty, []

    content_width = max(crop.width for _index, _region, crop, _source_box in prepared_crops)
    row_heights = [crop.height + OCR_SHEET_PADDING * 2 for _index, _region, crop, _source_box in prepared_crops]
    sheet_width = OCR_SHEET_LABEL_WIDTH + content_width + OCR_SHEET_PADDING * 3
    sheet_height = sum(row_heights) + OCR_SHEET_GAP * (len(row_heights) - 1) + OCR_SHEET_PADDING * 2
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    y_cursor = OCR_SHEET_PADDING

    for row_index, (index, region, crop, source_box) in enumerate(prepared_crops):
        row_height = row_heights[row_index]
        label = f"{index:02d}"
        row_y1 = y_cursor
        row_y2 = y_cursor + row_height
        crop_x = OCR_SHEET_LABEL_WIDTH + OCR_SHEET_PADDING * 2
        crop_y = y_cursor + OCR_SHEET_PADDING
        sheet.paste(crop, (crop_x, crop_y))
        draw.rectangle((OCR_SHEET_PADDING, row_y1, OCR_SHEET_LABEL_WIDTH, row_y2), fill=(241, 245, 249))
        draw.text((OCR_SHEET_PADDING + 10, row_y1 + OCR_SHEET_PADDING), label, fill=(20, 90, 122), font=font)
        draw.rectangle((OCR_SHEET_PADDING, row_y1, sheet_width - OCR_SHEET_PADDING, row_y2), outline=(217, 222, 231), width=1)
        rows.append(
            {
                "index": index,
                "sourceBox": list(source_box),
                "sheetBox": [crop_x, crop_y, crop.width, crop.height],
                "regionBox": list(region["box"]),
            }
        )
        y_cursor = row_y2 + OCR_SHEET_GAP

    return sheet, rows


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


