from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont

try:
    from .ocr_client import run_google_vision_ocr
    from .yolo_client import infer_book_spines
except ImportError:
    from ocr_client import run_google_vision_ocr
    from yolo_client import infer_book_spines


DEFAULT_FILTER_OPTIONS = {
    "maxBoxWidthRatio": 0.33,
    "maxBoxAreaRatio": 0.10,
}
OCR_SHEET_LABEL_WIDTH = 64
OCR_SHEET_PADDING = 16
OCR_SHEET_GAP = 18
OCR_SHEET_COLUMN_GAP = 24
OCR_SHEET_MIN_CROP_HEIGHT = 160
OCR_SHEET_MAX_CROP_WIDTH = 1120


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
    ocr_sheet, ocr_sheet_rows = _build_ocr_contact_sheet(display_image, regions)
    if ocr_sheet_rows:
        ocr_result = run_google_vision_ocr(_image_to_png_bytes(ocr_sheet))
        ocr_rows = _map_ocr_result_to_rows(ocr_result, ocr_sheet_rows)
    else:
        ocr_result = {"fullText": "", "annotations": []}
        ocr_rows = []

    call_numbers = [
        ocr_rows[index].get("text") or ""
        for index in range(len(regions))
    ]
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
        result["debug"] = _build_debug_payload(
            display_image,
            yolo_regions,
            regions,
            yolo_result,
            options,
            ocr_sheet,
            ocr_rows,
            ocr_result,
        )

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
    ocr_sheet: Image.Image,
    ocr_rows: list[dict[str, Any]],
    ocr_result: dict[str, Any],
) -> dict[str, Any]:
    predictions = yolo_result.get("predictions", [])
    return {
        "usedFallback": False,
        "boundaryCount": len(predictions),
        "boxCount": len(filtered_regions),
        "filteredOutCount": len(yolo_regions) - len(filtered_regions),
        "filterOptions": filter_options,
        "ocrSheet": {
            "rowCount": len(ocr_rows),
            "rows": ocr_rows,
        },
        "ocr": {
            "fullText": ocr_result.get("fullText", ""),
            "annotationCount": len(ocr_result.get("annotations", [])),
        },
        "model": yolo_result.get("model_id") or "models/yolo/yolo-model-v1.pt",
        "stages": [
            _debug_stage("Original", image),
            _debug_stage("YOLO predictions", _draw_regions(image, yolo_regions, outline=(245, 158, 11, 235))),
            _debug_stage("Size filter", _draw_regions(image, filtered_regions, outline=(43, 156, 94, 235))),
            _debug_stage("OCR contact sheet", ocr_sheet),
            _debug_stage("OCR bounding boxes", _draw_ocr_token_overlay(ocr_sheet, ocr_rows), _build_ocr_debug_details(ocr_rows)),
        ],
        "rawPredictionCount": len(predictions),
    }


def _debug_stage(label: str, image: Image.Image, details: list[str] | None = None) -> dict[str, Any]:
    stage = {"label": label, "image": image_to_data_url(image)}
    if details is not None:
        stage["details"] = details
    return stage


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

    prepared_rows = []
    for index, region, crop, source_box in prepared_crops:
        rotated_crop = crop.rotate(90, expand=True)
        prepared_rows.append((index, region, crop, rotated_crop, source_box))

    upright_width = max(crop.width for _index, _region, crop, _rotated_crop, _source_box in prepared_rows)
    rotated_width = max(rotated_crop.width for _index, _region, _crop, rotated_crop, _source_box in prepared_rows)
    row_heights = [
        max(crop.height, rotated_crop.height) + OCR_SHEET_PADDING * 2
        for _index, _region, crop, rotated_crop, _source_box in prepared_rows
    ]
    sheet_width = (
        OCR_SHEET_LABEL_WIDTH
        + upright_width
        + rotated_width
        + OCR_SHEET_PADDING * 3
        + OCR_SHEET_COLUMN_GAP
    )
    sheet_height = sum(row_heights) + OCR_SHEET_GAP * (len(row_heights) - 1) + OCR_SHEET_PADDING * 2
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    y_cursor = OCR_SHEET_PADDING

    for row_index, (index, region, crop, rotated_crop, source_box) in enumerate(prepared_rows):
        row_height = row_heights[row_index]
        label = f"{index:02d}"
        row_y1 = y_cursor
        row_y2 = y_cursor + row_height
        crop_x = OCR_SHEET_LABEL_WIDTH + OCR_SHEET_PADDING * 2
        crop_y = y_cursor + OCR_SHEET_PADDING + (row_height - OCR_SHEET_PADDING * 2 - crop.height) // 2
        rotated_x = crop_x + upright_width + OCR_SHEET_COLUMN_GAP
        rotated_y = y_cursor + OCR_SHEET_PADDING + (row_height - OCR_SHEET_PADDING * 2 - rotated_crop.height) // 2
        sheet.paste(crop, (crop_x, crop_y))
        sheet.paste(rotated_crop, (rotated_x, rotated_y))
        draw.rectangle((OCR_SHEET_PADDING, row_y1, OCR_SHEET_LABEL_WIDTH, row_y2), fill=(241, 245, 249))
        draw.text((OCR_SHEET_PADDING + 10, row_y1 + OCR_SHEET_PADDING), label, fill=(20, 90, 122), font=font)
        draw.rectangle((OCR_SHEET_PADDING, row_y1, sheet_width - OCR_SHEET_PADDING, row_y2), outline=(217, 222, 231), width=1)
        draw.line(
            (
                rotated_x - OCR_SHEET_COLUMN_GAP // 2,
                row_y1,
                rotated_x - OCR_SHEET_COLUMN_GAP // 2,
                row_y2,
            ),
            fill=(226, 232, 240),
            width=1,
        )
        rows.append(
            {
                "index": index,
                "sourceBox": list(source_box),
                "sheetBox": [crop_x, crop_y, crop.width, crop.height],
                "variants": [
                    {
                        "orientation": "upright",
                        "sheetBox": [crop_x, crop_y, crop.width, crop.height],
                    },
                    {
                        "orientation": "rotated_ccw_90",
                        "sheetBox": [rotated_x, rotated_y, rotated_crop.width, rotated_crop.height],
                    },
                ],
                "regionBox": list(region["box"]),
            }
        )
        y_cursor = row_y2 + OCR_SHEET_GAP

    return sheet, rows


def _map_ocr_result_to_rows(
    ocr_result: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mapped_rows = [
        {
            **row,
            "text": "",
            "tokens": [],
            "selectedOrientation": "upright",
            "variantResults": [
                {
                    **variant,
                    "text": "",
                    "tokens": [],
                }
                for variant in row.get("variants", [])
            ],
        }
        for row in rows
    ]

    for annotation in ocr_result.get("annotations", []):
        box = annotation["box"]
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        row_index, variant_index = _row_variant_index_for_point(center_x, center_y, mapped_rows)
        if row_index is None or variant_index is None:
            continue
        mapped_rows[row_index]["variantResults"][variant_index]["tokens"].append(
            {
                "text": annotation["text"],
                "box": box,
            }
        )

    for row in mapped_rows:
        for variant in row["variantResults"]:
            variant["tokens"].sort(key=lambda token: (token["box"][1], token["box"][0]))
            variant["text"] = " ".join(token["text"] for token in variant["tokens"]).strip()
        selected = max(row["variantResults"], key=lambda variant: len(variant["text"]), default=None)
        if selected:
            row["text"] = selected["text"]
            row["tokens"] = selected["tokens"]
            row["selectedOrientation"] = selected["orientation"]

    return mapped_rows


def _row_variant_index_for_point(x: float, y: float, rows: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    for row_index, row in enumerate(rows):
        variants = row.get("variantResults") or row.get("variants") or []
        for variant_index, variant in enumerate(variants):
            sheet_x, sheet_y, sheet_width, sheet_height = variant["sheetBox"]
            margin_y = max(8, sheet_height * 0.12)
            if sheet_x <= x <= sheet_x + sheet_width and sheet_y - margin_y <= y <= sheet_y + sheet_height + margin_y:
                return row_index, variant_index

        sheet_x, sheet_y, sheet_width, sheet_height = row["sheetBox"]
        margin_y = max(8, sheet_height * 0.12)
        if sheet_x <= x <= sheet_x + sheet_width and sheet_y - margin_y <= y <= sheet_y + sheet_height + margin_y:
            return row_index, None
    return None, None


def _draw_ocr_token_overlay(image: Image.Image, rows: list[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()
    variant_colors = {
        "upright": (37, 99, 235, 235),
        "rotated_ccw_90": (147, 51, 234, 235),
    }

    for row in rows:
        selected_orientation = row.get("selectedOrientation")
        for variant in row.get("variantResults", []):
            orientation = variant.get("orientation", "unknown")
            color = variant_colors.get(orientation, (71, 85, 105, 235))
            x, y, width, height = variant["sheetBox"]
            border_width = 4 if orientation == selected_orientation else 2
            draw.rectangle((x, y, x + width, y + height), outline=color, width=border_width)
            label = f"{row['index']:02d} {orientation}"
            if orientation == selected_orientation:
                label += " selected"
            _draw_debug_label(draw, (x, max(0, y - 18)), label, color, font)

            for token_index, token in enumerate(variant.get("tokens", []), start=1):
                box_x, box_y, box_width, box_height = token["box"]
                token_color = (14, 165, 233, 240) if orientation == selected_orientation else (168, 85, 247, 220)
                draw.rectangle(
                    (box_x, box_y, box_x + box_width, box_y + box_height),
                    outline=token_color,
                    width=2,
                )
                token_label = f"{token_index}. {_short_debug_text(token['text'])} [{box_x},{box_y},{box_width},{box_height}]"
                label_y = box_y - 18 if box_y > 22 else box_y + box_height + 2
                _draw_debug_label(draw, (box_x, label_y), token_label, token_color, font)

    return output


def _draw_debug_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    color: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x, y = position
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0] + 8
    height = text_box[3] - text_box[1] + 6
    draw.rectangle((x, y, x + width, y + height), fill=(15, 23, 42, 220), outline=color, width=1)
    draw.text((x + 4, y + 3), text, fill=(255, 255, 255, 255), font=font)


def _build_ocr_debug_details(rows: list[dict[str, Any]]) -> list[str]:
    details = []
    for row in rows:
        row_parts = []
        for variant in row.get("variantResults", []):
            tokens = variant.get("tokens", [])
            token_summary = ", ".join(
                f'"{token["text"]}" @ {token["box"]}'
                for token in tokens
            )
            if not token_summary:
                token_summary = "no tokens"
            selected = " selected" if variant.get("orientation") == row.get("selectedOrientation") else ""
            row_parts.append(f'{variant.get("orientation", "unknown")}{selected}: {token_summary}')
        details.append(f'Row {row["index"]:02d}: ' + " | ".join(row_parts))
    return details


def _short_debug_text(value: str, max_length: int = 18) -> str:
    clean = " ".join(value.split())
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 1] + "..."


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


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
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


