from __future__ import annotations

from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from ..analysis_models import DetectedRegion, OCRContactSheet
from ..image_processing import clamp


LABEL_WIDTH = 64
PADDING = 16
GAP = 18
COLUMN_GAP = 24
MIN_CROP_HEIGHT = 160
MAX_CROP_WIDTH = 1120


def build_ocr_contact_sheet(image: Image.Image, regions: Sequence[DetectedRegion]) -> OCRContactSheet:
    font = ImageFont.load_default()
    prepared = []
    image_width, image_height = image.size
    for index, region in enumerate(regions, start=1):
        x, y, width, height = region.box
        x1, y1 = clamp(x, 0, image_width - 1), clamp(y, 0, image_height - 1)
        x2, y2 = clamp(x + width, x1 + 1, image_width), clamp(y + height, y1 + 1, image_height)
        crop = image.crop((x1, y1, x2, y2))
        scale = max(1.0, MIN_CROP_HEIGHT / max(1, crop.height))
        if crop.width * scale > MAX_CROP_WIDTH:
            scale = MAX_CROP_WIDTH / max(1, crop.width)
        if scale != 1.0:
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
        prepared.append((index, region, crop, crop.rotate(90, expand=True), (x1, y1, x2 - x1, y2 - y1)))
    if not prepared:
        empty = Image.new("RGB", (480, 160), "white")
        ImageDraw.Draw(empty).text((PADDING, PADDING), "No label boxes detected", fill=(17, 24, 39), font=font)
        return OCRContactSheet(empty, [])
    upright_width = max(item[2].width for item in prepared)
    rotated_width = max(item[3].width for item in prepared)
    heights = [max(item[2].height, item[3].height) + PADDING * 2 for item in prepared]
    sheet_width = LABEL_WIDTH + upright_width + rotated_width + PADDING * 3 + COLUMN_GAP
    sheet_height = sum(heights) + GAP * (len(heights) - 1) + PADDING * 2
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    rows = []
    y_cursor = PADDING
    for row_index, (index, region, crop, rotated, source_box) in enumerate(prepared):
        row_height = heights[row_index]
        crop_x = LABEL_WIDTH + PADDING * 2
        crop_y = y_cursor + PADDING + (row_height - PADDING * 2 - crop.height) // 2
        rotated_x = crop_x + upright_width + COLUMN_GAP
        rotated_y = y_cursor + PADDING + (row_height - PADDING * 2 - rotated.height) // 2
        sheet.paste(crop, (crop_x, crop_y))
        sheet.paste(rotated, (rotated_x, rotated_y))
        draw.text((PADDING + 10, y_cursor + PADDING), f"{index:02d}", fill=(20, 90, 122), font=font)
        rows.append({
            "index": index,
            "sourceBox": list(source_box),
            "sheetBox": [crop_x, crop_y, crop.width, crop.height],
            "variants": [
                {"orientation": "upright", "sheetBox": [crop_x, crop_y, crop.width, crop.height]},
                {"orientation": "rotated_ccw_90", "sheetBox": [rotated_x, rotated_y, rotated.width, rotated.height]},
            ],
            "regionBox": list(region.box),
        })
        y_cursor += row_height + GAP
    return OCRContactSheet(sheet, rows)


def map_ocr_result_to_rows(ocr_result: dict[str, Any], contact_sheet: OCRContactSheet) -> list[dict[str, Any]]:
    rows = [{
        **row,
        "text": "",
        "tokens": [],
        "selectedOrientation": "upright",
        "variantResults": [{**variant, "text": "", "tokens": [], "averageConfidence": 0.0} for variant in row["variants"]],
    } for row in contact_sheet.rows]
    for annotation in ocr_result.get("annotations", []):
        box = annotation["box"]
        center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
        location = _find_variant(center, rows)
        if location is None:
            continue
        row_index, variant_index = location
        rows[row_index]["variantResults"][variant_index]["tokens"].append({
            "text": annotation["text"],
            "box": box,
            "confidence": float(annotation.get("confidence", 0.0) or 0.0),
        })
    for row in rows:
        for variant in row["variantResults"]:
            _sort_tokens_by_average_box_shape(variant["tokens"])
            variant["text"] = " ".join(token["text"] for token in variant["tokens"]).strip()
            tokens = variant["tokens"]
            variant["averageConfidence"] = sum(token["confidence"] for token in tokens) / len(tokens) if tokens else 0.0
        selected = max(row["variantResults"], key=lambda item: item["averageConfidence"])
        row.update(text=selected["text"], tokens=selected["tokens"], selectedOrientation=selected["orientation"])
        row["selectedAverageConfidence"] = selected["averageConfidence"]
    return rows


def _sort_tokens_by_average_box_shape(tokens: list[dict[str, Any]]) -> None:
    if not tokens:
        return
    average_width = sum(token["box"][2] for token in tokens) / len(tokens)
    average_height = sum(token["box"][3] for token in tokens) / len(tokens)
    key = (
        (lambda token: (token["box"][1], token["box"][0]))
        if average_width >= average_height
        else (lambda token: (token["box"][0], token["box"][1]))
    )
    tokens.sort(key=key)


def _find_variant(point: tuple[float, float], rows: list[dict[str, Any]]) -> tuple[int, int] | None:
    x, y = point
    for row_index, row in enumerate(rows):
        for variant_index, variant in enumerate(row["variantResults"]):
            sx, sy, width, height = variant["sheetBox"]
            margin = max(8, height * 0.12)
            if sx <= x <= sx + width and sy - margin <= y <= sy + height + margin:
                return row_index, variant_index
    return None
