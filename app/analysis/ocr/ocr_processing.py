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
MIN_OCR_TOKENS = 3
LINE_OVERLAP_THRESHOLD = 0.5


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
        "eligible": False,
        "selectedOrientation": None,
        "selectedAverageConfidence": 0.0,
        "variantResults": [
            {**variant, "text": "", "tokens": [], "averageConfidence": 0.0, "eligible": False}
            for variant in row["variants"]
        ],
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
            _sort_tokens_by_average_box_shape(variant["tokens"], variant["orientation"])
            variant["text"] = " ".join(token["text"] for token in variant["tokens"]).strip()
            tokens = variant["tokens"]
            variant["averageConfidence"] = sum(token["confidence"] for token in tokens) / len(tokens) if tokens else 0.0
            variant["eligible"] = len(tokens) >= MIN_OCR_TOKENS
        eligible_variants = [item for item in row["variantResults"] if item["eligible"]]
        row["eligible"] = bool(eligible_variants)
        if eligible_variants:
            selected = max(eligible_variants, key=lambda item: item["averageConfidence"])
            row.update(
                text=selected["text"],
                tokens=selected["tokens"],
                selectedOrientation=selected["orientation"],
                selectedAverageConfidence=selected["averageConfidence"],
            )
        else:
            row.update(text="", tokens=[], selectedOrientation=None, selectedAverageConfidence=0.0)
    return rows


def _sort_tokens_by_average_box_shape(
    tokens: list[dict[str, Any]],
    orientation: str,
) -> None:
    if not tokens:
        return
    average_width = sum(token["box"][2] for token in tokens) / len(tokens)
    average_height = sum(token["box"][3] for token in tokens) / len(tokens)
    is_horizontal = average_width >= average_height
    lines = (
        _group_tokens_by_overlap(tokens, start_index=1, length_index=3)
        if is_horizontal
        else _group_tokens_by_overlap(tokens, start_index=0, length_index=2)
    )
    center_index, length_index = (1, 3) if is_horizontal else (0, 2)
    lines.sort(
        key=lambda line: sum(
            token["box"][center_index] + token["box"][length_index] / 2
            for token in line
        ) / len(line)
    )
    for line in lines:
        if is_horizontal:
            line.sort(key=lambda token: (token["box"][0], token["box"][1]))
        elif orientation == "rotated_ccw_90":
            line.sort(key=lambda token: (-token["box"][1], token["box"][0]))
        else:
            line.sort(key=lambda token: (token["box"][1], token["box"][0]))
    tokens[:] = [token for line in lines for token in line]


def _group_tokens_by_overlap(
    tokens: list[dict[str, Any]],
    start_index: int,
    length_index: int,
) -> list[list[dict[str, Any]]]:
    adjacency = [[] for _token in tokens]
    for first_index in range(len(tokens)):
        for second_index in range(first_index + 1, len(tokens)):
            first_box = tokens[first_index]["box"]
            second_box = tokens[second_index]["box"]
            overlap_ratio = _interval_overlap_ratio(
                first_box[start_index],
                first_box[length_index],
                second_box[start_index],
                second_box[length_index],
            )
            if overlap_ratio >= LINE_OVERLAP_THRESHOLD:
                adjacency[first_index].append(second_index)
                adjacency[second_index].append(first_index)

    groups = []
    visited = set()
    for start in range(len(tokens)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        indexes = []
        while stack:
            current = stack.pop()
            indexes.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        groups.append([tokens[index] for index in indexes])
    return groups


def _interval_overlap_ratio(
    first_start: float,
    first_length: float,
    second_start: float,
    second_length: float,
) -> float:
    shorter = min(first_length, second_length)
    if shorter <= 0:
        return 0.0
    overlap = max(
        0.0,
        min(first_start + first_length, second_start + second_length)
        - max(first_start, second_start),
    )
    return overlap / shorter


def _find_variant(point: tuple[float, float], rows: list[dict[str, Any]]) -> tuple[int, int] | None:
    x, y = point
    for row_index, row in enumerate(rows):
        for variant_index, variant in enumerate(row["variantResults"]):
            sx, sy, width, height = variant["sheetBox"]
            margin = max(8, height * 0.12)
            if sx <= x <= sx + width and sy - margin <= y <= sy + height + margin:
                return row_index, variant_index
    return None
