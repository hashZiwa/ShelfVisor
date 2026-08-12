from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .analysis_models import DetectedRegion, SpineAnalysis
from .image_processing import clamp


def render_result_image(image: Image.Image, spines: Sequence[SpineAnalysis]) -> Image.Image:
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
        box = draw.textbbox((0, 0), label, font=font)
        label_width, label_height = box[2] - box[0] + 12, box[3] - box[1] + 10
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


def render_yolo_regions(
    image: Image.Image,
    regions: Sequence[DetectedRegion],
    outline: tuple[int, int, int, int],
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()
    for index, region in enumerate(regions, start=1):
        polygon = [tuple(point) for point in region.polygon]
        draw.polygon(polygon, outline=outline, fill=(*outline[:3], 35))
        draw.line(polygon + [polygon[0]], fill=outline, width=4)
        x, y, _width, _height = region.box
        label = str(index) if region.confidence is None else f"{index} {region.confidence:.2f}"
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255, 255), font=font)
    return output


def render_ocr_overlay(image: Image.Image, rows: Sequence[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()
    colors = {"upright": (37, 99, 235, 235), "rotated_ccw_90": (147, 51, 234, 235)}
    for row in rows:
        selected = row.get("selectedOrientation")
        for variant in row.get("variantResults", []):
            orientation = variant.get("orientation", "unknown")
            color = colors.get(orientation, (71, 85, 105, 235))
            x, y, width, height = variant["sheetBox"]
            is_selected = orientation == selected
            if is_selected:
                draw.rectangle((x, y, x + width, y + height), fill=(34, 197, 94, 28))
            draw.rectangle(
                (x, y, x + width, y + height),
                outline=(34, 197, 94, 245) if is_selected else color,
                width=6 if is_selected else 2,
            )
            for token in variant.get("tokens", []):
                box_x, box_y, box_width, box_height = token["box"]
                draw.rectangle(
                    (box_x, box_y, box_x + box_width, box_y + box_height),
                    outline=(14, 165, 233, 240) if is_selected else (168, 85, 247, 220),
                    width=2,
                )
    return output


def grid_ocr_overlay_tiles(
    image: Image.Image,
    rows: Sequence[dict[str, Any]],
    rows_per_tile: int = 3,
) -> Image.Image:
    if not rows:
        return image
    tile_gap, outer_padding, header_height = 24, 18, 28
    chunks = [rows[index : index + rows_per_tile] for index in range(0, len(rows), rows_per_tile)]
    tiles = []
    font = ImageFont.load_default()
    for chunk in chunks:
        y1, y2 = _ocr_chunk_vertical_bounds(chunk, image.height)
        cropped = image.crop((0, y1, image.width, y2))
        tile = Image.new("RGB", (cropped.width, cropped.height + header_height), "white")
        draw = ImageDraw.Draw(tile, "RGBA")
        draw.rectangle((0, 0, tile.width, header_height), fill=(15, 23, 42, 240))
        draw.text(
            (10, 8),
            f"Rows {chunk[0]['index']:02d}-{chunk[-1]['index']:02d}",
            fill=(255, 255, 255, 255),
            font=font,
        )
        tile.paste(cropped, (0, header_height))
        tiles.append(tile)
    grid = Image.new(
        "RGB",
        (sum(tile.width for tile in tiles) + tile_gap * (len(tiles) - 1) + outer_padding * 2,
         max(tile.height for tile in tiles) + outer_padding * 2),
        (248, 250, 252),
    )
    x = outer_padding
    for tile in tiles:
        grid.paste(tile, (x, outer_padding))
        x += tile.width + tile_gap
    return grid


def build_ocr_debug_details(rows: Sequence[dict[str, Any]]) -> list[str]:
    details = []
    for row in rows:
        parts = []
        for variant in row.get("variantResults", []):
            tokens = variant.get("tokens", [])
            token_summary = ", ".join(
                f'"{token["text"]}" c{float(token.get("confidence", 0.0) or 0.0):.2f} @ {token["box"]}'
                for token in tokens
            ) or "no tokens"
            selected = " selected" if variant.get("orientation") == row.get("selectedOrientation") else ""
            parts.append(
                f'{variant.get("orientation", "unknown")}{selected} '
                f'avg {float(variant.get("averageConfidence", 0.0) or 0.0):.2f}: {token_summary}'
            )
        details.append(f'Row {row["index"]:02d}: ' + " | ".join(parts))
    return details


def _ocr_chunk_vertical_bounds(rows: Sequence[dict[str, Any]], image_height: int) -> tuple[int, int]:
    y_values = []
    for row in rows:
        for variant in row.get("variantResults", []):
            _x, y, _width, height = variant["sheetBox"]
            y_values.extend([y, y + height])
        if "sheetBox" in row:
            _x, y, _width, height = row["sheetBox"]
            y_values.extend([y, y + height])
    if not y_values:
        return 0, image_height
    y1 = clamp(min(y_values) - 32, 0, image_height - 1)
    return y1, clamp(max(y_values) + 32, y1 + 1, image_height)
