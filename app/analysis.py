from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class BoundaryLine:
    x_top: int
    x_bottom: int
    source: str = "projection"

    @property
    def mid_x(self) -> int:
        return (self.x_top + self.x_bottom) // 2


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


def analyze_shelf_photo(image_bytes: bytes, include_debug: bool = False) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    display_image = _fit_image(image, max_side=1400)
    regions, debug = _detect_book_spines(display_image, include_debug=include_debug)
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
        )
        for i, region in enumerate(regions)
    ]

    annotated = draw_annotation(display_image, spines)
    result = {
        "summary": {
            "bookCount": len(spines),
            "misplacedCount": len([spine for spine in spines if spine.status != "ok"]),
            "status": "needs_review" if any(spine.status != "ok" for spine in spines) else "ok",
        },
        "spines": [spine.__dict__ for spine in spines],
        "annotatedImage": image_to_data_url(annotated),
    }

    if include_debug:
        result["debug"] = debug

    return result


def detect_book_spines(image: Image.Image) -> list[tuple[int, int, int, int]]:
    regions, _debug = _detect_book_spines(image, include_debug=False)
    return [region["box"] for region in regions]


def _detect_book_spines(image: Image.Image, include_debug: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    if width < 80 or height < 80:
        regions = [_region_from_box((0, 0, width, height))]
        return regions, _build_debug_payload(image, [], regions, [], used_fallback=True)

    roi_box, roi_mask = _find_shelf_roi(rgb, gray)
    roi_x, roi_y, roi_width, roi_height = roi_box
    roi_rgb = rgb[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width, :]
    roi_gray = gray[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    crop_top = int(roi_height * 0.02)
    crop_bottom = int(roi_height * 0.98)
    cropped_rgb = roi_rgb[crop_top:crop_bottom, :, :]
    cropped_gray = roi_gray[crop_top:crop_bottom, :]
    enhanced = _enhance_for_spine_edges(cropped_gray)
    raw_edges = cv2.Canny(enhanced, threshold1=45, threshold2=130)
    vertical_edges = _extract_vertical_edges(enhanced)

    local_lines = _detect_boundary_lines(cropped_rgb, enhanced, raw_edges, roi_width, crop_top, crop_bottom)
    local_regions = _lines_to_regions(local_lines, roi_width, crop_top, crop_bottom)
    local_regions = _merge_narrow_regions(local_regions, roi_width)
    local_regions = _trim_low_activity_edge_regions(local_regions, roi_rgb)
    lines = _translate_lines(local_lines, roi_x)
    regions = _translate_regions(local_regions, roi_x, roi_y)

    debug_stages: list[dict[str, str]] = []
    if include_debug:
        debug_stages = [
            _debug_stage("Original", image),
            _debug_stage("Shelf ROI mask", _gray_to_image(roi_mask)),
            _debug_stage("Shelf ROI contour", _draw_roi(image, roi_box)),
            _debug_stage("Shelf crop", _array_to_image(roi_rgb)),
            _debug_stage("Analysis crop", _array_to_image(cropped_rgb)),
            _debug_stage("Enhanced", _gray_to_image(enhanced)),
            _debug_stage("Vertical edges", _gray_to_image(vertical_edges)),
            _debug_stage(
                "Boundary candidates",
                _draw_boundaries(image, lines, roi_y + crop_top, roi_y + crop_bottom),
            ),
        ]

    used_fallback = False
    if len(regions) < 3:
        fallback_regions = [_region_from_box(box) for box in _fallback_even_boxes(roi_width, roi_height)]
        regions = _translate_regions(fallback_regions, roi_x, roi_y)
        used_fallback = True

    regions = regions[:24]
    debug = _build_debug_payload(image, debug_stages, regions, lines, used_fallback=used_fallback)
    return regions, debug


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

        label = f"{spine.index}. {spine.call_number}"
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


def _build_debug_payload(
    image: Image.Image,
    stages: list[dict[str, str]],
    regions: list[dict[str, Any]],
    lines: list[BoundaryLine],
    used_fallback: bool,
) -> dict[str, Any]:
    debug_stages = list(stages)
    debug_stages.append(_debug_stage("Final spine polygons", _draw_regions(image, regions)))
    return {
        "usedFallback": used_fallback,
        "boundaryCount": len(lines),
        "boxCount": len(regions),
        "stages": debug_stages,
    }


def _debug_stage(label: str, image: Image.Image) -> dict[str, str]:
    return {"label": label, "image": image_to_data_url(image)}


def _gray_to_image(array: np.ndarray) -> Image.Image:
    normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(normalized.astype(np.uint8), mode="L").convert("RGB")


def _array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def _draw_roi(image: Image.Image, roi_box: tuple[int, int, int, int]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    x, y, width, height = roi_box
    draw.rectangle((x, y, x + width, y + height), outline=(245, 158, 11, 240), width=5, fill=(245, 158, 11, 35))
    return output


def _draw_boundaries(
    image: Image.Image,
    lines: list[BoundaryLine],
    crop_top: int,
    crop_bottom: int,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    draw.rectangle((0, crop_top, output.width, crop_bottom), outline=(20, 90, 122, 220), width=3)
    for line in lines:
        draw.line((line.x_top, crop_top, line.x_bottom, crop_bottom), fill=(245, 158, 11, 230), width=3)
    return output


def _draw_regions(image: Image.Image, regions: list[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    font = ImageFont.load_default()
    for index, region in enumerate(regions, start=1):
        polygon = [tuple(point) for point in region["polygon"]]
        draw.polygon(polygon, outline=(43, 156, 94, 235), fill=(43, 156, 94, 35))
        draw.line(polygon + [polygon[0]], fill=(43, 156, 94, 235), width=4)
        x, y, _width, _height = region["box"]
        draw.text((x + 6, y + 6), str(index), fill=(255, 255, 255, 255), font=font)
    return output


def _fit_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def _enhance_for_spine_edges(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.GaussianBlur(enhanced, (5, 5), 0)


def _find_shelf_roi(rgb: np.ndarray, gray: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    height, width = gray.shape
    enhanced = _enhance_for_spine_edges(gray)
    edges = cv2.Canny(enhanced, threshold1=40, threshold2=120)

    sobel_x = np.abs(cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    local_mean = cv2.blur(enhanced.astype(np.float32), (17, 17))
    local_sq_mean = cv2.blur((enhanced.astype(np.float32) ** 2), (17, 17))
    texture = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

    score = (
        _normalize_score(edges.astype(np.float32)) * 0.34
        + _normalize_score(sobel_x) * 0.30
        + _normalize_score(saturation) * 0.18
        + _normalize_score(texture) * 0.18
    )
    threshold = max(float(np.percentile(score, 68)), float(score.mean() + score.std() * 0.2))
    mask = np.where(score >= threshold, 255, 0).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(25, width // 18), max(15, height // 35)),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(9, width // 90), max(9, height // 90)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contour_box = _largest_shelf_contour_box(mask, width, height)
    if contour_box is not None:
        refined_box = _refine_roi_box_with_projection(score, contour_box, width, height)
        return refined_box, mask

    projection_box = _projection_roi_from_score(score, width, height)
    return projection_box, mask


def _normalize_score(array: np.ndarray) -> np.ndarray:
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def _largest_shelf_contour_box(mask: np.ndarray, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    image_area = image_width * image_height
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if width < image_width * 0.25 or height < image_height * 0.16:
            continue
        if area < image_area * 0.06:
            continue
        aspect = width / max(height, 1)
        if aspect < 0.8 or aspect > 12:
            continue
        fill_ratio = cv2.contourArea(contour) / max(area, 1)
        score = area * (0.65 + min(fill_ratio, 0.7))
        candidates.append((score, (x, y, width, height)))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _projection_roi_from_score(score: np.ndarray, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    row_score = cv2.GaussianBlur(score.mean(axis=1).reshape(-1, 1), (1, 31), 0).ravel()
    y1, y2 = _largest_active_range(row_score, percentile=62, min_length=max(40, image_height // 6))
    scoped = score[y1:y2, :]
    column_score = cv2.GaussianBlur(scoped.mean(axis=0).reshape(1, -1), (31, 1), 0).ravel()
    x1, x2 = _largest_active_range(column_score, percentile=58, min_length=max(60, image_width // 5))
    return _pad_box((x1, y1, x2 - x1, y2 - y1), image_width, image_height, x_pad_ratio=0.05, y_pad_ratio=0.08)


def _refine_roi_box_with_projection(
    score: np.ndarray,
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    scoped = score[y : y + height, x : x + width]
    if scoped.size == 0:
        return _pad_box(box, image_width, image_height, x_pad_ratio=0.04, y_pad_ratio=0.08)

    row_score = cv2.GaussianBlur(scoped.mean(axis=1).reshape(-1, 1), (1, 23), 0).ravel()
    local_y1, local_y2 = _largest_active_range(row_score, percentile=56, min_length=max(30, height // 3))

    row_scoped = scoped[local_y1:local_y2, :]
    column_score = cv2.GaussianBlur(row_scoped.mean(axis=0).reshape(1, -1), (31, 1), 0).ravel()
    local_x1, local_x2 = _largest_active_range(column_score, percentile=54, min_length=max(50, width // 4))

    refined = (x + local_x1, y + local_y1, local_x2 - local_x1, local_y2 - local_y1)
    return _pad_box(refined, image_width, image_height, x_pad_ratio=0.025, y_pad_ratio=0.055)


def _largest_active_range(score: np.ndarray, percentile: float, min_length: int) -> tuple[int, int]:
    threshold = float(np.percentile(score, percentile))
    active = score >= threshold
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                ranges.append((start, index))
            start = None
    if start is not None and len(active) - start >= min_length:
        ranges.append((start, len(active)))

    if not ranges:
        return 0, len(score)
    return max(ranges, key=lambda item: (item[1] - item[0], score[item[0] : item[1]].mean()))


def _pad_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    x_pad_ratio: float,
    y_pad_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    x_pad = int(width * x_pad_ratio)
    y_pad = int(height * y_pad_ratio)
    x1 = _clamp(x - x_pad, 0, image_width - 1)
    y1 = _clamp(y - y_pad, 0, image_height - 1)
    x2 = _clamp(x + width + x_pad, x1 + 1, image_width)
    y2 = _clamp(y + height + y_pad, y1 + 1, image_height)
    return x1, y1, x2 - x1, y2 - y1


def _extract_vertical_edges(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, threshold1=45, threshold2=130)
    vertical_kernel_height = max(25, gray.shape[0] // 8)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))
    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
    return cv2.dilate(vertical_edges, np.ones((3, 3), dtype=np.uint8), iterations=1)


def _detect_boundary_lines(
    rgb: np.ndarray,
    gray: np.ndarray,
    edge_image: np.ndarray,
    image_width: int,
    crop_top: int,
    crop_bottom: int,
) -> list[BoundaryLine]:
    lines = _hough_near_vertical_lines(edge_image, crop_top, crop_bottom, image_width)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    channel_scores = []
    for channel in cv2.split(lab):
        sobel_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        channel_scores.append(np.abs(sobel_x).mean(axis=0))
    gray_sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    channel_scores.append(np.abs(gray_sobel).mean(axis=0))

    projection = np.max(np.vstack(channel_scores), axis=0)
    projection = cv2.GaussianBlur(projection.reshape(1, -1), (1, 13), 0).ravel()
    threshold = max(float(projection.mean() + projection.std() * 0.55), float(np.percentile(projection, 76)))
    lines.extend(BoundaryLine(int(column), int(column), source="projection") for column in np.where(projection > threshold)[0])

    return _cluster_boundary_lines(lines, image_width)


def _hough_near_vertical_lines(
    edge_image: np.ndarray,
    crop_top: int,
    crop_bottom: int,
    image_width: int,
) -> list[BoundaryLine]:
    min_line_length = max(60, int(edge_image.shape[0] * 0.38))
    max_line_gap = max(10, int(edge_image.shape[0] * 0.07))
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    boundaries: list[BoundaryLine] = []
    for raw_line in lines[:, 0]:
        x1, y1, x2, y2 = [int(value) for value in raw_line]
        dy = y2 - y1
        if abs(dy) < 20:
            continue
        slope = (x2 - x1) / dy
        if abs(slope) > 0.35:
            continue

        top_y = 0
        bottom_y = crop_bottom - crop_top
        x_top = int(round(x1 + slope * (top_y - y1)))
        x_bottom = int(round(x1 + slope * (bottom_y - y1)))
        if -image_width * 0.1 <= x_top <= image_width * 1.1 and -image_width * 0.1 <= x_bottom <= image_width * 1.1:
            boundaries.append(BoundaryLine(_clamp(x_top, 0, image_width), _clamp(x_bottom, 0, image_width), source="hough"))
    return boundaries


def _cluster_boundary_lines(lines: list[BoundaryLine], image_width: int) -> list[BoundaryLine]:
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda line: line.mid_x)
    clusters: list[list[BoundaryLine]] = [[sorted_lines[0]]]
    for line in sorted_lines[1:]:
        if line.mid_x - clusters[-1][-1].mid_x <= 6:
            clusters[-1].append(line)
        else:
            clusters.append([line])

    min_gap = max(18, int(image_width * 0.025))
    filtered: list[BoundaryLine] = []
    for cluster in clusters:
        representative_items = [item for item in cluster if item.source == "hough"] or cluster
        line = BoundaryLine(
            int(round(float(np.mean([item.x_top for item in representative_items])))),
            int(round(float(np.mean([item.x_bottom for item in representative_items])))),
            source="hough" if representative_items[0].source == "hough" else "projection",
        )
        if line.mid_x < min_gap or line.mid_x > image_width - min_gap:
            continue
        if filtered and line.mid_x - filtered[-1].mid_x < min_gap:
            continue
        filtered.append(line)
    return filtered


def _lines_to_regions(
    lines: list[BoundaryLine],
    image_width: int,
    crop_top: int,
    crop_bottom: int,
) -> list[dict[str, Any]]:
    boundaries = [BoundaryLine(0, 0), *lines, BoundaryLine(image_width, image_width)]
    regions: list[dict[str, Any]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        width = right.mid_x - left.mid_x
        if not _is_reasonable_spine(width, image_width):
            continue
        polygon = [
            [_clamp(left.x_top, 0, image_width), crop_top],
            [_clamp(right.x_top, 0, image_width), crop_top],
            [_clamp(right.x_bottom, 0, image_width), crop_bottom],
            [_clamp(left.x_bottom, 0, image_width), crop_bottom],
        ]
        regions.append(_region_from_polygon(polygon))
    return regions


def _merge_narrow_regions(regions: list[dict[str, Any]], image_width: int) -> list[dict[str, Any]]:
    if not regions:
        return regions

    min_width = max(28, int(image_width * 0.035))
    merged: list[dict[str, Any]] = []
    for region in regions:
        if region["box"][2] >= min_width or not merged:
            merged.append(region)
            continue

        merged[-1] = _merge_two_regions(merged[-1], region)

    if len(merged) >= 2 and merged[-1]["box"][2] < min_width:
        merged[-2] = _merge_two_regions(merged[-2], merged[-1])
        merged.pop()
    return merged


def _merge_two_regions(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    polygon = [
        left["polygon"][0],
        right["polygon"][1],
        right["polygon"][2],
        left["polygon"][3],
    ]
    return _region_from_polygon(polygon)


def _translate_regions(regions: list[dict[str, Any]], x_offset: int, y_offset: int) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for region in regions:
        polygon = [[point[0] + x_offset, point[1] + y_offset] for point in region["polygon"]]
        translated.append(_region_from_polygon(polygon))
    return translated


def _translate_lines(lines: list[BoundaryLine], x_offset: int) -> list[BoundaryLine]:
    return [BoundaryLine(line.x_top + x_offset, line.x_bottom + x_offset, source=line.source) for line in lines]


def _trim_low_activity_edge_regions(regions: list[dict[str, Any]], rgb: np.ndarray) -> list[dict[str, Any]]:
    if len(regions) < 4:
        return regions

    scores = [_box_visual_activity(rgb, region["box"]) for region in regions]
    median_score = float(np.median(scores))
    if median_score <= 0:
        return regions

    trimmed = list(regions)
    trimmed_scores = list(scores)
    threshold = median_score * 0.35

    while len(trimmed) >= 4 and trimmed_scores[0] < threshold:
        trimmed.pop(0)
        trimmed_scores.pop(0)

    while len(trimmed) >= 4 and trimmed_scores[-1] < threshold:
        trimmed.pop()
        trimmed_scores.pop()

    return trimmed


def _region_from_box(box: tuple[int, int, int, int]) -> dict[str, Any]:
    x, y, width, height = box
    return _region_from_polygon([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])


def _region_from_polygon(polygon: list[list[int]]) -> dict[str, Any]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return {"box": (x1, y1, x2 - x1, y2 - y1), "polygon": polygon}


def _box_visual_activity(rgb: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x, y, width, height = box
    inset = min(8, max(0, width // 5))
    region = rgb[y : y + height, x + inset : x + width - inset]
    if region.size == 0:
        return 0.0

    lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
    color_variation = float(np.mean(np.std(lab, axis=(0, 1))))
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    edge_variation = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))))
    return color_variation + edge_variation


def _is_reasonable_spine(segment_width: int, image_width: int) -> bool:
    return max(18, image_width * 0.015) <= segment_width <= image_width * 0.35


def _fallback_even_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    count = min(10, max(4, width // 120))
    top = int(height * 0.1)
    box_height = int(height * 0.8)
    box_width = width // count
    return [(i * box_width, top, box_width if i < count - 1 else width - i * box_width, box_height) for i in range(count)]


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
