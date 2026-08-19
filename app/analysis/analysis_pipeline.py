from __future__ import annotations

from dataclasses import asdict
from typing import Any

from PIL import Image

from .analysis_models import SpineAnalysis
from .call_number_processing import check_call_number_order
from .debug_artifacts import DebugArtifactCollector
from .image_processing import image_to_data_url, image_to_jpeg_bytes, image_to_png_bytes, prepare_image
from .ocr import build_ocr_contact_sheet, map_ocr_result_to_rows, run_ocr_inference
from .result_rendering import (
    build_ocr_debug_details,
    grid_ocr_overlay_tiles,
    render_ocr_overlay,
    render_result_image,
    render_yolo_regions,
)
from .yolo import filter_yolo_regions, normalize_filter_options, parse_yolo_predictions, run_yolo_inference


def analyze_shelf_photo(
    image_bytes: bytes,
    include_debug: bool = False,
    filter_options: dict[str, Any] | None = None,
    write_debug_artifacts: bool = False,
) -> dict[str, Any]:
    prepared = prepare_image(image_bytes, max_side=1400)
    image = prepared.image
    yolo_result = run_yolo_inference(image_to_jpeg_bytes(image))
    options = normalize_filter_options(filter_options)
    yolo_regions = parse_yolo_predictions(yolo_result, image.size)
    ocr_regions = filter_yolo_regions(yolo_regions, image.size, options)
    contact_sheet = build_ocr_contact_sheet(image, ocr_regions)
    if contact_sheet.rows:
        ocr_result = run_ocr_inference(image_to_png_bytes(contact_sheet.image))
        all_ocr_rows = map_ocr_result_to_rows(ocr_result, contact_sheet)
    else:
        ocr_result = {"fullText": "", "annotations": []}
        all_ocr_rows = []

    eligible_pairs = [
        (region, row)
        for region, row in zip(ocr_regions, all_ocr_rows)
        if row["eligible"]
    ]
    regions = [region for region, _row in eligible_pairs]
    ocr_rows = [row for _region, row in eligible_pairs]

    call_numbers = [row.get("text") or "" for row in ocr_rows]
    statuses = check_call_number_order(call_numbers)
    spines = [
        SpineAnalysis(
            index=index + 1,
            x=region.box[0],
            y=region.box[1],
            width=region.box[2],
            height=region.box[3],
            call_number=call_numbers[index],
            expected_rank=statuses[index].expected_rank,
            status=statuses[index].status,
            polygon=region.polygon,
            confidence=region.confidence,
            class_name=region.class_name,
        )
        for index, region in enumerate(regions)
    ]
    annotated = render_result_image(image, spines)
    result = {
        "summary": {
            "bookCount": len(spines),
            "misplacedCount": sum(spine.status != "ok" for spine in spines),
            "status": "needs_review" if any(spine.status != "ok" for spine in spines) else "ok",
            "detector": "local_yolo",
            "refinement": "none",
        },
        "spines": [asdict(spine) for spine in spines],
        "annotatedImage": image_to_data_url(annotated),
    }

    collector = DebugArtifactCollector(include_debug, write_debug_artifacts)
    if include_debug:
        predictions = yolo_result.get("predictions", [])
        ocr_overlay = grid_ocr_overlay_tiles(
            render_ocr_overlay(contact_sheet.image, all_ocr_rows),
            all_ocr_rows,
        )
        collector.set_metadata(
            usedFallback=False,
            boundaryCount=len(predictions),
            boxCount=len(ocr_regions),
            filteredOutCount=len(yolo_regions) - len(ocr_regions),
            filterOptions=options,
            ocrSheet={"rowCount": len(all_ocr_rows), "rows": all_ocr_rows},
            ocr={"fullText": ocr_result.get("fullText", ""), "annotationCount": len(ocr_result.get("annotations", []))},
            model=yolo_result.get("model_id") or "models/yolo/yolo-model-v1.pt",
            rawPredictionCount=len(predictions),
        )
        collector.add_image("original", "Original", image)
        collector.add_image("yolo-predictions", "YOLO predictions", render_yolo_regions(image, yolo_regions, (245, 158, 11, 235)))
        collector.add_image("size-filter", "Size filter", render_yolo_regions(image, ocr_regions, (43, 156, 94, 235)))
        collector.add_image("ocr-contact-sheet", "OCR contact sheet", contact_sheet.image)
        collector.add_image(
            "ocr-bounding-boxes",
            "OCR bounding boxes",
            ocr_overlay,
            build_ocr_debug_details(all_ocr_rows),
        )
        collector.add_image("final-result", "Final result", annotated)
        collector.write()
        result["debug"] = collector.build_payload()
    return result


def detect_book_spines(image: Image.Image) -> list[tuple[int, int, int, int]]:
    yolo_result = run_yolo_inference(image_to_jpeg_bytes(image.convert("RGB")))
    regions = parse_yolo_predictions(yolo_result, image.size)
    return [region.box for region in filter_yolo_regions(regions, image.size, normalize_filter_options(None))]


def mock_ocr_call_numbers(count: int) -> list[str]:
    return [f"811.{120 + index * 7} K{index + 1:02d}" for index in range(count)]
