from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .analysis import analyze_shelf_photo
except ImportError:
    from analysis import analyze_shelf_photo


ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
TEST_IMAGES_ROOT = ROOT / "test_images"
HOST = "127.0.0.1"
PORT = int(os.environ.get("SHELFVISOR_PORT", "8000"))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_BATCH_IMAGES = 5
TEST_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(
    title="ShelfVisor",
    description="Demo API for analyzing shelf photos and call-number order.",
    version="0.2.0",
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    debug: bool = Form(False),
    box_padding_x: float = Form(0.12),
    box_padding_y: float = Form(0.00),
    edge_weight: float = Form(0.45),
    color_weight: float = Form(0.45),
    hough_weight: float = Form(0.10),
    search_zone_ratio: float = Form(0.34),
    min_spine_width: int = Form(18),
    max_skew: float = Form(0.22),
    confidence_threshold: float = Form(0.15),
) -> dict[str, Any]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Please upload an image smaller than 15MB.")

    try:
        return analyze_shelf_photo(
            image_bytes,
            include_debug=debug,
            refinement_options=_refinement_options_from_form(
                box_padding_x,
                box_padding_y,
                edge_weight,
                color_weight,
                hough_weight,
                search_zone_ratio,
                min_spine_width,
                max_skew,
                confidence_threshold,
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}") from error


@app.post("/api/analyze-batch")
async def analyze_batch(
    images: list[UploadFile] = File(...),
    debug: bool = Form(False),
    box_padding_x: float = Form(0.12),
    box_padding_y: float = Form(0.00),
    edge_weight: float = Form(0.45),
    color_weight: float = Form(0.45),
    hough_weight: float = Form(0.10),
    search_zone_ratio: float = Form(0.34),
    min_spine_width: int = Form(18),
    max_skew: float = Form(0.22),
    confidence_threshold: float = Form(0.15),
) -> dict[str, Any]:
    if not images:
        raise HTTPException(status_code=400, detail="Please upload at least one image.")
    if len(images) > MAX_BATCH_IMAGES:
        raise HTTPException(status_code=400, detail=f"Please upload up to {MAX_BATCH_IMAGES} images.")

    options = _refinement_options_from_form(
        box_padding_x,
        box_padding_y,
        edge_weight,
        color_weight,
        hough_weight,
        search_zone_ratio,
        min_spine_width,
        max_skew,
        confidence_threshold,
    )

    prepared_images = []
    for index, image in enumerate(images):
        filename = image.filename or f"image-{index + 1}"
        if image.content_type and not image.content_type.startswith("image/"):
            prepared_images.append({"filename": filename, "error": "Please upload an image file."})
            continue

        image_bytes = await image.read()
        if not image_bytes:
            prepared_images.append({"filename": filename, "error": "The uploaded image is empty."})
            continue
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            prepared_images.append({"filename": filename, "error": "Please upload an image smaller than 15MB."})
            continue

        prepared_images.append({"filename": filename, "bytes": image_bytes})

    tasks = [
        _analyze_prepared_image(item, debug=debug, refinement_options=options)
        for item in prepared_images
    ]
    results = await asyncio.gather(*tasks)
    completed_count = sum(1 for result in results if result["status"] == "completed")
    failed_count = len(results) - completed_count

    return {
        "summary": {
            "imageCount": len(results),
            "completedCount": completed_count,
            "failedCount": failed_count,
            "maxImages": MAX_BATCH_IMAGES,
        },
        "results": results,
    }


@app.get("/api/test-images")
def list_test_images() -> dict[str, Any]:
    images = _test_images()
    return {
        "count": len(images),
        "maxImages": MAX_BATCH_IMAGES,
        "images": [{"name": image.name} for image in images],
    }


@app.post("/api/analyze-test-image")
async def analyze_test_image(
    debug: bool = Form(True),
    box_padding_x: float = Form(0.12),
    box_padding_y: float = Form(0.00),
    edge_weight: float = Form(0.45),
    color_weight: float = Form(0.45),
    hough_weight: float = Form(0.10),
    search_zone_ratio: float = Form(0.34),
    min_spine_width: int = Form(18),
    max_skew: float = Form(0.22),
    confidence_threshold: float = Form(0.15),
) -> dict[str, Any]:
    image_path = _first_test_image()
    if image_path is None:
        raise HTTPException(status_code=404, detail="No test image found in test_images.")

    try:
        result = analyze_shelf_photo(
            image_path.read_bytes(),
            include_debug=debug,
            refinement_options=_refinement_options_from_form(
                box_padding_x,
                box_padding_y,
                edge_weight,
                color_weight,
                hough_weight,
                search_zone_ratio,
                min_spine_width,
                max_skew,
                confidence_threshold,
            ),
        )
        result["summary"]["sourceImage"] = image_path.name
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error}") from error


@app.post("/api/analyze-test-images")
async def analyze_test_images(
    image_names: list[str] = Form(...),
    debug: bool = Form(True),
    box_padding_x: float = Form(0.12),
    box_padding_y: float = Form(0.00),
    edge_weight: float = Form(0.45),
    color_weight: float = Form(0.45),
    hough_weight: float = Form(0.10),
    search_zone_ratio: float = Form(0.34),
    min_spine_width: int = Form(18),
    max_skew: float = Form(0.22),
    confidence_threshold: float = Form(0.15),
) -> dict[str, Any]:
    selected_names = list(dict.fromkeys(image_names))
    if not selected_names:
        raise HTTPException(status_code=400, detail="Please select at least one test image.")
    if len(selected_names) > MAX_BATCH_IMAGES:
        raise HTTPException(status_code=400, detail=f"Please select up to {MAX_BATCH_IMAGES} test images.")

    options = _refinement_options_from_form(
        box_padding_x,
        box_padding_y,
        edge_weight,
        color_weight,
        hough_weight,
        search_zone_ratio,
        min_spine_width,
        max_skew,
        confidence_threshold,
    )

    prepared_images = []
    for image_name in selected_names:
        image_path = _test_image_by_name(image_name)
        if image_path is None:
            prepared_images.append({"filename": image_name, "error": "Test image not found."})
            continue
        prepared_images.append({"filename": image_path.name, "bytes": image_path.read_bytes()})

    results = await asyncio.gather(
        *[
            _analyze_prepared_image(item, debug=debug, refinement_options=options)
            for item in prepared_images
        ]
    )
    completed_count = sum(1 for result in results if result["status"] == "completed")
    failed_count = len(results) - completed_count

    return {
        "summary": {
            "imageCount": len(results),
            "completedCount": completed_count,
            "failedCount": failed_count,
            "maxImages": MAX_BATCH_IMAGES,
            "source": "test_images",
        },
        "results": results,
    }


def _first_test_image() -> Path | None:
    images = _test_images()
    return images[0] if images else None


def _test_images() -> list[Path]:
    if not TEST_IMAGES_ROOT.exists():
        return []
    return sorted(
        path for path in TEST_IMAGES_ROOT.iterdir() if path.is_file() and path.suffix.lower() in TEST_IMAGE_EXTENSIONS
    )


def _test_image_by_name(image_name: str) -> Path | None:
    safe_name = Path(image_name).name
    for image_path in _test_images():
        if image_path.name == safe_name:
            return image_path
    return None


async def _analyze_prepared_image(
    item: dict[str, Any],
    debug: bool,
    refinement_options: dict[str, Any],
) -> dict[str, Any]:
    filename = item["filename"]
    if "error" in item:
        return {"filename": filename, "status": "failed", "error": item["error"]}

    try:
        result = await asyncio.to_thread(
            analyze_shelf_photo,
            item["bytes"],
            include_debug=debug,
            refinement_options=refinement_options,
        )
        result["summary"]["sourceImage"] = filename
        return {"filename": filename, "status": "completed", "result": result}
    except Exception as error:
        return {"filename": filename, "status": "failed", "error": str(error)}


def _refinement_options_from_form(
    box_padding_x: float,
    box_padding_y: float,
    edge_weight: float,
    color_weight: float,
    hough_weight: float,
    search_zone_ratio: float,
    min_spine_width: int,
    max_skew: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    return {
        "boxPaddingX": box_padding_x,
        "boxPaddingY": box_padding_y,
        "edgeWeight": edge_weight,
        "colorWeight": color_weight,
        "houghWeight": hough_weight,
        "searchZoneRatio": search_zone_ratio,
        "minSpineWidth": min_spine_width,
        "maxSkew": max_skew,
        "confidenceThreshold": confidence_threshold,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


app.mount("/", StaticFiles(directory=WEB_ROOT), name="web")


def main() -> None:
    import uvicorn

    root_path = str(ROOT)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)

    uvicorn.run(
        "app.server:app",
        host=HOST,
        port=PORT,
        reload=True,
        app_dir=root_path,
        reload_dirs=[root_path],
    )


if __name__ == "__main__":
    main()
