# Image Analysis Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split ShelfVisor's image-analysis monolith into responsibility-focused YOLO, OCR, rendering, debugging, and orchestration modules while preserving existing API behavior.

**Architecture:** Replace `app/analysis.py` with an `app.analysis` package whose public API remains `analyze_shelf_photo`. Keep external inference isolated from deterministic post-processing, pass typed dataclasses between modules, and let the pipeline record stage outputs through a debug artifact collector that can serialize both API payloads and optional local files.

**Tech Stack:** Python 3, FastAPI, Pillow, Ultralytics YOLO, Google Cloud Vision, standard-library `unittest`, dataclasses

## Global Constraints

- Preserve the existing API request and response shapes before changing any analysis algorithm.
- Preserve uncommitted user changes in `app/analysis.py` and `app/ocr_client.py`; use their current working-tree contents as the migration source.
- Keep `from app.analysis import analyze_shelf_photo` as the supported server import.
- Do not require a real YOLO model or Google Vision credentials in unit tests.
- Write local debug artifacts only when debug output and local artifact writing are explicitly enabled.
- A local debug-write failure must not invalidate a successful analysis response.
- Keep batch-image failures isolated to the individual image.

---

## File Structure

**Create**

- `app/analysis/__init__.py`: stable public package API
- `app/analysis/analysis_models.py`: dataclasses shared across analysis responsibilities
- `app/analysis/image_processing.py`: image decoding, resize, crop, and encoding
- `app/analysis/yolo/__init__.py`: YOLO package exports
- `app/analysis/yolo/yolo_inference.py`: model configuration, loading, and inference
- `app/analysis/yolo/yolo_processing.py`: prediction conversion, sorting, and filtering
- `app/analysis/ocr/__init__.py`: OCR package exports
- `app/analysis/ocr/ocr_inference.py`: Google Vision authentication and inference
- `app/analysis/ocr/ocr_processing.py`: contact-sheet construction and OCR result mapping
- `app/analysis/call_number_processing.py`: call-number parsing and order evaluation
- `app/analysis/result_rendering.py`: final and debug image rendering
- `app/analysis/debug_artifacts.py`: debug payload collection and optional filesystem export
- `app/analysis/analysis_pipeline.py`: end-to-end orchestration
- `tests/test_image_processing.py`
- `tests/test_yolo_processing.py`
- `tests/test_ocr_processing.py`
- `tests/test_call_number_processing.py`
- `tests/test_debug_artifacts.py`
- `tests/test_analysis_pipeline.py`

**Modify**

- `app/server.py`: retain the stable package import and current endpoint behavior
- `.gitignore`: exclude `debug_runs/`

**Remove after migration**

- `app/analysis.py`
- `app/yolo_client.py`
- `app/ocr_client.py`

---

### Task 1: Lock Existing Behavior and Introduce Shared Models

**Files:**
- Create: `app/analysis/analysis_models.py`
- Create: `app/analysis/image_processing.py`
- Create: `tests/test_image_processing.py`

**Interfaces:**
- Produces: `PreparedImage`, `DetectedRegion`, `OCRContactSheet`, `OCRRowResult`, `SpineAnalysis`
- Produces: `prepare_image(image_bytes: bytes, max_side: int = 1400) -> PreparedImage`
- Produces: `image_to_jpeg_bytes`, `image_to_png_bytes`, `image_to_data_url`, `crop_to_box`

- [ ] **Step 1: Write failing image-processing tests**

```python
import io
import unittest
from PIL import Image

from app.analysis.image_processing import (
    image_to_data_url,
    prepare_image,
)


class ImageProcessingTests(unittest.TestCase):
    def test_prepare_image_converts_rgb_and_limits_long_side(self):
        source = Image.new("RGBA", (2000, 1000), (255, 0, 0, 128))
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")

        prepared = prepare_image(buffer.getvalue(), max_side=1400)

        self.assertEqual(prepared.image.mode, "RGB")
        self.assertEqual(prepared.image.size, (1400, 700))
        self.assertEqual(prepared.original_size, (2000, 1000))

    def test_image_to_data_url_returns_jpeg_url(self):
        value = image_to_data_url(Image.new("RGB", (10, 10), "white"))
        self.assertTrue(value.startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run: `python -m unittest tests.test_image_processing -v`

Expected: FAIL because `app.analysis.image_processing` does not exist.

- [ ] **Step 3: Implement typed models and image helpers**

Define frozen or mutable dataclasses as required by current behavior. `DetectedRegion` must contain `box`, `polygon`, `confidence`, `class_name`, and `raw`. `PreparedImage` must contain the resized RGB image and `original_size`. Port the existing `_fit_image`, `_image_to_jpeg_bytes`, `_image_to_png_bytes`, and `image_to_data_url` logic without changing quality settings.

- [ ] **Step 4: Run the tests**

Run: `python -m unittest tests.test_image_processing -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit the shared foundation**

```powershell
git add app/analysis/analysis_models.py app/analysis/image_processing.py tests/test_image_processing.py
git commit -m "refactor: add analysis models and image helpers"
```

---

### Task 2: Separate YOLO Inference from YOLO Processing

**Files:**
- Create: `app/analysis/yolo/__init__.py`
- Create: `app/analysis/yolo/yolo_inference.py`
- Create: `app/analysis/yolo/yolo_processing.py`
- Create: `tests/test_yolo_processing.py`
- Source: `app/yolo_client.py`
- Source: `app/analysis.py`

**Interfaces:**
- Consumes: `DetectedRegion`
- Produces: `run_yolo_inference(image_bytes: bytes) -> dict[str, Any]`
- Produces: `normalize_filter_options(options: Mapping[str, Any] | None) -> dict[str, float]`
- Produces: `parse_yolo_predictions(result, image_size) -> list[DetectedRegion]`
- Produces: `filter_yolo_regions(regions, image_size, options) -> list[DetectedRegion]`

- [ ] **Step 1: Write failing deterministic YOLO tests**

```python
import unittest

from app.analysis.yolo.yolo_processing import (
    filter_yolo_regions,
    normalize_filter_options,
    parse_yolo_predictions,
)


class YoloProcessingTests(unittest.TestCase):
    def test_predictions_are_converted_clamped_and_sorted_left_to_right(self):
        raw = {"predictions": [
            {"x": 80, "y": 50, "width": 20, "height": 60, "confidence": 0.8, "class": "book"},
            {"x": 20, "y": 50, "width": 20, "height": 60, "confidence": 0.9, "class": "book"},
        ]}
        regions = parse_yolo_predictions(raw, (100, 100))
        self.assertEqual([region.box[0] for region in regions], [10, 70])

    def test_size_filter_removes_wide_regions(self):
        regions = parse_yolo_predictions(
            {"predictions": [{"x": 50, "y": 50, "width": 80, "height": 20}]},
            (100, 100),
        )
        options = normalize_filter_options({"maxBoxWidthRatio": 0.33, "maxBoxAreaRatio": 1.0})
        self.assertEqual(filter_yolo_regions(regions, (100, 100), options), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m unittest tests.test_yolo_processing -v`

Expected: FAIL because the YOLO package is absent.

- [ ] **Step 3: Port deterministic YOLO post-processing**

Move region parsing, polygon fallback, box calculation, sorting, count limiting, filter-option normalization, and size filtering from the current working-tree version of `app/analysis.py`. Preserve the current limit of 80 regions and current default ratios.

- [ ] **Step 4: Port YOLO inference**

Move the current working-tree implementation of `app/yolo_client.py` into `yolo_inference.py`. Rename the public entry point to `run_yolo_inference` and export it from `yolo/__init__.py`. Keep lazy imports and model caching so importing deterministic processing does not load Ultralytics.

- [ ] **Step 5: Run YOLO unit tests**

Run: `python -m unittest tests.test_yolo_processing -v`

Expected: all tests PASS without loading the model.

- [ ] **Step 6: Commit the YOLO package**

```powershell
git add app/analysis/yolo tests/test_yolo_processing.py
git commit -m "refactor: separate YOLO inference and processing"
```

---

### Task 3: Separate OCR Inference from OCR Processing

**Files:**
- Create: `app/analysis/ocr/__init__.py`
- Create: `app/analysis/ocr/ocr_inference.py`
- Create: `app/analysis/ocr/ocr_processing.py`
- Create: `tests/test_ocr_processing.py`
- Source: `app/ocr_client.py`
- Source: `app/analysis.py`

**Interfaces:**
- Consumes: `DetectedRegion`, `OCRContactSheet`, `OCRRowResult`
- Produces: `run_ocr_inference(image_bytes: bytes) -> dict[str, Any]`
- Produces: `build_ocr_contact_sheet(image, regions) -> OCRContactSheet`
- Produces: `map_ocr_result_to_rows(ocr_result, contact_sheet) -> list[OCRRowResult]`

- [ ] **Step 1: Write failing contact-sheet and mapping tests**

Create two narrow tests using a 100×100 in-memory image and one `DetectedRegion`. Assert that the sheet contains both `upright` and `rotated_ccw_90` variants, and that an annotation centered inside an upright variant is assigned to that row with its text and confidence preserved.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m unittest tests.test_ocr_processing -v`

Expected: FAIL because the OCR package is absent.

- [ ] **Step 3: Port OCR contact-sheet and mapping logic**

Move all `OCR_SHEET_*` constants, contact-sheet construction, row/variant hit testing, token ordering, direction selection, and debug-detail preparation from the current working-tree version of `app/analysis.py`. Replace internal row dictionaries with the models from Task 1, converting to dictionaries only at the API/debug boundary.

- [ ] **Step 4: Port Google Vision inference**

Move the current working-tree version of `app/ocr_client.py` into `ocr_inference.py`, including credential discovery, local environment loading, cached client creation, full-text token extraction, and fallback annotation extraction. Export `run_ocr_inference` from `ocr/__init__.py`.

- [ ] **Step 5: Run OCR unit tests**

Run: `python -m unittest tests.test_ocr_processing -v`

Expected: all tests PASS without importing Google Cloud Vision at module import time.

- [ ] **Step 6: Commit the OCR package**

```powershell
git add app/analysis/ocr tests/test_ocr_processing.py
git commit -m "refactor: separate OCR inference and processing"
```

---

### Task 4: Extract Call-Number Evaluation and Rendering

**Files:**
- Create: `app/analysis/call_number_processing.py`
- Create: `app/analysis/result_rendering.py`
- Create: `tests/test_call_number_processing.py`
- Source: `app/analysis.py`

**Interfaces:**
- Produces: `call_number_sort_key(value: str) -> tuple[Any, ...]`
- Produces: `check_call_number_order(call_numbers: Sequence[str]) -> list[OrderStatus]`
- Produces: `render_result_image(image, spines) -> Image.Image`
- Produces: `render_yolo_regions(image, regions, outline) -> Image.Image`
- Produces: `render_ocr_overlay(contact_sheet, rows) -> Image.Image`

- [ ] **Step 1: Write failing call-number tests**

```python
import unittest
from app.analysis.call_number_processing import call_number_sort_key, check_call_number_order


class CallNumberProcessingTests(unittest.TestCase):
    def test_numeric_segments_sort_numerically(self):
        values = ["811.20 B", "811.3 A"]
        self.assertEqual(sorted(values, key=call_number_sort_key), ["811.3 A", "811.20 B"])

    def test_current_compatibility_status_is_preserved(self):
        result = check_call_number_order(["A", "B"])
        self.assertEqual([item.status for item in result], ["ok", "ok"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m unittest tests.test_call_number_processing -v`

Expected: FAIL because the module is absent.

- [ ] **Step 3: Port call-number logic without algorithm changes**

Move `call_number_sort_key` and `check_call_number_order`. Preserve the current placeholder-compatible behavior of marking each item `ok`; do not silently introduce new ordering decisions during the structural refactor.

- [ ] **Step 4: Port all drawing functions**

Move final annotations, YOLO region drawing, OCR token overlays, OCR grid tiling, and label shortening to `result_rendering.py`. Rendering functions return Pillow images and never write files.

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_call_number_processing -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit evaluation and rendering**

```powershell
git add app/analysis/call_number_processing.py app/analysis/result_rendering.py tests/test_call_number_processing.py
git commit -m "refactor: extract call number and rendering responsibilities"
```

---

### Task 5: Add Debug Collection and Optional Local Artifacts

**Files:**
- Create: `app/analysis/debug_artifacts.py`
- Create: `tests/test_debug_artifacts.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `DebugArtifactCollector`
- Constructor: `DebugArtifactCollector(enabled: bool, write_local: bool = False, root: Path | None = None)`
- Produces: `add_image(key, label, image, details=None) -> None`
- Produces: `set_metadata(**values) -> None`
- Produces: `build_payload() -> dict[str, Any]`
- Produces: `write() -> Path | None`

- [ ] **Step 1: Write failing debug tests**

Use `tempfile.TemporaryDirectory` for both tests. Assert that a disabled collector creates no directory. Assert that an enabled local collector creates one run directory containing `manifest.json` and numbered image files, while `build_payload()` contains the existing `stages` list with data URLs.

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest tests.test_debug_artifacts -v`

Expected: FAIL because `DebugArtifactCollector` is absent.

- [ ] **Step 3: Implement the collector**

Use a UTC timestamp plus a short UUID for collision-resistant run IDs. Default the root to `SHELFVISOR_DEBUG_ROOT` when set, otherwise `<project-root>/debug_runs`. Catch `OSError` during local writes, retain the analysis payload, and record `artifactWriteError` in debug metadata.

- [ ] **Step 4: Exclude generated output**

Add exactly this entry to `.gitignore`:

```gitignore
debug_runs/
```

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_debug_artifacts -v`

Expected: all tests PASS and no repository-local debug directory is created by tests.

- [ ] **Step 6: Commit debug artifacts**

```powershell
git add app/analysis/debug_artifacts.py tests/test_debug_artifacts.py .gitignore
git commit -m "feat: add optional local debug artifacts"
```

---

### Task 6: Assemble the Pipeline and Preserve the Public API

**Files:**
- Create: `app/analysis/analysis_pipeline.py`
- Create: `app/analysis/__init__.py`
- Create: `tests/test_analysis_pipeline.py`
- Modify: `app/server.py`
- Remove: `app/analysis.py`
- Remove: `app/yolo_client.py`
- Remove: `app/ocr_client.py`

**Interfaces:**
- Produces: `analyze_shelf_photo(image_bytes, include_debug=False, filter_options=None, write_debug_artifacts=False) -> dict[str, Any]`
- Preserves: `detect_book_spines(image)`, `mock_ocr_call_numbers(count)`, `check_call_number_order`, and `call_number_sort_key` package exports when current callers require them

- [ ] **Step 1: Write a failing pipeline test with inference doubles**

Patch `analysis_pipeline.run_yolo_inference` and `analysis_pipeline.run_ocr_inference`. Feed a small in-memory JPEG, return one deterministic YOLO box and one OCR annotation, then assert that `summary`, `spines`, and `annotatedImage` match the existing response keys. Add a debug-enabled assertion for the existing `debug.stages` labels.

- [ ] **Step 2: Verify the pipeline test fails**

Run: `python -m unittest tests.test_analysis_pipeline -v`

Expected: FAIL because the new pipeline is absent.

- [ ] **Step 3: Implement the thin pipeline**

Port only orchestration from the current `analyze_shelf_photo`. Call the modules in this order: image preparation, YOLO inference, YOLO parsing, filtering, OCR contact-sheet construction, OCR inference, OCR mapping, call-number evaluation, spine construction, final rendering, debug payload building, optional local write.

- [ ] **Step 4: Add the stable package exports**

In `app/analysis/__init__.py`, export `analyze_shelf_photo` and any compatibility helpers still used by tests or scripts. Do not export private rendering, mapping, or filesystem helpers.

- [ ] **Step 5: Switch the server and remove superseded modules**

Ensure `app/server.py` imports only `from .analysis import analyze_shelf_photo` with the existing direct-run fallback if still needed. Remove the three superseded top-level modules only after `rg "yolo_client|ocr_client|app\\.analysis\\.py"` finds no live imports.

- [ ] **Step 6: Run focused and full unit tests**

Run:

```powershell
python -m unittest tests.test_analysis_pipeline -v
python -m unittest discover -s tests -v
```

Expected: pipeline test PASS; full suite PASS without real inference services.

- [ ] **Step 7: Run syntax and import verification**

Run:

```powershell
python -m compileall app tests
python -c "from app.analysis import analyze_shelf_photo; from app.server import app; print(app.title)"
```

Expected: compilation succeeds and the import command prints `ShelfVisor`.

- [ ] **Step 8: Review the final diff for accidental behavior changes**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only planned package, test, server, ignore, and removal changes are present, plus the user's pre-existing changes now represented in their migrated destinations.

- [ ] **Step 9: Commit the completed migration**

```powershell
git add app/analysis app/server.py tests
git add -u app/analysis.py app/yolo_client.py app/ocr_client.py
git commit -m "refactor: modularize image analysis pipeline"
```

---

## Final Verification

- [ ] Run `python -m unittest discover -s tests -v`; expect all tests to pass.
- [ ] Run `python -m compileall app tests`; expect no syntax errors.
- [ ] Run `python -c "from app.analysis import analyze_shelf_photo; from app.server import app; print(app.title)"`; expect `ShelfVisor`.
- [ ] Start the server with `python -m app.server` and verify `GET /api/health` returns `{"status":"ok"}`.
- [ ] With debug disabled, verify no `debug_runs/` directory is created.
- [ ] With debug and local artifact writing enabled, verify one run directory contains `manifest.json` and six ordered stage images.
- [ ] Confirm no real credentials, model binaries, or generated debug images are staged.
