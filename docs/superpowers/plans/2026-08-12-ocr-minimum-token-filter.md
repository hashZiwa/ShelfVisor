# OCR Minimum Token Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject OCR orientations with at most two tokens, show rejected candidates in red in `OCR bounding boxes`, and remove fully rejected image sets from all later analysis.

**Architecture:** OCR processing records `eligible` on every orientation and row while preserving every row for debugging. The pipeline pairs rows with their corresponding YOLO regions, filters both collections together for final analysis, and keeps the unfiltered rows only for the OCR debug artifact.

**Tech Stack:** Python 3, Pillow, `unittest`

## Global Constraints

- An orientation needs at least 3 OCR token boxes to be eligible.
- Select only among eligible orientations; one eligible orientation wins regardless of confidence, while two use existing average-confidence selection.
- If both orientations are rejected, preserve the row through `OCR bounding boxes` and remove it from every later result.
- Render each rejected orientation with a red image border while preserving its token boxes and confidence captions.
- Filter OCR rows and corresponding YOLO regions together, then renumber final spines from 1.
- Preserve OCR contact-sheet contents, token sorting, confidence calculation, debug-stage order, and response field names.
- Do not perform Git staging, commits, merges, pushes, or PR operations in this session.

---

### Task 1: Record Orientation and Row Eligibility

**Files:**
- Modify: `tests/test_ocr_processing.py`
- Modify: `app/analysis/ocr/ocr_processing.py`

**Interfaces:**
- Produces: `MIN_OCR_TOKENS = 3`.
- Adds: `variantResults[*].eligible: bool`.
- Adds: `row["eligible"]: bool`.
- Preserves: `map_ocr_result_to_rows(...) -> list[dict[str, Any]]` and all existing fields.

- [x] Add a two-variant synthetic contact-sheet helper with non-overlapping boxes.

```python
def _two_variant_sheet(self):
    return OCRContactSheet(
        Image.new("RGB", (200, 200), "white"),
        [{
            "index": 1,
            "sourceBox": [0, 0, 100, 100],
            "sheetBox": [0, 0, 90, 200],
            "variants": [
                {"orientation": "upright", "sheetBox": [0, 0, 90, 200]},
                {"orientation": "rotated_ccw_90", "sheetBox": [110, 0, 90, 200]},
            ],
            "regionBox": [0, 0, 100, 100],
        }],
    )
```

- [x] Add failing tests proving: two tokens are rejected; three tokens are eligible; a sole eligible orientation is selected despite lower confidence; two eligible orientations use average confidence; two rejected orientations produce `selectedOrientation=None`, empty final tokens/text, and `eligible=False`.

```python
def test_only_orientation_with_at_least_three_tokens_can_be_selected(self):
    annotations = [
        {"text": f"u{i}", "box": [10, 10 + i * 20, 30, 10], "confidence": 0.4}
        for i in range(3)
    ] + [
        {"text": f"r{i}", "box": [120, 10 + i * 20, 30, 10], "confidence": 0.99}
        for i in range(2)
    ]
    row = map_ocr_result_to_rows({"annotations": annotations}, self._two_variant_sheet())[0]
    self.assertTrue(row["eligible"])
    self.assertEqual(row["selectedOrientation"], "upright")
    self.assertEqual([variant["eligible"] for variant in row["variantResults"]], [True, False])

def test_two_eligible_orientations_still_select_higher_average_confidence(self):
    annotations = [
        {"text": f"u{i}", "box": [10, 10 + i * 20, 30, 10], "confidence": 0.6}
        for i in range(3)
    ] + [
        {"text": f"r{i}", "box": [120, 10 + i * 20, 30, 10], "confidence": 0.9}
        for i in range(3)
    ]
    row = map_ocr_result_to_rows({"annotations": annotations}, self._two_variant_sheet())[0]
    self.assertEqual(row["selectedOrientation"], "rotated_ccw_90")

def test_row_is_rejected_when_both_orientations_have_at_most_two_tokens(self):
    annotations = [
        {"text": f"u{i}", "box": [10, 10 + i * 20, 30, 10], "confidence": 0.9}
        for i in range(2)
    ] + [
        {"text": f"r{i}", "box": [120, 10 + i * 20, 30, 10], "confidence": 0.9}
        for i in range(2)
    ]
    row = map_ocr_result_to_rows({"annotations": annotations}, self._two_variant_sheet())[0]
    self.assertFalse(row["eligible"])
    self.assertIsNone(row["selectedOrientation"])
    self.assertEqual(row["tokens"], [])
    self.assertEqual(row["text"], "")
```
- [x] Run `python -m unittest tests.test_ocr_processing -v` and confirm failures are caused by missing eligibility behavior.
- [x] Initialize variants with `eligible=False`, calculate `eligible = len(tokens) >= MIN_OCR_TOKENS` after token sorting, select with `max(..., averageConfidence)` only from eligible variants, and explicitly clear final row fields when no candidate survives.

```python
MIN_OCR_TOKENS = 3

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
```
- [x] Update the existing single-annotation mapping test to use three annotations because one token is no longer a valid selected OCR result.
- [x] Run `python -m unittest tests.test_ocr_processing -v` and confirm all OCR processing tests pass.

### Task 2: Render Rejected OCR Candidates in Red

**Files:**
- Modify: `tests/test_result_rendering.py`
- Modify: `app/analysis/result_rendering.py`

**Interfaces:**
- Consumes: `variant["eligible"]` and `row["eligible"]` from Task 1.
- Preserves: `render_ocr_overlay(...) -> Image.Image` and `build_ocr_debug_details(...) -> list[str]`.

- [x] Add a failing Pillow test with one rejected and one selected eligible orientation; inspect border pixels to prove the rejected image is red and the selected image remains green.

```python
def test_rejected_orientation_is_red_while_selected_orientation_stays_green(self):
    row = self._row([0.95], [0.80, 0.81, 0.82])
    row["selectedOrientation"] = "rotated_ccw_90"
    row["eligible"] = True
    row["variantResults"][0]["eligible"] = False
    row["variantResults"][1]["eligible"] = True
    rendered = render_ocr_overlay(Image.new("RGB", (200, 120), "white"), [row])
    rejected_pixel = rendered.getpixel((20, 20))
    selected_pixel = rendered.getpixel((100, 30))
    self.assertGreater(rejected_pixel[0], rejected_pixel[1] * 2)
    self.assertGreater(selected_pixel[1], selected_pixel[0] * 2)
```
- [x] Add a failing detail test proving rejected orientations and fully rejected rows are explicitly labeled `rejected`.

```python
def test_debug_details_label_rejected_orientation_and_row(self):
    row = self._row([0.95], [0.63])
    row["eligible"] = False
    row["selectedOrientation"] = None
    for variant in row["variantResults"]:
        variant["eligible"] = False
    detail = build_ocr_debug_details([row])[0]
    self.assertIn("Row 01 rejected", detail)
    self.assertIn("upright rejected", detail)
    self.assertIn("rotated_ccw_90 rejected", detail)
```
- [x] Run `python -m unittest tests.test_result_rendering -v` and confirm the new tests fail against the existing blue/purple borders and status-free details.
- [x] In `render_ocr_overlay`, give `eligible is False` precedence over selection colors and draw a red border; leave token rectangles and confidence captions unchanged.

```python
is_rejected = variant.get("eligible") is False
border_color = (
    (220, 38, 38, 245)
    if is_rejected
    else ((34, 197, 94, 245) if is_selected else color)
)
border_width = 4 if is_rejected else (6 if is_selected else 2)
draw.rectangle((x, y, x + width, y + height), outline=border_color, width=border_width)
```
- [x] In `build_ocr_debug_details`, append `eligible` or `rejected` to each orientation description and mark a row rejected when `row["eligible"] is False`. Treat missing eligibility fields as eligible so existing isolated rendering fixtures remain compatible.

```python
eligibility = "rejected" if variant.get("eligible") is False else "eligible"
row_status = " rejected" if row.get("eligible") is False else ""
details.append(f'Row {row["index"]:02d}{row_status}: ' + " | ".join(parts))
```
- [x] Run `python -m unittest tests.test_result_rendering -v` and confirm all rendering tests pass.

### Task 3: Remove Fully Rejected Sets After OCR Debugging

**Files:**
- Modify: `tests/test_analysis_pipeline.py`
- Modify: `app/analysis/analysis_pipeline.py`

**Interfaces:**
- Consumes: ordered `(DetectedRegion, OCR row)` pairs and `row["eligible"]`.
- Produces: final `regions` and `ocr_rows` containing only eligible pairs.
- Preserves: unfiltered OCR rows for debug metadata, overlay rendering, and detail strings.

- [x] Change the existing pipeline OCR fixture to contain three tokens so its single row remains eligible under the new rule.

```python
"annotations": [
    {"text": "811.3", "box": [100, 40, 10, 10], "confidence": 0.95},
    {"text": "A", "box": [100, 55, 10, 10], "confidence": 0.95},
    {"text": "1", "box": [100, 70, 10, 10], "confidence": 0.95},
]
```
- [x] Add a failing pipeline test with two YOLO regions and a deterministic two-row contact sheet: reject the first row with at most two tokens in both orientations, accept the second with three tokens, and assert `bookCount == 1`, one final spine numbered `1`, and only the second region's geometry in that spine.

```python
@patch("app.analysis.analysis_pipeline.run_ocr_inference")
@patch("app.analysis.analysis_pipeline.build_ocr_contact_sheet")
@patch("app.analysis.analysis_pipeline.run_yolo_inference")
def test_fully_rejected_row_is_debugged_then_removed_from_final_results(
    self, run_yolo, build_sheet, run_ocr
):
    run_yolo.return_value = {
        "predictions": [
            {"x": 25, "y": 50, "width": 20, "height": 40, "confidence": 0.9, "class": "book"},
            {"x": 75, "y": 50, "width": 20, "height": 40, "confidence": 0.9, "class": "book"},
        ],
        "model_id": "test-model.pt",
    }
    build_sheet.return_value = OCRContactSheet(Image.new("RGB", (200, 200), "white"), [
        {"index": 1, "sheetBox": [0, 0, 80, 80], "variants": [
            {"orientation": "upright", "sheetBox": [0, 0, 80, 80]},
            {"orientation": "rotated_ccw_90", "sheetBox": [100, 0, 80, 80]},
        ]},
        {"index": 2, "sheetBox": [0, 100, 80, 80], "variants": [
            {"orientation": "upright", "sheetBox": [0, 100, 80, 80]},
            {"orientation": "rotated_ccw_90", "sheetBox": [100, 100, 80, 80]},
        ]},
    ])
    run_ocr.return_value = {"fullText": "", "annotations": [
        {"text": "bad1", "box": [10, 10, 20, 10], "confidence": 0.9},
        {"text": "bad2", "box": [10, 30, 20, 10], "confidence": 0.9},
        {"text": "good1", "box": [10, 110, 20, 10], "confidence": 0.9},
        {"text": "good2", "box": [10, 130, 20, 10], "confidence": 0.9},
        {"text": "good3", "box": [10, 150, 20, 10], "confidence": 0.9},
    ]}
    result = analyze_shelf_photo(self.image_bytes, include_debug=True)
    self.assertEqual(result["summary"]["bookCount"], 1)
    self.assertEqual(result["spines"][0]["index"], 1)
    self.assertEqual(result["spines"][0]["x"], 65)
    self.assertEqual(result["debug"]["ocrSheet"]["rowCount"], 2)
    self.assertFalse(result["debug"]["ocrSheet"]["rows"][0]["eligible"])
```
- [x] In the same test enable debug and assert `debug.ocrSheet.rowCount == 2`, the first debug row is rejected, and the `OCR bounding boxes` stage is still present before `Final result`.
- [x] Run `python -m unittest tests.test_analysis_pipeline -v` and confirm the filtering assertions fail against the current index-based pipeline.
- [x] Keep `all_ocr_rows` from OCR mapping, build eligible `(region, row)` pairs with `zip`, and derive final `regions` and `ocr_rows` from those pairs before call-number checking and spine creation.

```python
all_ocr_rows = map_ocr_result_to_rows(ocr_result, contact_sheet)
eligible_pairs = [
    (region, row)
    for region, row in zip(regions, all_ocr_rows)
    if row["eligible"]
]
regions = [region for region, _row in eligible_pairs]
ocr_rows = [row for _region, row in eligible_pairs]
```
- [x] Use `all_ocr_rows` for `render_ocr_overlay`, `grid_ocr_overlay_tiles`, `build_ocr_debug_details`, and `ocrSheet` debug metadata; use filtered rows everywhere after that conceptual debug boundary.

```python
ocr_overlay = grid_ocr_overlay_tiles(
    render_ocr_overlay(contact_sheet.image, all_ocr_rows),
    all_ocr_rows,
)
collector.set_metadata(ocrSheet={"rowCount": len(all_ocr_rows), "rows": all_ocr_rows})
collector.add_image(
    "ocr-bounding-boxes",
    "OCR bounding boxes",
    ocr_overlay,
    build_ocr_debug_details(all_ocr_rows),
)
```
- [x] Run `python -m unittest tests.test_analysis_pipeline -v` and confirm all pipeline tests pass.

### Task 4: Full Verification

**Files:**
- Verify all modified production, test, design, and plan files.

- [x] Run `python -m unittest discover -s tests -v` and confirm all tests pass.
- [x] Run `python -m compileall -q app tests` and confirm exit code 0 with no output.
- [x] Run `git diff --check` and confirm no whitespace errors; line-ending conversion warnings are acceptable.
- [x] Review `git diff` and confirm changes are limited to OCR eligibility, rejected debug rendering, paired pipeline filtering, tests, and approved documentation.
