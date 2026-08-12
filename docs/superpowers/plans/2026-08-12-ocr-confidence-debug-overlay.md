# OCR Confidence Debug Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display each upright and rotated OCR token confidence below its corresponding image in the `OCR bounding boxes` debug artifact.

**Architecture:** Keep OCR inference, token mapping, and orientation selection unchanged. Add a small pure caption formatter and rebuild each debug-only OCR row with dynamically sized caption space before the existing three-rows-per-horizontal-tile composition.

**Tech Stack:** Python 3, Pillow (`PIL.Image`, `PIL.ImageDraw`, `PIL.ImageFont`), `unittest`

## Global Constraints

- Show captions below both `upright` and `rotated_ccw_90` images.
- Format one token per line as `boxN: 0.00`, with numbering restarting at `box1` for each orientation.
- Preserve the current token order: vertical coordinate first, horizontal coordinate second.
- Render no caption or placeholder for an orientation with no tokens.
- Preserve the selected orientation's green highlight, all token bounding boxes, OCR data, and the three-row horizontal tile layout.
- Restrict production changes to debug rendering in `app/analysis/result_rendering.py`.

---

### Task 1: Render Per-Orientation Confidence Captions

**Files:**
- Create: `tests/test_result_rendering.py`
- Modify: `app/analysis/result_rendering.py:58-156`

**Interfaces:**
- Consumes: existing OCR row dictionaries containing `variantResults[*].orientation`, `variantResults[*].sheetBox`, and ordered `variantResults[*].tokens[*].confidence`.
- Produces: `_ocr_confidence_labels(tokens: Sequence[dict[str, Any]]) -> list[str]`.
- Produces: `_render_ocr_overlay_row(image: Image.Image, row: dict[str, Any], font: ImageFont.ImageFont) -> Image.Image`.
- Preserves: `grid_ocr_overlay_tiles(image: Image.Image, rows: Sequence[dict[str, Any]], rows_per_tile: int = 3) -> Image.Image`.

- [x] **Step 1: Add failing tests for caption content and independent orientation numbering**

Create `tests/test_result_rendering.py` with a row fixture containing two upright tokens and one rotated token. Test the intended pure formatting interface directly:

```python
import unittest

from PIL import Image, ImageChops

from app.analysis.result_rendering import _ocr_confidence_labels, grid_ocr_overlay_tiles


class OCRDebugRenderingTests(unittest.TestCase):
    def test_confidence_labels_follow_token_order_and_use_two_decimals(self):
        tokens = [
            {"confidence": 0.951},
            {"confidence": 0.874},
        ]

        self.assertEqual(
            _ocr_confidence_labels(tokens),
            ["box1: 0.95", "box2: 0.87"],
        )

    def test_each_orientation_starts_numbering_at_box_one(self):
        upright = [{"confidence": 0.95}, {"confidence": 0.87}]
        rotated = [{"confidence": 0.63}]

        self.assertEqual(_ocr_confidence_labels(upright)[0], "box1: 0.95")
        self.assertEqual(_ocr_confidence_labels(rotated)[0], "box1: 0.63")

    def test_empty_orientation_has_no_confidence_labels(self):
        self.assertEqual(_ocr_confidence_labels([]), [])
```

- [x] **Step 2: Run the caption tests and verify RED**

Run: `python -m unittest tests.test_result_rendering.OCRDebugRenderingTests.test_confidence_labels_follow_token_order_and_use_two_decimals tests.test_result_rendering.OCRDebugRenderingTests.test_each_orientation_starts_numbering_at_box_one tests.test_result_rendering.OCRDebugRenderingTests.test_empty_orientation_has_no_confidence_labels -v`

Expected: ERROR with `ImportError` because `_ocr_confidence_labels` does not exist.

- [x] **Step 3: Add the minimal caption formatter**

Add this helper near the OCR debug rendering functions in `app/analysis/result_rendering.py`:

```python
def _ocr_confidence_labels(tokens: Sequence[dict[str, Any]]) -> list[str]:
    return [
        f"box{index}: {float(token.get('confidence', 0.0) or 0.0):.2f}"
        for index, token in enumerate(tokens, start=1)
    ]
```

- [x] **Step 4: Run the caption tests and verify GREEN**

Run: `python -m unittest tests.test_result_rendering.OCRDebugRenderingTests.test_confidence_labels_follow_token_order_and_use_two_decimals tests.test_result_rendering.OCRDebugRenderingTests.test_each_orientation_starts_numbering_at_box_one tests.test_result_rendering.OCRDebugRenderingTests.test_empty_orientation_has_no_confidence_labels -v`

Expected: all three tests PASS.

- [x] **Step 5: Add failing integration tests for captions below both images and dynamic height**

Extend `tests/test_result_rendering.py` with a helper that builds one OCR row. The token boxes stay inside their respective `sheetBox` values, matching production data:

```python
    def _row(self, upright_confidences, rotated_confidences):
        def tokens(x, y, confidences):
            return [
                {
                    "text": str(index),
                    "box": [x + 4, y + 4 + index * 10, 8, 8],
                    "confidence": confidence,
                }
                for index, confidence in enumerate(confidences)
            ]

        return {
            "index": 1,
            "sheetBox": [20, 20, 40, 60],
            "selectedOrientation": "upright",
            "variantResults": [
                {
                    "orientation": "upright",
                    "sheetBox": [20, 20, 40, 60],
                    "tokens": tokens(20, 20, upright_confidences),
                },
                {
                    "orientation": "rotated_ccw_90",
                    "sheetBox": [100, 30, 60, 40],
                    "tokens": tokens(100, 30, rotated_confidences),
                },
            ],
        }

    def test_grid_adds_caption_pixels_below_both_orientation_images(self):
        row = self._row([0.95], [0.63])
        source = Image.new("RGB", (200, 120), "white")

        rendered = grid_ocr_overlay_tiles(source, [row])

        upright_caption_area = rendered.crop((38, 118, 98, 130))
        rotated_caption_area = rendered.crop((118, 108, 188, 120))
        upright_difference = ImageChops.difference(
            upright_caption_area,
            Image.new("RGB", upright_caption_area.size, "white"),
        )
        rotated_difference = ImageChops.difference(
            rotated_caption_area,
            Image.new("RGB", rotated_caption_area.size, "white"),
        )
        self.assertIsNotNone(upright_difference.getbbox())
        self.assertIsNotNone(rotated_difference.getbbox())

    def test_more_tokens_expand_the_debug_tile_height(self):
        source = Image.new("RGB", (200, 120), "white")

        one_line = grid_ocr_overlay_tiles(source, [self._row([0.95], [])])
        three_lines = grid_ocr_overlay_tiles(source, [self._row([0.95, 0.87, 0.72], [])])

        self.assertGreater(three_lines.height, one_line.height)
```

- [x] **Step 6: Run the integration tests and verify RED**

Run: `python -m unittest tests.test_result_rendering.OCRDebugRenderingTests.test_grid_adds_caption_pixels_below_both_orientation_images tests.test_result_rendering.OCRDebugRenderingTests.test_more_tokens_expand_the_debug_tile_height -v`

Expected: FAIL because the existing grid only crops the fixed-height contact sheet and does not draw confidence captions or expand with token count.

- [x] **Step 7: Implement debug-row composition with per-orientation caption space**

In `app/analysis/result_rendering.py`:

1. Add caption layout constants close to the OCR rendering functions:

```python
OCR_ROW_PADDING = 12
OCR_ROW_SOURCE_PADDING = 8
OCR_CAPTION_GAP = 4
OCR_CAPTION_LINE_GAP = 2
```

2. Implement `_render_ocr_overlay_row` so it:

   - derives tight source crop bounds from a new `_ocr_row_vertical_bounds(row, image.height)` helper using `OCR_ROW_SOURCE_PADDING`, which is smaller than half of the contact sheet's 18-pixel row gap and therefore cannot include pixels from an adjacent OCR row;
   - calls `_ocr_confidence_labels` separately for every item in `row["variantResults"]`;
   - uses `draw.textbbox((0, 0), "box1: 0.00", font=font)` to calculate line height;
   - calculates each caption's local start y as `sheet_y + sheet_height - source_y1 + OCR_CAPTION_GAP`;
   - calculates the output height from the greatest caption bottom plus `OCR_ROW_PADDING`;
   - pastes the source crop once into a white RGB canvas of the expanded height;
   - draws each orientation's labels at `sheet_x`, directly below that orientation image;
   - adds no height or text for an empty label list beyond the source crop's existing padding.

3. Replace each chunk crop in `grid_ocr_overlay_tiles` with a vertical stack of `_render_ocr_overlay_row(image, row, font)` results. Keep the dark `Rows NN-NN` header, horizontal `tile_gap`, `outer_padding`, and three rows per tile. Use `OCR_ROW_PADDING` between composed rows and retain the full source image width.

The core row composer should have this shape:

```python
def _render_ocr_overlay_row(
    image: Image.Image,
    row: dict[str, Any],
    font: ImageFont.ImageFont,
) -> Image.Image:
    source_y1, source_y2 = _ocr_row_vertical_bounds(row, image.height)
    source = image.crop((0, source_y1, image.width, source_y2))
    captions = []
    required_height = source.height
    line_box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
        (0, 0), "box1: 0.00", font=font
    )
    line_height = line_box[3] - line_box[1]
    for variant in row.get("variantResults", []):
        labels = _ocr_confidence_labels(variant.get("tokens", []))
        if not labels:
            continue
        x, y, _width, height = variant["sheetBox"]
        start_y = y + height - source_y1 + OCR_CAPTION_GAP
        caption_bottom = start_y + len(labels) * line_height + (len(labels) - 1) * OCR_CAPTION_LINE_GAP
        required_height = max(required_height, caption_bottom + OCR_ROW_PADDING)
        captions.append((x, start_y, labels))
    output = Image.new("RGB", (source.width, required_height), "white")
    output.paste(source, (0, 0))
    draw = ImageDraw.Draw(output)
    for x, start_y, labels in captions:
        for line_index, label in enumerate(labels):
            draw.text(
                (x, start_y + line_index * (line_height + OCR_CAPTION_LINE_GAP)),
                label,
                fill=(51, 65, 85),
                font=font,
            )
    return output
```

Add the tight-bound helper and remove `_ocr_chunk_vertical_bounds` after its last caller is replaced:

```python
def _ocr_row_vertical_bounds(row: dict[str, Any], image_height: int) -> tuple[int, int]:
    y_values = []
    for variant in row.get("variantResults", []):
        _x, y, _width, height = variant["sheetBox"]
        y_values.extend([y, y + height])
    if "sheetBox" in row:
        _x, y, _width, height = row["sheetBox"]
        y_values.extend([y, y + height])
    if not y_values:
        return 0, image_height
    y1 = clamp(min(y_values) - OCR_ROW_SOURCE_PADDING, 0, image_height - 1)
    return y1, clamp(max(y_values) + OCR_ROW_SOURCE_PADDING, y1 + 1, image_height)
```

- [x] **Step 8: Run the result-rendering tests and verify GREEN**

Run: `python -m unittest tests.test_result_rendering -v`

Expected: all tests PASS with no errors or warnings.

- [x] **Step 9: Run the complete regression suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS. In particular, `tests.test_analysis_pipeline.AnalysisPipelineTests.test_debug_payload_contains_ordered_stage_labels` still reports the same six debug stages in the same order.

- [x] **Step 10: Generate and review a deterministic debug preview**

Create the ignored preview directory:

Run: `New-Item -ItemType Directory -Force tmp`

Generate a preview from the test fixture without YOLO, Google Vision, network access, or credentials:

Run: `python -c "from PIL import Image; from tests.test_result_rendering import OCRDebugRenderingTests; from app.analysis.result_rendering import grid_ocr_overlay_tiles, render_ocr_overlay; case=OCRDebugRenderingTests(); row=case._row([0.95,0.87],[0.63]); source=Image.new('RGB',(200,120),'white'); overlay=render_ocr_overlay(source,[row]); grid_ocr_overlay_tiles(overlay,[row]).save('tmp/ocr-confidence-debug-preview.png')"`

Open `tmp/ocr-confidence-debug-preview.png` with the local image viewer and verify:

- both orientations show `boxN: 0.00` captions directly below their own images;
- numbering restarts at `box1` for the second orientation;
- captions do not overlap the next OCR row;
- the selected orientation retains its green outline;
- each horizontal tile still contains at most three OCR rows.

- [x] **Step 11: Commit the implementation**

```bash
git add app/analysis/result_rendering.py tests/test_result_rendering.py docs/superpowers/plans/2026-08-12-ocr-confidence-debug-overlay.md
git commit -m "feat: show OCR confidence labels in debug image"
```
