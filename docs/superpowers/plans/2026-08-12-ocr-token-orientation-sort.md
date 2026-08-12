# OCR Token Orientation-Aware Sort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sort OCR tokens top-to-bottom or left-to-right according to the average aspect ratio of the token boxes in each orientation image.

**Architecture:** Keep annotation-to-variant mapping, confidence aggregation, and orientation selection unchanged. Add a pure in-place token sorting helper in `ocr_processing.py`, then invoke it once per variant before text assembly.

**Tech Stack:** Python 3, `unittest`

## Global Constraints

- Decide independently for every `upright` and `rotated_ccw_90` variant.
- If average box width is greater than or equal to average box height, sort by `(y, x)`.
- If average box width is less than average box height, sort by `(x, y)`.
- Use each box's existing `[x, y, width, height]` values and preserve deterministic secondary ordering.
- Empty and one-token lists remain valid.
- Do not change confidence calculation, orientation selection, contact-sheet construction, debug layout, or response fields.
- Do not perform Git staging, commits, merges, pushes, or PR operations in this session.

---

### Task 1: Add Orientation-Aware Token Sorting

**Files:**
- Modify: `tests/test_ocr_processing.py`
- Modify: `app/analysis/ocr/ocr_processing.py:70-110`

**Interfaces:**
- Consumes: token dictionaries whose `box` field is `[x, y, width, height]`.
- Produces: `_sort_tokens_by_average_box_shape(tokens: list[dict[str, Any]]) -> None`.
- Preserves: `map_ocr_result_to_rows(ocr_result: dict[str, Any], contact_sheet: OCRContactSheet) -> list[dict[str, Any]]`.

- [x] **Step 1: Add a test helper that creates a one-variant contact sheet**

Extend `tests/test_ocr_processing.py` imports and class with:

```python
from app.analysis.analysis_models import DetectedRegion, OCRContactSheet


    def _single_variant_sheet(self):
        return OCRContactSheet(
            Image.new("RGB", (200, 200), "white"),
            [{
                "index": 1,
                "sourceBox": [0, 0, 100, 100],
                "sheetBox": [0, 0, 200, 200],
                "variants": [{"orientation": "upright", "sheetBox": [0, 0, 200, 200]}],
                "regionBox": [0, 0, 100, 100],
            }],
        )
```

- [x] **Step 2: Add failing tests for horizontal, vertical, and equal average shapes**

Add these tests. Annotation input order intentionally differs from expected reading order:

```python
    def test_wide_tokens_are_sorted_top_to_bottom(self):
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "bottom", "box": [10, 100, 40, 10], "confidence": 0.9},
                {"text": "top", "box": [80, 20, 40, 10], "confidence": 0.9},
            ]},
            self._single_variant_sheet(),
        )

        self.assertEqual([token["text"] for token in rows[0]["tokens"]], ["top", "bottom"])
        self.assertEqual(rows[0]["text"], "top bottom")

    def test_tall_tokens_are_sorted_left_to_right(self):
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "right", "box": [100, 10, 10, 40], "confidence": 0.9},
                {"text": "left", "box": [20, 80, 10, 40], "confidence": 0.9},
            ]},
            self._single_variant_sheet(),
        )

        self.assertEqual([token["text"] for token in rows[0]["tokens"]], ["left", "right"])
        self.assertEqual(rows[0]["text"], "left right")

    def test_equal_average_width_and_height_sort_top_to_bottom(self):
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "bottom", "box": [10, 100, 20, 20], "confidence": 0.9},
                {"text": "top", "box": [80, 20, 20, 20], "confidence": 0.9},
            ]},
            self._single_variant_sheet(),
        )

        self.assertEqual([token["text"] for token in rows[0]["tokens"]], ["top", "bottom"])
        self.assertEqual(rows[0]["text"], "top bottom")
```

- [x] **Step 3: Run the three tests and verify RED**

Run: `python -m unittest tests.test_ocr_processing.OCRProcessingTests.test_wide_tokens_are_sorted_top_to_bottom tests.test_ocr_processing.OCRProcessingTests.test_tall_tokens_are_sorted_left_to_right tests.test_ocr_processing.OCRProcessingTests.test_equal_average_width_and_height_sort_top_to_bottom -v`

Expected: the tall-token test FAILS with actual token order `['right', 'left']`; wide and equal cases PASS because they match the existing `(y, x)` behavior.

- [x] **Step 4: Implement the minimal sorting helper**

Add this helper to `app/analysis/ocr/ocr_processing.py`:

```python
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
```

Replace the existing fixed sort in `map_ocr_result_to_rows`:

```python
            _sort_tokens_by_average_box_shape(variant["tokens"])
```

- [x] **Step 5: Run the OCR processing tests and verify GREEN**

Run: `python -m unittest tests.test_ocr_processing -v`

Expected: all OCR processing tests PASS.

- [x] **Step 6: Run the complete regression suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS, including OCR confidence-label rendering tests whose numbering follows the reordered token list.

- [x] **Step 7: Run syntax and diff checks**

Run: `python -m compileall -q app tests`

Expected: exit code 0 with no output.

Run: `git diff --check`

Expected: no whitespace errors. Line-ending conversion warnings are acceptable.
