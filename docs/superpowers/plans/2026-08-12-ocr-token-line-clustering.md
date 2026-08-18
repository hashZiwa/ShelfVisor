# OCR Token Line Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group OCR token boxes into horizontal or vertical lines using 50% interval overlap, then order each line according to box shape and image orientation.

**Architecture:** Extend the existing average-box-shape sorter with pairwise same-line detection and connected-component grouping. Keep OCR mapping, confidence selection, and minimum-token filtering unchanged; only the order of each variant's token list changes.

**Tech Stack:** Python 3, `unittest`

## Global Constraints

- Average width greater than or equal to average height means horizontal boxes; otherwise vertical boxes.
- Horizontal boxes use vertical overlap to form horizontal lines; vertical boxes use horizontal overlap to form vertical lines.
- Two intervals belong to the same line when overlap divided by the shorter interval length is at least `0.5`.
- Horizontal lines are ordered top-to-bottom and read left-to-right for both orientations.
- Vertical lines are ordered left-to-right; `upright` reads top-to-bottom and `rotated_ccw_90` reads bottom-to-top inside each line.
- Zero-length intervals do not join another token's line.
- Preserve minimum-token eligibility, average confidence, selected-orientation behavior, debug rendering, and pipeline filtering.
- Do not perform Git staging, commits, merges, pushes, or PR operations in this session.

---

### Task 1: Prove Orientation-Aware Line Reading Behavior

**Files:**
- Modify: `tests/test_ocr_processing.py`

**Interfaces:**
- Consumes: `map_ocr_result_to_rows(...)` with a single variant whose orientation is configurable.
- Verifies: observable `row["tokens"]` and `row["text"]` order.

- [x] **Step 1: Generalize the single-variant test sheet**

Change the helper to accept an orientation while preserving current callers:

```python
def _single_variant_sheet(self, orientation="upright"):
    return OCRContactSheet(
        Image.new("RGB", (200, 200), "white"),
        [{
            "index": 1,
            "sourceBox": [0, 0, 100, 100],
            "sheetBox": [0, 0, 200, 200],
            "variants": [{"orientation": orientation, "sheetBox": [0, 0, 200, 200]}],
            "regionBox": [0, 0, 100, 100],
        }],
    )
```

- [x] **Step 2: Add failing same-horizontal-line test**

```python
def test_horizontal_line_ignores_small_y_differences_and_reads_left_to_right(self):
    rows = map_ocr_result_to_rows(
        {"annotations": [
            {"text": "right", "box": [110, 14, 40, 20], "confidence": 0.9},
            {"text": "left", "box": [10, 10, 40, 20], "confidence": 0.9},
            {"text": "middle", "box": [60, 18, 40, 20], "confidence": 0.9},
        ]},
        self._single_variant_sheet(),
    )
    self.assertEqual(
        [token["text"] for token in rows[0]["tokens"]],
        ["left", "middle", "right"],
    )
```

- [x] **Step 3: Add failing upright and rotated vertical-line tests**

```python
def test_upright_vertical_line_ignores_small_x_differences_and_reads_down(self):
    rows = map_ocr_result_to_rows(
        {"annotations": [
            {"text": "bottom", "box": [14, 110, 20, 40], "confidence": 0.9},
            {"text": "top", "box": [10, 10, 20, 40], "confidence": 0.9},
            {"text": "middle", "box": [18, 60, 20, 40], "confidence": 0.9},
        ]},
        self._single_variant_sheet("upright"),
    )
    self.assertEqual(
        [token["text"] for token in rows[0]["tokens"]],
        ["top", "middle", "bottom"],
    )

def test_rotated_vertical_line_reads_from_bottom_to_top(self):
    rows = map_ocr_result_to_rows(
        {"annotations": [
            {"text": "top", "box": [10, 10, 20, 40], "confidence": 0.9},
            {"text": "bottom", "box": [14, 110, 20, 40], "confidence": 0.9},
            {"text": "middle", "box": [18, 60, 20, 40], "confidence": 0.9},
        ]},
        self._single_variant_sheet("rotated_ccw_90"),
    )
    self.assertEqual(
        [token["text"] for token in rows[0]["tokens"]],
        ["bottom", "middle", "top"],
    )
```

- [x] **Step 4: Run the three tests and verify RED**

Run: `python -m unittest tests.test_ocr_processing.OCRProcessingTests.test_horizontal_line_ignores_small_y_differences_and_reads_left_to_right tests.test_ocr_processing.OCRProcessingTests.test_upright_vertical_line_ignores_small_x_differences_and_reads_down tests.test_ocr_processing.OCRProcessingTests.test_rotated_vertical_line_reads_from_bottom_to_top -v`

Expected: all three FAIL because the current sorter prioritizes raw `y` or `x` and does not receive orientation.

### Task 2: Implement 50% Overlap Line Clustering

**Files:**
- Modify: `app/analysis/ocr/ocr_processing.py`

**Interfaces:**
- Produces: `LINE_OVERLAP_THRESHOLD = 0.5`.
- Produces: `_interval_overlap_ratio(first_start: float, first_length: float, second_start: float, second_length: float) -> float`.
- Produces: `_group_tokens_by_overlap(tokens: list[dict[str, Any]], start_index: int, length_index: int) -> list[list[dict[str, Any]]]`.
- Changes: `_sort_tokens_by_average_box_shape(tokens: list[dict[str, Any]], orientation: str) -> None`.

- [x] **Step 1: Add overlap-ratio helper**

```python
LINE_OVERLAP_THRESHOLD = 0.5

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
```

- [x] **Step 2: Add connected-component grouping helper**

Build an undirected adjacency list for every token pair whose relevant intervals overlap by at least `LINE_OVERLAP_THRESHOLD`. Traverse from token indices in input order with a stack, collecting each connected component as one line. Return groups containing the original token dictionaries.

```python
def _group_tokens_by_overlap(tokens, start_index, length_index):
    adjacency = [[] for _token in tokens]
    for first_index in range(len(tokens)):
        for second_index in range(first_index + 1, len(tokens)):
            first_box = tokens[first_index]["box"]
            second_box = tokens[second_index]["box"]
            ratio = _interval_overlap_ratio(
                first_box[start_index], first_box[length_index],
                second_box[start_index], second_box[length_index],
            )
            if ratio >= LINE_OVERLAP_THRESHOLD:
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
```

- [x] **Step 3: Replace the fixed sorter with line ordering**

For horizontal boxes, group with `(start_index=1, length_index=3)`, sort lines by average vertical center, and sort tokens inside each line by `(x, y)`. For vertical boxes, group with `(0, 2)`, sort lines by average horizontal center, then use `(y, x)` for `upright` and `(-y, x)` for `rotated_ccw_90`. Flatten the lines back into `tokens[:]`.

```python
def _sort_tokens_by_average_box_shape(tokens, orientation):
    if not tokens:
        return
    average_width = sum(token["box"][2] for token in tokens) / len(tokens)
    average_height = sum(token["box"][3] for token in tokens) / len(tokens)
    is_horizontal = average_width >= average_height
    lines = _group_tokens_by_overlap(tokens, 1, 3) if is_horizontal else _group_tokens_by_overlap(tokens, 0, 2)
    center_index, length_index = (1, 3) if is_horizontal else (0, 2)
    lines.sort(key=lambda line: sum(
        token["box"][center_index] + token["box"][length_index] / 2
        for token in line
    ) / len(line))
    for line in lines:
        if is_horizontal:
            line.sort(key=lambda token: (token["box"][0], token["box"][1]))
        elif orientation == "rotated_ccw_90":
            line.sort(key=lambda token: (-token["box"][1], token["box"][0]))
        else:
            line.sort(key=lambda token: (token["box"][1], token["box"][0]))
    tokens[:] = [token for line in lines for token in line]
```

- [x] **Step 4: Pass variant orientation from OCR mapping**

```python
_sort_tokens_by_average_box_shape(variant["tokens"], variant["orientation"])
```

- [x] **Step 5: Run the three focused tests and verify GREEN**

Run the command from Task 1 Step 4. Expected: all three PASS.

### Task 3: Protect Multiple Lines and the 50% Boundary

**Files:**
- Modify: `tests/test_ocr_processing.py`

- [x] **Step 1: Add a horizontal multiple-line test**

```python
def test_horizontal_lines_are_ordered_top_to_bottom_then_left_to_right(self):
    rows = map_ocr_result_to_rows(
        {"annotations": [
            {"text": "top-right", "box": [100, 14, 40, 20], "confidence": 0.9},
            {"text": "bottom-right", "box": [100, 84, 40, 20], "confidence": 0.9},
            {"text": "bottom-left", "box": [10, 80, 40, 20], "confidence": 0.9},
            {"text": "top-left", "box": [10, 10, 40, 20], "confidence": 0.9},
        ]},
        self._single_variant_sheet(),
    )
    self.assertEqual(
        [token["text"] for token in rows[0]["tokens"]],
        ["top-left", "top-right", "bottom-left", "bottom-right"],
    )
```

- [x] **Step 2: Add a rotated multiple-column test**

```python
def test_rotated_vertical_columns_are_left_to_right_and_bottom_to_top(self):
    rows = map_ocr_result_to_rows(
        {"annotations": [
            {"text": "right-top", "box": [110, 10, 20, 40], "confidence": 0.9},
            {"text": "left-middle", "box": [14, 60, 20, 40], "confidence": 0.9},
            {"text": "right-bottom", "box": [114, 110, 20, 40], "confidence": 0.9},
            {"text": "left-top", "box": [10, 10, 20, 40], "confidence": 0.9},
            {"text": "right-middle", "box": [118, 60, 20, 40], "confidence": 0.9},
            {"text": "left-bottom", "box": [18, 110, 20, 40], "confidence": 0.9},
        ]},
        self._single_variant_sheet("rotated_ccw_90"),
    )
    self.assertEqual(
        [token["text"] for token in rows[0]["tokens"]],
        [
            "left-bottom", "left-middle", "left-top",
            "right-bottom", "right-middle", "right-top",
        ],
    )
```

- [x] **Step 3: Add an exact-50%-overlap test**

```python
def test_exactly_half_overlapping_intervals_share_a_horizontal_line(self):
    rows = map_ocr_result_to_rows(
        {"annotations": [
            {"text": "right", "box": [110, 10, 40, 20], "confidence": 0.9},
            {"text": "left", "box": [10, 20, 40, 20], "confidence": 0.9},
            {"text": "middle", "box": [60, 15, 40, 20], "confidence": 0.9},
        ]},
        self._single_variant_sheet(),
    )
    self.assertEqual(
        [token["text"] for token in rows[0]["tokens"]],
        ["left", "middle", "right"],
    )
```

- [x] **Step 4: Run OCR processing tests**

Run: `python -m unittest tests.test_ocr_processing -v`

Expected: all OCR processing tests PASS.

### Task 4: Full Verification

- [x] Run `python -m unittest discover -s tests -v` and confirm all tests pass.
- [x] Run `python -m compileall -q app tests` and confirm exit code 0 with no output.
- [x] Run `git diff --check` and confirm no whitespace errors; line-ending warnings are acceptable.
- [x] Review `git diff` and confirm changes are limited to OCR line clustering, orientation-aware token ordering, tests, and approved documents.
