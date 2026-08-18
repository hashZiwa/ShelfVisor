# OCR Call-Number Reconstruction and Candidate Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct a canonical call number from noisy, variably split OCR tokens and select the better image variant by structure score, lower-quartile confidence, median confidence, and average confidence.

**Architecture:** Add a pure dynamic-programming reconstruction module that preserves character order while allowing penalized deletions. Keep OCR token mapping and confidence-based candidate selection in `ocr_processing.py`, and expose the selection evidence through the existing debug renderer.

**Tech Stack:** Python 3, standard-library `dataclasses`, `functools`, `math`, `statistics`, existing `unittest` suite

**Spec:** `docs/superpowers/specs/2026-08-18-ocr-call-number-reconstruction-scoring-design.md`

## Global Constraints

- OCR tokens and characters must never be reordered.
- Reconstruction may delete input characters and join existing fragments, but may not substitute characters or invent punctuation, digits, or letters.
- Major classification is `N00`, including `000`.
- Detailed classification is exactly three digits with an optional `.` plus one or more digits.
- A reconstructed detail receives a consistency adjustment based on whether its first digit matches the major's first digit.
- Book/author symbols contain one or more Unicode letters followed by zero to three digits, repeated without presentation spaces.
- A trailing `v.number` or `c.number` suffix is case-insensitive on input and lowercase in output.
- Numeric substrings stay as strings so leading zeroes such as `005` survive.
- Token count eligibility and confidence statistics use all original OCR tokens, including discarded noise.
- Structure differences of at most 2 points advance to confidence comparison.
- Preserve existing raw `text`, `tokens`, and average-confidence fields.
- Do not change OCR inference, token reading order, or normal result UI structure.

---

### Task 1: Pure Noise-Tolerant Call-Number Reconstructor

**Files:**
- Create: `app/analysis/ocr/call_number_reconstruction.py`
- Create: `tests/test_call_number_reconstruction.py`

**Interfaces:**
- Consumes: `reconstruct_call_number(token_texts: Sequence[object])`.
- Produces: frozen `ReconstructionResult` with `text: str`, `major: str`, `detail: str`, `symbols: str`, `suffix: str`, `structure_score: int`, `retained_positions: tuple[int, ...]`, `discarded_character_count: int`, and `completed_section_count: int`.
- Internal DP returns immutable `_ParsePath` values and compares them by `(score, completed_section_count, -discarded_character_count, retained-position preference)`.

- [ ] **Step 1: Write failing tests for canonical sections, split independence, and leading zeroes**

```python
import unittest

from app.analysis.ocr.call_number_reconstruction import reconstruct_call_number


class CallNumberReconstructionTests(unittest.TestCase):
    def test_token_splits_do_not_change_reconstruction(self):
        combined = reconstruct_call_number(["500", "519.5", "ㅅ", "21", "c.2"])
        split = reconstruct_call_number(["5", "00", "519", ".5", "ㅅ21", "c", ".2"])

        self.assertEqual(combined.text, "500 519.5 ㅅ21 c.2")
        self.assertEqual(split.text, combined.text)
        self.assertEqual(split.structure_score, combined.structure_score)

    def test_integer_detail_is_valid(self):
        result = reconstruct_call_number(["500", "519", "ㅅ21"])
        self.assertEqual(result.text, "500 519 ㅅ21")

    def test_zero_major_preserves_detail_leading_zero(self):
        result = reconstruct_call_number(["000", "005.12", "ㄱ7"])
        self.assertEqual(result.major, "000")
        self.assertEqual(result.detail, "005.12")
        self.assertEqual(result.text, "000 005.12 ㄱ7")

    def test_major_detail_match_scores_above_conflict(self):
        matching = reconstruct_call_number(["500", "519", "ㅅ21"])
        conflicting = reconstruct_call_number(["500", "619", "ㅅ21"])
        self.assertEqual(matching.structure_score - conflicting.structure_score, 8)
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run: `python -m unittest tests.test_call_number_reconstruction -v`

Expected: ERROR with `ModuleNotFoundError` for `app.analysis.ocr.call_number_reconstruction`.

- [ ] **Step 3: Add result types, scoring constants, and the DP grammar**

Create these public and internal data contracts:

```python
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Sequence


MAJOR_MATCH_SCORE = 6
MAJOR_MISSING_SCORE = -6
DETAIL_MATCH_SCORE = 6
DETAIL_MISSING_SCORE = -8
DETAIL_PREFIX_MATCH_SCORE = 3
DETAIL_PREFIX_CONFLICT_SCORE = -5
FIRST_SYMBOL_SCORE = 6
MISSING_SYMBOL_SCORE = -6
ADDITIONAL_SYMBOL_SCORE = 2
SUFFIX_SCORE = 2
DISCARDED_CHARACTER_SCORE = -1
NUMERIC_SYMBOL_RUN_SCORE = -6


@dataclass(frozen=True)
class ReconstructionResult:
    text: str
    major: str
    detail: str
    symbols: str
    suffix: str
    structure_score: int
    retained_positions: tuple[int, ...]
    discarded_character_count: int
    completed_section_count: int


@dataclass(frozen=True)
class _ParsePath:
    major: str = ""
    detail: str = ""
    symbols: str = ""
    suffix: str = ""
    score: int = 0
    retained_positions: tuple[int, ...] = ()
    discarded_character_count: int = 0
    completed_section_count: int = 0


class _Phase(Enum):
    MAJOR = auto()
    DETAIL = auto()
    SYMBOLS = auto()
    SUFFIX = auto()
    DONE = auto()
```

Implement `reconstruct_call_number()` by flattening non-whitespace characters in source order, retaining each compact-stream position. Use a memoized search whose state includes compact-stream index, phase, pattern progress, the recovered major digit, current symbol digit count, and whether the first symbol group is complete. At every state:

1. A delete transition advances the input only and applies `DISCARDED_CHARACTER_SCORE`.
2. A retain transition advances input and grammar progress and prepends the retained character and position to the returned immutable path.
3. A missing-section transition advances grammar phase without input and applies the relevant missing penalty.
4. Completing major, detail, first symbol, later symbol, or suffix applies the specified score exactly once.
5. Deleting the first contiguous digit run encountered where a new symbol must begin applies `NUMERIC_SYMBOL_RUN_SCORE` once in addition to per-character deletion penalties.

The detail state must branch after its third digit: it may finish as an integer, or consume a literal period followed by one or more digits. The DP decides the fraction length by the best downstream path. The suffix branch must compete with treating `v` or `c` as another symbol group, and receives `SUFFIX_SCORE` only when it consumes a literal period and at least one digit at the reconstructed end.

Format the result without numeric conversion:

```python
def _format_sections(path: _ParsePath) -> str:
    return " ".join(
        section for section in (path.major, path.detail, path.symbols, path.suffix)
        if section
    )
```

If `completed_section_count == 0`, use `" ".join(str(text).split() for text in token_texts if str(text).strip())` as the display fallback while retaining missing-section penalties.

- [ ] **Step 4: Run canonical reconstruction tests and verify they pass**

Run: `python -m unittest tests.test_call_number_reconstruction -v`

Expected: PASS for the four tests above.

- [ ] **Step 5: Add failing tests for deletion, symbol repetition, suffix normalization, and prohibited invention**

```python
    def test_noise_is_deleted_between_and_inside_tokens(self):
        between = reconstruct_call_number(["500", "ㄹ", "519.5", "ㅅ21", "c.2"])
        inside = reconstruct_call_number(["500", "ㄹ519.5", "ㅅ21", "c.2"])
        self.assertEqual(between.text, "500 519.5 ㅅ21 c.2")
        self.assertEqual(inside.text, between.text)
        self.assertEqual(inside.discarded_character_count, 1)

    def test_repeated_symbol_groups_share_one_output_section(self):
        result = reconstruct_call_number(["500", "519.5", "ㅅ", "21", "한"])
        self.assertEqual(result.symbols, "ㅅ21한")
        self.assertEqual(result.text, "500 519.5 ㅅ21한")

    def test_split_and_uppercase_suffixes_are_normalized(self):
        variants = (["V.2"], ["v", ".2"], ["v.", "2"])
        for suffix_tokens in variants:
            with self.subTest(suffix_tokens=suffix_tokens):
                result = reconstruct_call_number(["500", "519.5", "ㅅ21", *suffix_tokens])
                self.assertEqual(result.suffix, "v.2")

    def test_missing_suffix_period_is_not_invented(self):
        result = reconstruct_call_number(["500", "519.5", "ㅅ21", "v2"])
        self.assertNotEqual(result.suffix, "v.2")

    def test_numeric_leading_symbol_run_receives_large_penalty(self):
        valid = reconstruct_call_number(["500", "519.5", "ㅅ21", "한"])
        numeric_leading = reconstruct_call_number(["500", "519.5", "ㅅ21", "123", "한"])
        self.assertGreaterEqual(valid.structure_score - numeric_leading.structure_score, 6)

    def test_unstructured_input_uses_raw_fallback(self):
        result = reconstruct_call_number(["@@", "--", "??"])
        self.assertEqual(result.text, "@@ -- ??")
        self.assertEqual(result.completed_section_count, 0)
```

- [ ] **Step 6: Run the new tests and verify at least the noise/suffix cases fail before completing the DP**

Run: `python -m unittest tests.test_call_number_reconstruction -v`

Expected: FAIL in the newly added noise, suffix, symbol, or penalty behaviors until all DP branches and tie-break rules are implemented.

- [ ] **Step 7: Complete DP transitions and deterministic path comparison**

Use one comparison function everywhere:

```python
def _path_rank(path: _ParsePath) -> tuple[object, ...]:
    return (
        path.score,
        path.completed_section_count,
        -path.discarded_character_count,
        tuple(-position for position in path.retained_positions),
    )


def _best(paths: Sequence[_ParsePath]) -> _ParsePath:
    return max(paths, key=_path_rank)
```

When a suffix completes, normalize only its marker with `marker.lower()`; retain the input period and digits verbatim. When Unicode symbol letters are tested, use `character.isalpha()` so Korean jamo, completed Hangul, and Latin letters are accepted without a hard-coded alphabet.

- [ ] **Step 8: Run reconstruction tests and the existing OCR-processing tests**

Run: `python -m unittest tests.test_call_number_reconstruction tests.test_ocr_processing -v`

Expected: PASS with no existing token-order regression.

- [ ] **Step 9: Commit the pure reconstructor**

```bash
git add app/analysis/ocr/call_number_reconstruction.py tests/test_call_number_reconstruction.py docs/superpowers/plans/2026-08-18-ocr-call-number-reconstruction-scoring.md
git commit -m "feat: reconstruct call numbers from noisy OCR"
```

---

### Task 2: Confidence Statistics and Ordered Variant Selection

**Files:**
- Modify: `app/analysis/ocr/ocr_processing.py:71-116`
- Modify: `tests/test_ocr_processing.py:1-140`

**Interfaces:**
- Consumes: `reconstruct_call_number(token_texts: Sequence[object]) -> ReconstructionResult` from Task 1.
- Produces: `_confidence_statistics(tokens: Sequence[dict[str, Any]]) -> tuple[float, float, float]` in lower-quartile, median, average order.
- Produces: `_select_best_variant(eligible_variants: Sequence[dict[str, Any]]) -> dict[str, Any]` using the exact five-stage comparison.
- Adds variant fields `reconstructedText`, `structureScore`, `discardedCharacterCount`, `lowerQuartileConfidence`, and `medianConfidence` while preserving existing fields.

- [ ] **Step 1: Replace the obsolete average-only test with failing structure-priority integration tests**

Add a helper that positions explicit text/confidence values in either contact-sheet variant:

```python
    def _variant_annotations(self, x, texts, confidences):
        return [
            {
                "text": text,
                "box": [x, 10 + index * 20, 30, 10],
                "confidence": confidence,
            }
            for index, (text, confidence) in enumerate(zip(texts, confidences))
        ]
```

Replace `test_two_eligible_orientations_still_select_higher_average_confidence` and add:

```python
    def test_better_structure_beats_higher_average_confidence(self):
        annotations = self._variant_annotations(
            10, ["500", "519.5", "ㅅ21"], [0.60, 0.60, 0.60]
        ) + self._variant_annotations(
            120, ["noise", "words", "only"], [0.99, 0.99, 0.99]
        )

        row = map_ocr_result_to_rows({"annotations": annotations}, self._two_variant_sheet())[0]

        self.assertEqual(row["selectedOrientation"], "upright")
        self.assertEqual(row["text"], "500 519.5 ㅅ21")
        self.assertEqual(row["variantResults"][0]["text"], "500 519.5 ㅅ21")

    def test_discarded_noise_still_affects_confidence_statistics(self):
        annotations = self._variant_annotations(
            10, ["500", "ㄹ", "519.5", "ㅅ21"], [0.90, 0.10, 0.90, 0.90]
        )
        row = map_ocr_result_to_rows({"annotations": annotations}, self._two_variant_sheet())[0]
        variant = row["variantResults"][0]

        self.assertEqual(variant["reconstructedText"], "500 519.5 ㅅ21")
        self.assertAlmostEqual(variant["lowerQuartileConfidence"], 0.10)
        self.assertAlmostEqual(variant["medianConfidence"], 0.90)
        self.assertAlmostEqual(variant["averageConfidence"], 0.70)
```

- [ ] **Step 2: Run the focused tests and verify missing diagnostic fields or wrong selection**

Run: `python -m unittest tests.test_ocr_processing.OCRProcessingTests.test_better_structure_beats_higher_average_confidence tests.test_ocr_processing.OCRProcessingTests.test_discarded_noise_still_affects_confidence_statistics -v`

Expected: FAIL because selection still uses only `averageConfidence` and reconstruction fields do not exist.

- [ ] **Step 3: Calculate reconstruction and confidence statistics for every variant**

Add imports and helpers:

```python
from math import ceil
from statistics import median

from .call_number_reconstruction import reconstruct_call_number


STRUCTURE_NEAR_TIE_MARGIN = 2


def _confidence_statistics(tokens: Sequence[dict[str, Any]]) -> tuple[float, float, float]:
    values = sorted(float(token.get("confidence", 0.0) or 0.0) for token in tokens)
    if not values:
        return 0.0, 0.0, 0.0
    lower_count = max(1, ceil(len(values) * 0.25))
    lower_quartile = sum(values[:lower_count]) / lower_count
    return lower_quartile, float(median(values)), sum(values) / len(values)
```

For each sorted variant, call the reconstructor with `[token["text"] for token in tokens]`, calculate all statistics from the complete `tokens` list, and update without replacing raw values:

```python
variant.update(
    reconstructedText=reconstruction.text,
    structureScore=reconstruction.structure_score,
    discardedCharacterCount=reconstruction.discarded_character_count,
    lowerQuartileConfidence=lower_quartile,
    medianConfidence=median_confidence,
    averageConfidence=average_confidence,
    eligible=len(tokens) >= MIN_OCR_TOKENS,
)
```

- [ ] **Step 4: Implement the explicit near-tie comparator and selected-row assignment**

```python
def _select_best_variant(
    eligible_variants: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected = eligible_variants[0]
    for candidate in eligible_variants[1:]:
        score_difference = candidate["structureScore"] - selected["structureScore"]
        if abs(score_difference) > STRUCTURE_NEAR_TIE_MARGIN:
            if score_difference > 0:
                selected = candidate
            continue
        for key in (
            "lowerQuartileConfidence",
            "medianConfidence",
            "averageConfidence",
        ):
            if candidate[key] > selected[key]:
                selected = candidate
                break
            if candidate[key] < selected[key]:
                break
    return selected
```

Set `row["text"] = selected["reconstructedText"]`; preserve `row["tokens"]`, `row["selectedOrientation"]`, and `row["selectedAverageConfidence"]` exactly as before.

- [ ] **Step 5: Run the focused integration tests and verify they pass**

Run: `python -m unittest tests.test_ocr_processing.OCRProcessingTests.test_better_structure_beats_higher_average_confidence tests.test_ocr_processing.OCRProcessingTests.test_discarded_noise_still_affects_confidence_statistics -v`

Expected: PASS.

- [ ] **Step 6: Add failing tests for every near-tie confidence stage and stable final tie**

Use `_select_best_variant` directly with literal diagnostic dictionaries so each stage is isolated:

```python
    def test_structure_difference_above_margin_wins(self):
        lower_structure = self._ranked_variant(10, 0.99, 0.99, 0.99)
        higher_structure = self._ranked_variant(13, 0.10, 0.10, 0.10)
        self.assertIs(_select_best_variant([lower_structure, higher_structure]), higher_structure)

    def test_near_tie_uses_lower_quartile_before_median(self):
        better_median = self._ranked_variant(10, 0.40, 0.95, 0.95)
        better_low = self._ranked_variant(12, 0.50, 0.50, 0.50)
        self.assertIs(_select_best_variant([better_median, better_low]), better_low)

    def test_lower_quartile_tie_uses_median_before_average(self):
        better_average = self._ranked_variant(10, 0.40, 0.50, 0.99)
        better_median = self._ranked_variant(10, 0.40, 0.60, 0.60)
        self.assertIs(_select_best_variant([better_average, better_median]), better_median)

    def test_median_tie_uses_average(self):
        first = self._ranked_variant(10, 0.40, 0.60, 0.70)
        second = self._ranked_variant(10, 0.40, 0.60, 0.80)
        self.assertIs(_select_best_variant([first, second]), second)

    def test_complete_tie_preserves_first_variant(self):
        first = self._ranked_variant(10, 0.40, 0.60, 0.80)
        second = self._ranked_variant(10, 0.40, 0.60, 0.80)
        self.assertIs(_select_best_variant([first, second]), first)
```

Define `_ranked_variant` in the test class with the exact keys consumed by `_select_best_variant`:

```python
    def _ranked_variant(self, structure, lower_quartile, median_value, average):
        return {
            "structureScore": structure,
            "lowerQuartileConfidence": lower_quartile,
            "medianConfidence": median_value,
            "averageConfidence": average,
        }
```

- [ ] **Step 7: Run all OCR-processing tests and verify the hierarchy**

Run: `python -m unittest tests.test_ocr_processing -v`

Expected: PASS, including existing 3-token eligibility and reading-order tests.

- [ ] **Step 8: Commit OCR integration and selection**

```bash
git add app/analysis/ocr/ocr_processing.py tests/test_ocr_processing.py
git commit -m "feat: rank OCR variants by call number validity"
```

---

### Task 3: Observable Reconstruction Diagnostics

**Files:**
- Modify: `app/analysis/result_rendering.py:151-169`
- Modify: `tests/test_result_rendering.py:10-125`

**Interfaces:**
- Consumes: existing variant results plus `reconstructedText`, `structureScore`, `discardedCharacterCount`, `lowerQuartileConfidence`, and `medianConfidence` from Task 2.
- Produces: the existing `build_ocr_debug_details(rows: Sequence[dict[str, Any]]) -> list[str]` with raw/reconstructed text and all comparison metrics included.

- [ ] **Step 1: Write the failing diagnostic rendering test**

```python
    def test_debug_details_include_reconstruction_selection_metrics(self):
        row = self._row([0.95, 0.80, 0.70], [0.63, 0.62, 0.61])
        variant = row["variantResults"][0]
        variant.update(
            text="500 ㄹ 519.5 ㅅ21",
            reconstructedText="500 519.5 ㅅ21",
            structureScore=18,
            discardedCharacterCount=1,
            lowerQuartileConfidence=0.70,
            medianConfidence=0.80,
            averageConfidence=0.82,
            eligible=True,
        )

        detail = build_ocr_debug_details([row])[0]

        self.assertIn('raw "500 ㄹ 519.5 ㅅ21"', detail)
        self.assertIn('parsed "500 519.5 ㅅ21"', detail)
        self.assertIn("structure 18", detail)
        self.assertIn("discarded 1", detail)
        self.assertIn("low25 0.70", detail)
        self.assertIn("median 0.80", detail)
        self.assertIn("avg 0.82", detail)
```

- [ ] **Step 2: Run the focused test and verify diagnostics are absent**

Run: `python -m unittest tests.test_result_rendering.OCRDebugRenderingTests.test_debug_details_include_reconstruction_selection_metrics -v`

Expected: FAIL because existing debug details show only average confidence and token boxes.

- [ ] **Step 3: Extend debug detail formatting with safe defaults**

Build the candidate prefix as:

```python
raw_text = str(variant.get("text", ""))
reconstructed = str(variant.get("reconstructedText", raw_text))
score = int(variant.get("structureScore", 0) or 0)
discarded = int(variant.get("discardedCharacterCount", 0) or 0)
low25 = float(variant.get("lowerQuartileConfidence", 0.0) or 0.0)
median_confidence = float(variant.get("medianConfidence", 0.0) or 0.0)
average = float(variant.get("averageConfidence", 0.0) or 0.0)

candidate_summary = (
    f'raw "{raw_text}" parsed "{reconstructed}" structure {score} '
    f'discarded {discarded} low25 {low25:.2f} '
    f'median {median_confidence:.2f} avg {average:.2f}'
)
```

Keep the existing eligibility, selected marker, and per-token text/confidence/box summary after this prefix.

- [ ] **Step 4: Run result-rendering and OCR-processing tests**

Run: `python -m unittest tests.test_result_rendering tests.test_ocr_processing -v`

Expected: PASS with selected/rejected overlay behavior unchanged.

- [ ] **Step 5: Commit debug diagnostics**

```bash
git add app/analysis/result_rendering.py tests/test_result_rendering.py
git commit -m "feat: show OCR reconstruction scoring details"
```

---

### Task 4: Full Regression Verification

**Files:**
- Verify: `app/analysis/ocr/call_number_reconstruction.py`
- Verify: `app/analysis/ocr/ocr_processing.py`
- Verify: `app/analysis/result_rendering.py`
- Verify: `tests/test_call_number_reconstruction.py`
- Verify: `tests/test_ocr_processing.py`
- Verify: `tests/test_result_rendering.py`

**Interfaces:**
- Consumes: all deliverables from Tasks 1-3.
- Produces: a clean, fully verified working tree on the current `google-vision-OCR` branch.

- [ ] **Step 1: Run the complete automated test suite**

Run: `python -m unittest discover -s tests -v`

Expected: All tests PASS with zero failures and zero errors.

- [ ] **Step 2: Run JavaScript UI regression test**

Run: `node --test tests/test_web_ui.js`

Expected: PASS.

- [ ] **Step 3: Check formatting and scope**

Run: `git diff --check`

Expected: exit code 0 with no whitespace-error output.

Run: `git status --short`

Expected: no task-related uncommitted files; preserve unrelated user-owned files unchanged.
