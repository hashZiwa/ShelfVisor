# OCR Call-Number Reconstruction and Candidate Scoring

## Goal

Choose the better OCR image variant by reconstructing the most plausible call number from its OCR characters, scoring that reconstructed structure, and using robust confidence statistics only when the structural scores are effectively tied.

The selected call number must be displayed in a canonical, sectioned form such as:

```text
500 519.5 ㅅ21 c.2
```

## Existing Behavior

Each detected book-label image has two OCR variants: the original crop and a 90-degree counter-clockwise rotation. The current pipeline:

1. Rejects variants with fewer than three OCR tokens.
2. Sorts each variant's tokens into reading order.
3. Selects the eligible variant with the highest average token confidence.

This treats OCR token boundaries as meaningful and cannot distinguish a plausible call number from a structurally invalid high-confidence result.

## Required Selection Order

Compare the two variants in this order:

1. Reject variants with fewer than three original OCR tokens.
2. Compare reconstructed call-number structure scores.
3. If the structure-score difference is at most 2 points, compare the mean confidence of the lowest-confidence 25% of original tokens.
4. If tied, compare median confidence.
5. If tied, compare average confidence.
6. If still tied, preserve the current stable behavior and select the first variant, which is the original crop.

For the lower-quartile mean, sort token confidence values ascending and average the first `ceil(token_count * 0.25)` values, always including at least one token.

All confidence statistics use every original OCR token, including tokens or characters discarded during reconstruction. Reconstruction must not improve a candidate's confidence statistics by hiding noisy OCR results.

## Reconstruction Principles

### Allowed operations

The reconstructor may:

- Preserve characters in their original order.
- Delete an unsuitable complete token.
- Delete unsuitable characters from inside a token.
- Join characters that OCR split across token boundaries.
- Normalize an accepted `V` or `C` suffix marker to lowercase.
- Insert presentation spaces between reconstructed semantic sections.

The reconstructor must not:

- Reorder tokens or characters.
- Substitute one recognized character for another.
- Invent a missing punctuation mark, digit, or letter.

For example, `v` plus `.2` may become `v.2` because every output character exists in the input. `v2` must not become `v.2` because that would invent a period.

### Character stream

After the existing reading-order sort, flatten token text into an ordered character stream. Each character retains its source token index so diagnostics can identify what was retained or discarded. Whitespace is ignored for structural matching. OCR token boundaries do not define call-number sections.

The following tokenizations must therefore be structurally equivalent:

```text
["500", "519.5", "ㅅ", "21", "c.2"]
["5", "00", "519", ".5", "ㅅ21", "c", ".2"]
```

### Grammar

The canonical call number contains these ordered sections:

1. **Major classification:** one digit followed by `00`, including `000`.
2. **Detailed classification:** exactly three digits with an optional decimal part containing one or more digits.
3. **Book/author symbols:** one or more repetitions of letters followed by zero to three digits.
4. **Optional suffix:** `v.` or `c.`, case-insensitive, followed by one or more digits.

The detailed classification's first digit should equal the major classification's first digit:

```text
500 -> 519 or 519.5
000 -> 005 or 005.12
```

All numeric sections remain strings. The implementation must never convert them to integers or floating-point values, so leading zeroes such as `005` are preserved.

The symbols section is rendered without spaces between repetitions. For example, tokens `ㅅ`, `21`, and `한` reconstruct to `ㅅ21한`. Presentation spaces appear only between the major, detail, symbols, and suffix sections.

### Noise handling

Use a weighted dynamic-programming parser. It advances through the ordered character stream and grammar states while considering retain or delete actions. The highest-scoring complete or partial path becomes the reconstructed candidate.

This allows:

```text
500ㄹ519.5ㅅ21c.2 -> 500 519.5 ㅅ21 c.2
ㄹ519.5           -> 519.5
```

In both cases `ㄹ` may be deleted while all retained characters keep their original order.

## Structure Score

Use the following initial weights:

| Structural feature | Score |
|---|---:|
| Major classification `N00` present | +6 |
| Major classification missing | -6 |
| Three-digit detailed classification, with or without decimal part | +6 |
| Detailed classification missing | -8 |
| Major/detail first digits match | +3 |
| Major/detail first digits conflict | -5 |
| First valid book/author symbol group | +6 |
| No valid book/author symbol group | -6 |
| Each additional symbol group | +2 |
| Valid trailing `v.number` or `c.number` suffix | +2 |
| Discarded general character | -1 per character |
| Discarded numeric-leading run in the symbol region | additional -6 per run |

When the major classification is missing, the detailed classification can still receive its format score, but no digit-consistency adjustment applies.

The suffix bonus is intentionally small. A suffix-only difference and a one-character noise difference fall within the 2-point structural near-tie margin and therefore advance to confidence comparison. Missing core sections or a numeric-leading symbol run should normally produce a decisive structural difference.

If multiple reconstruction paths have the same score, prefer them in this order:

1. More completed core sections.
2. Fewer discarded characters.
3. Earlier retained characters in the source stream.

If no core section can be reconstructed, return the whitespace-normalized raw token text as the display fallback and assign the missing-section penalties. Do not raise an exception.

## Module Boundaries

### `app/analysis/ocr/call_number_reconstruction.py`

Add a pure reconstruction module that knows nothing about images or OCR APIs. It consumes ordered token texts and returns a reconstruction result containing:

- `text`: canonical sectioned output.
- `major`, `detail`, `symbols`, and `suffix` strings when present.
- `structure_score`.
- Retained source character positions.
- `discarded_character_count`.
- Completed-section count for deterministic tie-breaking.

The exact representation may use a frozen dataclass, but callers must not need to understand the DP's internal state.

### `app/analysis/ocr/ocr_processing.py`

Keep token mapping and reading-order sorting here. After sorting each variant:

1. Preserve its raw `text` and `tokens`.
2. Reconstruct its canonical call number.
3. Calculate lower-quartile mean, median, and average from all original token confidences.
4. Store reconstruction and comparison diagnostics on the variant result.
5. Select among eligible variants with the required comparison order.
6. Set the selected row's `text` to `reconstructedText` while preserving the selected raw tokens.

Candidate comparison should live in a named helper rather than an opaque tuple because structure scores use a 2-point near-tie rule before confidence statistics apply.

### `app/analysis/result_rendering.py`

Extend existing OCR debug details to show, for each candidate:

- Raw OCR text.
- Reconstructed text.
- Structure score.
- Discarded-character count.
- Lower-quartile mean, median, and average confidence.
- Eligibility and selected state.

This makes the selection reason observable without changing the normal result UI beyond showing the reconstructed selected call number.

## Data Compatibility

Preserve the existing fields and meanings used elsewhere:

- `variantResults[*].text` remains the raw joined OCR text.
- `variantResults[*].tokens` remains the original sorted token list.
- `variantResults[*].averageConfidence` remains available.
- `row.tokens` remains the selected variant's original sorted tokens.
- `row.selectedAverageConfidence` remains the selected variant's average confidence.

Add these diagnostic fields to each variant:

- `reconstructedText`
- `structureScore`
- `discardedCharacterCount`
- `lowerQuartileConfidence`
- `medianConfidence`

Set `row.text` to the selected variant's `reconstructedText`.

## Error and Edge-Case Handling

- Empty OCR results and variants with fewer than three tokens remain ineligible.
- A variant with three or more tokens remains eligible even if reconstruction is partial; its missing-section penalties lower its structural score.
- Non-string token text is normalized safely to a string before reconstruction.
- Missing, null, or invalid confidence values continue to behave as `0.0`.
- Suffix recognition is anchored to the reconstructed end. A middle `v.2` or `c.2` is not awarded a suffix bonus.
- Uppercase `V` and `C` are accepted only as suffix markers and rendered lowercase.
- Leading zeroes are preserved in all numeric output.

## Testing Strategy

### Reconstruction unit tests

Create `tests/test_call_number_reconstruction.py` with focused cases for:

- Equivalent reconstruction across different OCR token splits.
- Noise deletion between tokens and inside a token.
- Preservation of source character order.
- Prohibition of substitutions and invented punctuation.
- Integer and decimal detailed classifications.
- Major/detail digit agreement and disagreement.
- `000 -> 005` reconstruction with leading-zero preservation.
- Repeated symbol groups rendered as one section.
- Combined and split `v.2` and `c.2` suffix variants, including uppercase input.
- Numeric-leading symbol penalties.
- Partial reconstruction and raw fallback behavior.

### OCR integration tests

Extend `tests/test_ocr_processing.py` to verify:

- Fewer than three original tokens still makes a variant ineligible.
- A decisively better structure beats higher confidence.
- A structure-score difference of 2 or less advances to lower-quartile comparison.
- Lower-quartile ties advance to median confidence.
- Median ties advance to average confidence.
- Complete ties retain the original orientation.
- Confidence statistics include discarded tokens.
- Raw candidate text and tokens remain unchanged.
- The selected row text uses the reconstructed canonical string.

### Debug rendering tests

Extend `tests/test_result_rendering.py` to verify that reconstruction and comparison diagnostics appear for both orientations without changing selected/rejected visual semantics.

Run the focused reconstruction, OCR processing, and result-rendering tests first, then run the complete test suite.

## Out of Scope

- Reordering OCR characters or tokens.
- Correcting one recognized character into another.
- Inventing missing punctuation or digits.
- Learning score weights automatically from a dataset.
- Changing the OCR provider or OCR request format.
- Changing token reading-order logic.
