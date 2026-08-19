import unittest

from PIL import Image

from app.analysis.analysis_models import DetectedRegion, OCRContactSheet
from app.analysis.ocr.ocr_processing import (
    _select_best_variant,
    build_ocr_contact_sheet,
    map_ocr_result_to_rows,
)


class OCRProcessingTests(unittest.TestCase):
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

    def _variant_annotations(self, x, texts, confidences):
        return [
            {
                "text": text,
                "box": [x, 10 + index * 20, 30, 10],
                "confidence": confidence,
            }
            for index, (text, confidence) in enumerate(zip(texts, confidences))
        ]

    def _ranked_variant(
        self, structure, lower_quartile, median_value, average, has_confidence=True
    ):
        return {
            "structureScore": structure,
            "lowerQuartileConfidence": lower_quartile,
            "medianConfidence": median_value,
            "averageConfidence": average,
            "hasConfidence": has_confidence,
        }

    def test_contact_sheet_contains_both_orientations(self):
        region = DetectedRegion((10, 10, 20, 60), [[10, 10], [30, 10], [30, 70], [10, 70]])
        sheet = build_ocr_contact_sheet(Image.new("RGB", (100, 100), "white"), [region])
        self.assertEqual([item["orientation"] for item in sheet.rows[0]["variants"]], ["upright", "rotated_ccw_90"])

    def test_annotation_is_mapped_to_its_variant(self):
        region = DetectedRegion((10, 10, 20, 60), [[10, 10], [30, 10], [30, 70], [10, 70]])
        sheet = build_ocr_contact_sheet(Image.new("RGB", (100, 100), "white"), [region])
        x, y, width, height = sheet.rows[0]["variants"][0]["sheetBox"]
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "811.1", "box": [x + 1, y + 10, width - 2, 20], "confidence": 0.9},
                {"text": "A", "box": [x + 1, y + 50, width - 2, 20], "confidence": 0.9},
                {"text": "2", "box": [x + 1, y + 90, width - 2, 20], "confidence": 0.9},
            ]},
            sheet,
        )
        self.assertEqual(rows[0]["text"], "811.1 A2")
        self.assertEqual(rows[0]["variantResults"][0]["text"], "811.1 A 2")
        self.assertEqual(rows[0]["selectedOrientation"], "upright")

    def test_wide_tokens_are_sorted_top_to_bottom(self):
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "bottom", "box": [10, 100, 40, 10], "confidence": 0.9},
                {"text": "top", "box": [80, 20, 40, 10], "confidence": 0.9},
                {"text": "middle", "box": [50, 60, 40, 10], "confidence": 0.9},
            ]},
            self._single_variant_sheet(),
        )

        self.assertEqual([token["text"] for token in rows[0]["tokens"]], ["top", "middle", "bottom"])
        self.assertEqual(rows[0]["variantResults"][0]["text"], "top middle bottom")

    def test_tall_tokens_are_sorted_left_to_right(self):
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "right", "box": [100, 10, 10, 40], "confidence": 0.9},
                {"text": "left", "box": [20, 80, 10, 40], "confidence": 0.9},
                {"text": "middle", "box": [60, 45, 10, 40], "confidence": 0.9},
            ]},
            self._single_variant_sheet(),
        )

        self.assertEqual([token["text"] for token in rows[0]["tokens"]], ["left", "middle", "right"])
        self.assertEqual(rows[0]["variantResults"][0]["text"], "left middle right")

    def test_equal_average_width_and_height_sort_top_to_bottom(self):
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "bottom", "box": [10, 100, 20, 20], "confidence": 0.9},
                {"text": "top", "box": [80, 20, 20, 20], "confidence": 0.9},
                {"text": "middle", "box": [50, 60, 20, 20], "confidence": 0.9},
            ]},
            self._single_variant_sheet(),
        )

        self.assertEqual([token["text"] for token in rows[0]["tokens"]], ["top", "middle", "bottom"])
        self.assertEqual(rows[0]["variantResults"][0]["text"], "top middle bottom")

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

    def test_text_detection_can_accept_one_annotation_when_the_caller_lowers_the_token_gate(self):
        row = map_ocr_result_to_rows(
            {
                "annotations": [
                    {
                        "text": "500 519.5 ㅅ21",
                        "box": [10, 10, 60, 20],
                        "confidence": None,
                    }
                ]
            },
            self._two_variant_sheet(),
            minimum_tokens=1,
        )[0]

        self.assertTrue(row["eligible"])
        self.assertEqual(row["text"], "500 519.5 ㅅ21")

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

    def test_structure_difference_above_margin_wins(self):
        lower_structure = self._ranked_variant(10, 0.99, 0.99, 0.99)
        higher_structure = self._ranked_variant(13, 0.10, 0.10, 0.10)

        self.assertIs(
            _select_best_variant([lower_structure, higher_structure]),
            higher_structure,
        )

    def test_missing_confidence_uses_the_higher_structure_score_even_within_margin(self):
        upright = self._ranked_variant(10, 0.0, 0.0, 0.0, has_confidence=False)
        rotated = self._ranked_variant(12, 0.0, 0.0, 0.0, has_confidence=False)

        self.assertIs(_select_best_variant([upright, rotated]), rotated)

    def test_near_tie_uses_lower_quartile_before_median(self):
        better_median = self._ranked_variant(10, 0.40, 0.95, 0.95)
        better_low = self._ranked_variant(12, 0.50, 0.50, 0.50)

        self.assertIs(
            _select_best_variant([better_median, better_low]),
            better_low,
        )

    def test_lower_quartile_tie_uses_median_before_average(self):
        better_average = self._ranked_variant(10, 0.40, 0.50, 0.99)
        better_median = self._ranked_variant(10, 0.40, 0.60, 0.60)

        self.assertIs(
            _select_best_variant([better_average, better_median]),
            better_median,
        )

    def test_median_tie_uses_average(self):
        first = self._ranked_variant(10, 0.40, 0.60, 0.70)
        second = self._ranked_variant(10, 0.40, 0.60, 0.80)

        self.assertIs(_select_best_variant([first, second]), second)

    def test_complete_tie_preserves_first_variant(self):
        first = self._ranked_variant(10, 0.40, 0.60, 0.80)
        second = self._ranked_variant(10, 0.40, 0.60, 0.80)

        self.assertIs(_select_best_variant([first, second]), first)

    def test_malformed_text_and_confidence_values_are_normalized_safely(self):
        annotations = self._variant_annotations(
            10,
            [500, None, "ㅅ21"],
            ["bad", float("nan"), None],
        )

        row = map_ocr_result_to_rows({"annotations": annotations}, self._two_variant_sheet())[0]
        variant = row["variantResults"][0]

        self.assertEqual(variant["text"], "500 None ㅅ21")
        self.assertIsNone(variant["tokens"][1]["text"])
        self.assertEqual(variant["lowerQuartileConfidence"], 0.0)
        self.assertEqual(variant["medianConfidence"], 0.0)
        self.assertEqual(variant["averageConfidence"], 0.0)

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

    def test_gradually_shifted_boxes_share_a_line_through_connected_overlap(self):
        rows = map_ocr_result_to_rows(
            {"annotations": [
                {"text": "right", "box": [110, 10, 40, 20], "confidence": 0.9},
                {"text": "middle", "box": [60, 20, 40, 20], "confidence": 0.9},
                {"text": "left", "box": [10, 30, 40, 20], "confidence": 0.9},
            ]},
            self._single_variant_sheet(),
        )

        self.assertEqual(
            [token["text"] for token in rows[0]["tokens"]],
            ["left", "middle", "right"],
        )


if __name__ == "__main__":
    unittest.main()
