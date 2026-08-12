import unittest

from PIL import Image

from app.analysis.analysis_models import DetectedRegion, OCRContactSheet
from app.analysis.ocr.ocr_processing import build_ocr_contact_sheet, map_ocr_result_to_rows


class OCRProcessingTests(unittest.TestCase):
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
        self.assertEqual(rows[0]["text"], "811.1 A 2")
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
        self.assertEqual(rows[0]["text"], "top middle bottom")

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
        self.assertEqual(rows[0]["text"], "left middle right")

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
        self.assertEqual(rows[0]["text"], "top middle bottom")

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


if __name__ == "__main__":
    unittest.main()
