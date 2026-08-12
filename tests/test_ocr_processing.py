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

    def test_contact_sheet_contains_both_orientations(self):
        region = DetectedRegion((10, 10, 20, 60), [[10, 10], [30, 10], [30, 70], [10, 70]])
        sheet = build_ocr_contact_sheet(Image.new("RGB", (100, 100), "white"), [region])
        self.assertEqual([item["orientation"] for item in sheet.rows[0]["variants"]], ["upright", "rotated_ccw_90"])

    def test_annotation_is_mapped_to_its_variant(self):
        region = DetectedRegion((10, 10, 20, 60), [[10, 10], [30, 10], [30, 70], [10, 70]])
        sheet = build_ocr_contact_sheet(Image.new("RGB", (100, 100), "white"), [region])
        x, y, width, height = sheet.rows[0]["variants"][0]["sheetBox"]
        rows = map_ocr_result_to_rows(
            {"annotations": [{"text": "811.1", "box": [x + 1, y + 1, width - 2, height - 2], "confidence": 0.9}]},
            sheet,
        )
        self.assertEqual(rows[0]["text"], "811.1")
        self.assertEqual(rows[0]["selectedOrientation"], "upright")

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


if __name__ == "__main__":
    unittest.main()
