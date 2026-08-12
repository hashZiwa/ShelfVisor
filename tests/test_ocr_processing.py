import unittest

from PIL import Image

from app.analysis.analysis_models import DetectedRegion
from app.analysis.ocr.ocr_processing import build_ocr_contact_sheet, map_ocr_result_to_rows


class OCRProcessingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
