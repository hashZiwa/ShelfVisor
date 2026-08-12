import io
import unittest
from unittest.mock import patch

from PIL import Image

from app.analysis.analysis_models import OCRContactSheet
from app.analysis.analysis_pipeline import analyze_shelf_photo


class AnalysisPipelineTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        Image.new("RGB", (100, 100), "white").save(buffer, format="JPEG")
        self.image_bytes = buffer.getvalue()
        self.yolo_result = {
            "predictions": [{
                "x": 50, "y": 50, "width": 20, "height": 40,
                "confidence": 0.9, "class": "book",
            }],
            "model_id": "test-model.pt",
        }
        self.ocr_result = {
            "fullText": "811.3 A",
            "annotations": [
                {"text": "811.3", "box": [100, 40, 10, 10], "confidence": 0.95},
                {"text": "A", "box": [100, 55, 10, 10], "confidence": 0.95},
                {"text": "", "box": [100, 70, 10, 10], "confidence": 0.95},
            ],
        }

    @patch("app.analysis.analysis_pipeline.run_ocr_inference")
    @patch("app.analysis.analysis_pipeline.run_yolo_inference")
    def test_pipeline_preserves_response_shape(self, run_yolo, run_ocr):
        run_yolo.return_value = self.yolo_result
        run_ocr.return_value = self.ocr_result

        result = analyze_shelf_photo(self.image_bytes)

        self.assertEqual(result["summary"]["bookCount"], 1)
        self.assertEqual(result["spines"][0]["call_number"], "811.3 A")
        self.assertTrue(result["annotatedImage"].startswith("data:image/jpeg;base64,"))

    @patch("app.analysis.analysis_pipeline.run_ocr_inference")
    @patch("app.analysis.analysis_pipeline.run_yolo_inference")
    def test_debug_payload_contains_ordered_stage_labels(self, run_yolo, run_ocr):
        run_yolo.return_value = self.yolo_result
        run_ocr.return_value = self.ocr_result

        result = analyze_shelf_photo(self.image_bytes, include_debug=True)

        self.assertEqual(
            [stage["label"] for stage in result["debug"]["stages"]],
            ["Original", "YOLO predictions", "Size filter", "OCR contact sheet", "OCR bounding boxes", "Final result"],
        )

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
        build_sheet.return_value = OCRContactSheet(
            Image.new("RGB", (200, 200), "white"),
            [
                {
                    "index": 1,
                    "sheetBox": [0, 0, 80, 80],
                    "variants": [
                        {"orientation": "upright", "sheetBox": [0, 0, 80, 80]},
                        {"orientation": "rotated_ccw_90", "sheetBox": [100, 0, 80, 80]},
                    ],
                },
                {
                    "index": 2,
                    "sheetBox": [0, 100, 80, 80],
                    "variants": [
                        {"orientation": "upright", "sheetBox": [0, 100, 80, 80]},
                        {"orientation": "rotated_ccw_90", "sheetBox": [100, 100, 80, 80]},
                    ],
                },
            ],
        )
        run_ocr.return_value = {
            "fullText": "",
            "annotations": [
                {"text": "bad1", "box": [10, 10, 20, 10], "confidence": 0.9},
                {"text": "bad2", "box": [10, 30, 20, 10], "confidence": 0.9},
                {"text": "good1", "box": [10, 110, 20, 10], "confidence": 0.9},
                {"text": "good2", "box": [10, 130, 20, 10], "confidence": 0.9},
                {"text": "good3", "box": [10, 150, 20, 10], "confidence": 0.9},
            ],
        }

        result = analyze_shelf_photo(self.image_bytes, include_debug=True)

        self.assertEqual(result["summary"]["bookCount"], 1)
        self.assertEqual(len(result["spines"]), 1)
        self.assertEqual(result["spines"][0]["index"], 1)
        self.assertEqual(result["spines"][0]["x"], 65)
        self.assertEqual(result["debug"]["ocrSheet"]["rowCount"], 2)
        self.assertFalse(result["debug"]["ocrSheet"]["rows"][0]["eligible"])
        self.assertEqual(
            [stage["label"] for stage in result["debug"]["stages"]][-2:],
            ["OCR bounding boxes", "Final result"],
        )


if __name__ == "__main__":
    unittest.main()
