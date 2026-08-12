import io
import unittest
from unittest.mock import patch

from PIL import Image

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
            "annotations": [{"text": "811.3 A", "box": [100, 40, 10, 10], "confidence": 0.95}],
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


if __name__ == "__main__":
    unittest.main()
