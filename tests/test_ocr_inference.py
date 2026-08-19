import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.analysis.ocr.ocr_inference import run_text_detection_inference


class OCRInferenceTests(unittest.TestCase):
    @patch("app.analysis.ocr.ocr_inference._credentials_path", return_value="credentials.json")
    @patch("app.analysis.ocr.ocr_inference._client")
    def test_text_detection_normalizes_individual_annotations_without_word_confidence(
        self, client, _credentials_path
    ):
        aggregate = SimpleNamespace(
            description="500\n519.5\nㅅ21\n",
            bounding_poly=SimpleNamespace(vertices=[]),
        )
        words = [
            SimpleNamespace(
                description="500",
                bounding_poly=SimpleNamespace(
                    vertices=[
                        SimpleNamespace(x=10, y=20),
                        SimpleNamespace(x=40, y=20),
                        SimpleNamespace(x=40, y=30),
                        SimpleNamespace(x=10, y=30),
                    ]
                ),
            ),
            SimpleNamespace(
                description="519.5",
                bounding_poly=SimpleNamespace(
                    vertices=[
                        SimpleNamespace(x=10, y=40),
                        SimpleNamespace(x=55, y=40),
                        SimpleNamespace(x=55, y=50),
                        SimpleNamespace(x=10, y=50),
                    ]
                ),
            ),
        ]
        client.return_value.text_detection.return_value = SimpleNamespace(
            error=SimpleNamespace(message=""),
            text_annotations=[aggregate, *words],
        )

        result = run_text_detection_inference(b"png")

        self.assertEqual(result["fullText"], "500\n519.5\nㅅ21\n")
        self.assertEqual(
            result["annotations"],
            [
                {"text": "500", "box": [10, 20, 30, 10], "confidence": None},
                {"text": "519.5", "box": [10, 40, 45, 10], "confidence": None},
            ],
        )


if __name__ == "__main__":
    unittest.main()
