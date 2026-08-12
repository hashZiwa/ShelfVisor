import unittest

from app.analysis.yolo.yolo_processing import (
    filter_yolo_regions,
    normalize_filter_options,
    parse_yolo_predictions,
)


class YoloProcessingTests(unittest.TestCase):
    def test_predictions_are_converted_and_sorted_left_to_right(self):
        raw = {
            "predictions": [
                {"x": 80, "y": 50, "width": 20, "height": 60, "confidence": 0.8, "class": "book"},
                {"x": 20, "y": 50, "width": 20, "height": 60, "confidence": 0.9, "class": "book"},
            ]
        }

        regions = parse_yolo_predictions(raw, (100, 100))

        self.assertEqual([region.box[0] for region in regions], [10, 70])

    def test_size_filter_removes_wide_regions(self):
        regions = parse_yolo_predictions(
            {"predictions": [{"x": 50, "y": 50, "width": 80, "height": 20}]},
            (100, 100),
        )
        options = normalize_filter_options({"maxBoxWidthRatio": 0.33, "maxBoxAreaRatio": 1.0})

        self.assertEqual(filter_yolo_regions(regions, (100, 100), options), [])


if __name__ == "__main__":
    unittest.main()
