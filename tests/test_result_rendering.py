import unittest

from PIL import Image, ImageChops

from app.analysis.result_rendering import _ocr_confidence_labels, grid_ocr_overlay_tiles


class OCRDebugRenderingTests(unittest.TestCase):
    def _row(self, upright_confidences, rotated_confidences):
        def tokens(x, y, confidences):
            return [
                {
                    "text": str(index),
                    "box": [x + 4, y + 4 + index * 10, 8, 8],
                    "confidence": confidence,
                }
                for index, confidence in enumerate(confidences)
            ]

        return {
            "index": 1,
            "sheetBox": [20, 20, 40, 60],
            "selectedOrientation": "upright",
            "variantResults": [
                {
                    "orientation": "upright",
                    "sheetBox": [20, 20, 40, 60],
                    "tokens": tokens(20, 20, upright_confidences),
                },
                {
                    "orientation": "rotated_ccw_90",
                    "sheetBox": [100, 30, 60, 40],
                    "tokens": tokens(100, 30, rotated_confidences),
                },
            ],
        }

    def test_confidence_labels_follow_token_order_and_use_two_decimals(self):
        tokens = [
            {"confidence": 0.951},
            {"confidence": 0.874},
        ]

        self.assertEqual(
            _ocr_confidence_labels(tokens),
            ["box1: 0.95", "box2: 0.87"],
        )

    def test_each_orientation_starts_numbering_at_box_one(self):
        upright = [{"confidence": 0.95}, {"confidence": 0.87}]
        rotated = [{"confidence": 0.63}]

        self.assertEqual(_ocr_confidence_labels(upright)[0], "box1: 0.95")
        self.assertEqual(_ocr_confidence_labels(rotated)[0], "box1: 0.63")

    def test_empty_orientation_has_no_confidence_labels(self):
        self.assertEqual(_ocr_confidence_labels([]), [])

    def test_grid_adds_caption_pixels_below_both_orientation_images(self):
        row = self._row([0.95], [0.63])
        source = Image.new("RGB", (200, 120), "white")

        rendered = grid_ocr_overlay_tiles(source, [row])

        upright_caption_area = rendered.crop((38, 118, 98, 130))
        rotated_caption_area = rendered.crop((118, 108, 188, 120))
        upright_difference = ImageChops.difference(
            upright_caption_area,
            Image.new("RGB", upright_caption_area.size, "white"),
        )
        rotated_difference = ImageChops.difference(
            rotated_caption_area,
            Image.new("RGB", rotated_caption_area.size, "white"),
        )
        self.assertIsNotNone(upright_difference.getbbox())
        self.assertIsNotNone(rotated_difference.getbbox())

    def test_more_tokens_expand_the_debug_tile_height(self):
        source = Image.new("RGB", (200, 120), "white")

        one_line = grid_ocr_overlay_tiles(source, [self._row([0.95], [])])
        three_lines = grid_ocr_overlay_tiles(source, [self._row([0.95, 0.87, 0.72], [])])

        self.assertGreater(three_lines.height, one_line.height)


if __name__ == "__main__":
    unittest.main()
