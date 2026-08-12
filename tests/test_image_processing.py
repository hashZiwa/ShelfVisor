import io
import unittest

from PIL import Image

from app.analysis.image_processing import image_to_data_url, prepare_image


class ImageProcessingTests(unittest.TestCase):
    def test_prepare_image_converts_rgb_and_limits_long_side(self):
        source = Image.new("RGBA", (2000, 1000), (255, 0, 0, 128))
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")

        prepared = prepare_image(buffer.getvalue(), max_side=1400)

        self.assertEqual(prepared.image.mode, "RGB")
        self.assertEqual(prepared.image.size, (1400, 700))
        self.assertEqual(prepared.original_size, (2000, 1000))

    def test_image_to_data_url_returns_jpeg_url(self):
        value = image_to_data_url(Image.new("RGB", (10, 10), "white"))
        self.assertTrue(value.startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
