from __future__ import annotations

import base64
import io

from PIL import Image

from .analysis_models import PreparedImage


def prepare_image(image_bytes: bytes, max_side: int = 1400) -> PreparedImage:
    source = Image.open(io.BytesIO(image_bytes))
    original_size = source.size
    image = fit_image(source.convert("RGB"), max_side=max_side)
    return PreparedImage(image=image, original_size=original_size)


def fit_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def image_to_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(image_to_jpeg_bytes(image, quality=88)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
