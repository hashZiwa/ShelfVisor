from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class PreparedImage:
    image: Image.Image
    original_size: tuple[int, int]


@dataclass
class DetectedRegion:
    box: tuple[int, int, int, int]
    polygon: list[list[int]]
    confidence: float | None = None
    class_name: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRContactSheet:
    image: Image.Image
    rows: list[dict[str, Any]]


@dataclass
class OCRRowResult:
    index: int
    text: str
    tokens: list[dict[str, Any]]
    selected_orientation: str
    data: dict[str, Any]


@dataclass
class OrderStatus:
    label: str
    expected_rank: int
    status: str


@dataclass
class SpineAnalysis:
    index: int
    x: int
    y: int
    width: int
    height: int
    call_number: str
    expected_rank: int
    status: str
    polygon: list[list[int]]
    confidence: float | None
    class_name: str | None
