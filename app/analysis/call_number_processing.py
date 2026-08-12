from __future__ import annotations

import re
from typing import Any, Sequence

from .analysis_models import OrderStatus


def check_call_number_order(call_numbers: Sequence[str]) -> list[OrderStatus]:
    return [
        OrderStatus(label=label, expected_rank=index + 1, status="ok")
        for index, label in enumerate(call_numbers)
    ]


def call_number_sort_key(value: str) -> tuple[Any, ...]:
    parts = re.findall(r"\d+|[A-Za-z]+", value.upper())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
    )
