from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .image_processing import image_to_data_url


class DebugArtifactCollector:
    def __init__(
        self,
        enabled: bool,
        write_local: bool = False,
        root: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.write_local = write_local
        configured_root = os.environ.get("SHELFVISOR_DEBUG_ROOT")
        self.root = root or (Path(configured_root) if configured_root else Path(__file__).resolve().parents[2] / "debug_runs")
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.metadata: dict[str, Any] = {}
        self._stages: list[dict[str, Any]] = []

    def add_image(
        self,
        key: str,
        label: str,
        image: Image.Image,
        details: list[str] | None = None,
        number: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._stages.append({
            "key": key,
            "label": label,
            "image_object": image.copy(),
            "details": details,
            "number": number,
        })

    def set_metadata(self, **values: Any) -> None:
        if self.enabled:
            self.metadata.update(values)

    def build_payload(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        payload = dict(self.metadata)
        payload["runId"] = self.run_id
        payload["stages"] = []
        for stage in self._stages:
            value = {"label": stage["label"], "image": image_to_data_url(stage["image_object"])}
            if stage["number"] is not None:
                value["number"] = stage["number"]
            if stage["details"] is not None:
                value["details"] = stage["details"]
            payload["stages"].append(value)
        return payload

    def write(self) -> Path | None:
        if not self.enabled or not self.write_local:
            return None
        run_directory = self.root / self.run_id
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
            stage_files = []
            for index, stage in enumerate(self._stages, start=1):
                key = _safe_key(stage["key"])
                use_png = "contact-sheet" in key
                suffix = ".png" if use_png else ".jpg"
                filename = f"{index:02d}-{key}{suffix}"
                image = stage["image_object"]
                if use_png:
                    image.save(run_directory / filename, format="PNG")
                else:
                    image.convert("RGB").save(run_directory / filename, format="JPEG", quality=88)
                stage_file = {"key": stage["key"], "label": stage["label"], "file": filename}
                if stage["number"] is not None:
                    stage_file["number"] = stage["number"]
                stage_files.append(stage_file)
            manifest = {**self.metadata, "runId": self.run_id, "createdAt": datetime.now(timezone.utc).isoformat(), "stages": stage_files}
            (run_directory / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return run_directory
        except OSError as error:
            self.metadata["artifactWriteError"] = str(error)
            return None


def _safe_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "stage"
