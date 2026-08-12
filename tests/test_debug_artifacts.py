import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.analysis.debug_artifacts import DebugArtifactCollector


class DebugArtifactCollectorTests(unittest.TestCase):
    def test_disabled_collector_creates_no_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "debug"
            collector = DebugArtifactCollector(enabled=False, write_local=True, root=root)
            collector.add_image("original", "Original", Image.new("RGB", (4, 4), "white"))

            self.assertIsNone(collector.write())
            self.assertFalse(root.exists())
            self.assertEqual(collector.build_payload(), {})

    def test_enabled_local_collector_writes_manifest_and_numbered_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "debug"
            collector = DebugArtifactCollector(enabled=True, write_local=True, root=root)
            collector.set_metadata(model="test-model")
            collector.add_image("original", "Original", Image.new("RGB", (4, 4), "white"))
            collector.add_image("final", "Final result", Image.new("RGB", (4, 4), "black"))

            run_directory = collector.write()

            self.assertIsNotNone(run_directory)
            assert run_directory is not None
            self.assertTrue((run_directory / "manifest.json").is_file())
            self.assertEqual(
                sorted(path.name for path in run_directory.glob("*.jpg")),
                ["01-original.jpg", "02-final.jpg"],
            )
            manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"], "test-model")
            payload = collector.build_payload()
            self.assertEqual([stage["label"] for stage in payload["stages"]], ["Original", "Final result"])
            self.assertTrue(payload["stages"][0]["image"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
