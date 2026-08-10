import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "download_omni_stage3_media.py"
SPEC = importlib.util.spec_from_file_location("download_omni_stage3_media", MODULE_PATH)
assert SPEC and SPEC.loader
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


class DownloadOmniStage3MediaTest(unittest.TestCase):
    def test_plan_respects_source_and_total_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = [
                {"source": "a", "video_name": "one", "download_url": "https://example/a", "relative_path": "a/one.mp4", "size_bytes": 7},
                {"source": "a", "video_name": "two", "download_url": "https://example/b", "relative_path": "a/two.mp4", "size_bytes": 7},
                {"source": "b", "video_name": "three", "download_url": "https://example/c", "relative_path": "b/three.mp4", "size_bytes": 6},
            ]
            plan = DOWNLOADER.build_plan(rows, Path(temporary), {"a": 10, "b": 10}, total_limit_bytes=13, max_file_bytes=None)
            self.assertEqual([item["planned_status"] for item in plan], ["ready", "blocked", "ready"])
            self.assertEqual(plan[1]["planned_reason"], "source_quota_exceeded")

    def test_destination_cannot_escape_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                DOWNLOADER.safe_destination(Path(temporary), "../escape.mp4")


if __name__ == "__main__":
    unittest.main()
