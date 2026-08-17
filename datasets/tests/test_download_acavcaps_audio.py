import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "download_acavcaps_audio.py"
SPEC = importlib.util.spec_from_file_location("download_acavcaps_audio", MODULE_PATH)
assert SPEC and SPEC.loader
DOWNLOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOADER)


class DownloadAcavcapsAudioTest(unittest.TestCase):
    def test_command_keeps_clip_range_and_proxy(self) -> None:
        row = {
            "sample_key": "wn17EHKNLoE_85_9075_95_9075",
            "category": "00A",
            "video_id": "wn17EHKNLoE",
            "start_seconds": 85.9075,
            "end_seconds": 95.9075,
        }
        command = DOWNLOADER.yt_dlp_command(row, Path("/data/maoyy/audio"), "yt-dlp", "http://127.0.0.1:17897")
        self.assertIn("*85.9075-95.9075", command)
        self.assertIn("http://127.0.0.1:17897", command)
        self.assertIn("https://www.youtube.com/watch?v=wn17EHKNLoE", command)
        output = command[command.index("--output") + 1].replace("\\", "/")
        self.assertEqual(output, "/data/maoyy/audio/00A/wn17EHKNLoE_85_9075_95_9075.%(ext)s")


if __name__ == "__main__":
    unittest.main()
