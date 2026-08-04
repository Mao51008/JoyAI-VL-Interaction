import json
import tempfile
import unittest
from pathlib import Path

from training.omni.projector_stage1.preflight import run


class Stage1PreflightTest(unittest.TestCase):
    def _sample(self, audio_path: Path) -> dict:
        return {
            "schema_version": "omni-training-v1",
            "sample_id": "preflight-sample",
            "duration_ms": 1000,
            "provenance": {
                "dataset": "test",
                "version": "1",
                "source_uri": "local://test",
                "license_name": "CC0-1.0",
                "license_tier": "redistributable",
                "allows_training": True,
                "allows_modification": True,
                "allows_redistribution": True,
                "allows_commercial_use": True,
            },
            "audio": [{
                "path": str(audio_path),
                "start_ms": 0,
                "end_ms": 1000,
                "sample_rate": 16000,
                "num_samples": 16000,
                "channel": "user_audio",
            }],
            "video": [],
            "text": [],
            "targets": [],
            "metadata": {"assistant_target_text": "test transcript"},
        }

    def test_accepts_config_after_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.flac"
            audio.write_bytes(b"test")
            config = root / "config.json"
            config.write_text(
                json.dumps({"stage": "projector_only_audio_alignment", "dimensions_require_probe": False}),
                encoding="utf-8",
            )
            manifest = root / "samples.jsonl"
            manifest.write_text(json.dumps(self._sample(audio)) + "\n", encoding="utf-8")

            result = run(config, manifest)

        self.assertFalse(result["dimensions_require_probe"])

    def test_rejects_config_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                json.dumps({"stage": "projector_only_audio_alignment", "dimensions_require_probe": True}),
                encoding="utf-8",
            )
            manifest = root / "samples.jsonl"
            manifest.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "still requires"):
                run(config, manifest, check_media=False)


if __name__ == "__main__":
    unittest.main()
