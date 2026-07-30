import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from training.omni.schema import load_samples

MODULE_PATH = Path(__file__).parents[1] / "convert_audit_to_omni.py"
SPEC = importlib.util.spec_from_file_location("convert_audit_to_omni", MODULE_PATH)
assert SPEC and SPEC.loader
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class ConvertAuditToOmniTest(unittest.TestCase):
    def test_audited_video_becomes_valid_audio_video_text_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "source": "CharadesEgo",
                            "video_name": "ABC_action_1",
                            "task_type": "chat",
                            "question": [{"content": "What?", "time": "1"}],
                            "response": [{"content": "Answer.", "time": "2"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            audit = root / "audit.jsonl"
            audit.write_text(
                json.dumps(
                    {
                        "video_name": "ABC_action_1",
                        "media_path": str(root / "ABC_action_1.mp4"),
                        "media_usable": True,
                        "audio_usable": True,
                        "duration_s": 3.0,
                        "audio_duration_s": 3.0,
                        "audio_sample_rate": 16000,
                        "frame_status": "ok",
                        "frames": [
                            {"path": str(root / "frame_0.jpg"), "timestamp_ms": 0},
                            {"path": str(root / "frame_1.jpg"), "timestamp_ms": 1000},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            provenance = root / "provenance.json"
            provenance.write_text(
                json.dumps(
                    {
                        "dataset": "test",
                        "version": "1",
                        "source_uri": "local://test",
                        "license_name": "test research",
                        "license_tier": "research_only",
                        "allows_training": True,
                        "allows_modification": True,
                        "allows_redistribution": False,
                        "allows_commercial_use": False,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "omni.jsonl"
            summary = CONVERTER.convert(
                annotations,
                audit,
                provenance,
                output,
                source="CharadesEgo",
            )
            self.assertEqual(summary["converted_samples"], 1)
            sample = load_samples(output)[0]
            self.assertTrue(sample.audio)
            self.assertTrue(sample.video)
            self.assertTrue(sample.text)
            self.assertEqual(sample.audio[0].channel, "environment_audio")
            self.assertEqual(
                sample.metadata["audio_semantics"],
                "original_scene_audio_not_user_speech",
            )

    def test_timestamp_equal_to_duration_is_rejected(self) -> None:
        converted = CONVERTER._convert_targets(
            [{"content": "too late", "time": "3"}],
            3000,
        )
        self.assertIsNone(converted)


if __name__ == "__main__":
    unittest.main()
