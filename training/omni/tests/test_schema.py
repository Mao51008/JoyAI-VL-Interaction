import unittest

from training.omni.schema import ActionKind, OmniSample
from training.omni.timeline import build_decision_steps


def provenance() -> dict:
    return {
        "dataset": "synthetic",
        "version": "1",
        "source_uri": "local://test",
        "license_name": "CC0-1.0",
        "license_tier": "redistributable",
        "allows_training": True,
        "allows_modification": True,
        "allows_redistribution": True,
        "allows_commercial_use": True,
    }


class OmniSchemaTest(unittest.TestCase):
    def test_audio_only_without_text_is_valid(self) -> None:
        sample = OmniSample.from_dict(
            {
                "schema_version": "omni-training-v1",
                "sample_id": "audio-only",
                "duration_ms": 2000,
                "provenance": provenance(),
                "audio": [
                    {
                        "path": "fake://audio.wav",
                        "start_ms": 0,
                        "end_ms": 2000,
                        "sample_rate": 16000,
                        "num_samples": 32000,
                    }
                ],
                "targets": [
                    {
                        "timestamp_ms": 1000,
                        "action": "response",
                        "text": "answer",
                    }
                ],
            }
        )
        self.assertEqual(sample.modality_presence, (True, False, False))

    def test_license_that_disallows_training_is_rejected(self) -> None:
        blocked = provenance()
        blocked["allows_training"] = False
        with self.assertRaisesRegex(ValueError, "does not allow training"):
            OmniSample.from_dict(
                {
                    "schema_version": "omni-training-v1",
                    "sample_id": "blocked",
                    "duration_ms": 1000,
                    "provenance": blocked,
                    "text": [{"text": "hello", "timestamp_ms": 0}],
                }
            )

    def test_invalid_audio_length_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            OmniSample.from_dict(
                {
                    "schema_version": "omni-training-v1",
                    "sample_id": "bad-audio",
                    "duration_ms": 1000,
                    "provenance": provenance(),
                    "audio": [
                        {
                            "path": "fake://bad.wav",
                            "start_ms": 0,
                            "end_ms": 1000,
                            "sample_rate": 16000,
                            "num_samples": 8000,
                        }
                    ],
                }
            )

    def test_sparse_targets_become_dense_silence_steps(self) -> None:
        sample = OmniSample.from_dict(
            {
                "schema_version": "omni-training-v1",
                "sample_id": "dense-actions",
                "duration_ms": 3000,
                "provenance": provenance(),
                "text": [{"text": "hello", "timestamp_ms": 0}],
                "targets": [
                    {
                        "timestamp_ms": 2000,
                        "action": "delegate",
                        "text": "background task",
                    }
                ],
            }
        )
        steps = build_decision_steps(sample)
        self.assertEqual(
            [step.action for step in steps],
            [ActionKind.SILENCE, ActionKind.SILENCE, ActionKind.DELEGATE],
        )


if __name__ == "__main__":
    unittest.main()
