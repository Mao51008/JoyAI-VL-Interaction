import json
import unittest

from training.omni.midasheng.cache_features import checkpoint_identity


class MiDashengCheckpointTest(unittest.TestCase):
    def test_checkpoint_identity_accepts_final_midasheng_encoder_config(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "model_type": "midashenglm", "audio_encoder_config": {"embed_dim": 1280, "sample_rate": 16000},
            }), encoding="utf-8")
            identity = checkpoint_identity(root)
            self.assertEqual(identity["audio_encoder_dim"], 1280)
            self.assertEqual(identity["sample_rate"], 16000)

    def test_checkpoint_identity_rejects_an_unrelated_model(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "model_type": "qwen3_asr", "audio_encoder_config": {"embed_dim": 2048, "sample_rate": 16000},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "MiDashengLM"):
                checkpoint_identity(root)
