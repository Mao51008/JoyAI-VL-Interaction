import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.omni.batching import (
    FakeAudioEncoder,
    FakeStreamingAudioState,
    collate_samples,
)
from training.omni.checkpoint import load_latest
from training.omni.fake_model import FakeOmniDecisionModel
from training.omni.schema import load_samples
from training.omni.train_fake import run

ROOT = Path(__file__).parents[3]
EXAMPLES = ROOT / "training" / "omni" / "examples" / "b2_fake_samples.jsonl"


class B2FakeTrainingTest(unittest.TestCase):
    def test_collator_supports_mixed_and_missing_modalities(self) -> None:
        samples = load_samples(EXAMPLES)
        batch = collate_samples(samples, FakeAudioEncoder())
        self.assertEqual(batch.audio_hidden_states.shape[0], 4)
        self.assertEqual(batch.audio_hidden_states.shape[2], 8)
        self.assertEqual(batch.modality_presence.tolist()[0], [1.0, 0.0, 0.0])
        self.assertEqual(batch.modality_presence.tolist()[-1], [0.0, 1.0, 1.0])
        self.assertTrue(np.all(batch.action_labels[~batch.step_mask] == -100))

    def test_fake_encoder_commits_one_second_and_bounds_active_window(self) -> None:
        encoder = FakeAudioEncoder()
        stream = FakeStreamingAudioState(encoder, "stream")
        partial = stream.append(1500)
        self.assertEqual(partial.committed_length, 13)
        bounded = stream.append(7500)
        self.assertEqual(bounded.hidden_states.shape[0], 104)
        self.assertEqual(bounded.history_token_count, 13)
        stream.reset()
        self.assertEqual(stream.total_ms, 0)
        self.assertEqual(stream.history_token_count, 0)

    def test_fake_training_updates_bridge_and_loss(self) -> None:
        samples = load_samples(EXAMPLES)
        batch = collate_samples(samples, FakeAudioEncoder())
        model = FakeOmniDecisionModel()
        original = model.bridge.copy()
        losses = [model.train_step(batch, 0.1) for _ in range(30)]
        self.assertLess(losses[-1], losses[0])
        self.assertFalse(np.array_equal(model.bridge, original))

    def test_checkpoint_resume_restores_and_continues(self) -> None:
        config = {
            "step_ms": 1000,
            "learning_rate": 0.1,
            "total_steps": 2,
            "save_every": 1,
            "fake_audio_encoder": {
                "hidden_size": 8,
                "tokens_per_second": 13,
                "commit_ms": 1000,
                "active_window_ms": 8000,
            },
            "fake_model": {"bridge_hidden_size": 6, "seed": 7},
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = run(EXAMPLES, output, config, resume=False)
            self.assertEqual(first["final_step"], 2)
            config["total_steps"] = 4
            resumed = run(EXAMPLES, output, config, resume=True)
            self.assertEqual(resumed["initial_step"], 2)
            self.assertEqual(resumed["final_step"], 4)
            _model, state, checkpoint = load_latest(output)
            self.assertEqual(state.global_step, 4)
            self.assertEqual(checkpoint.name, "checkpoint-00000004.json")
            latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["global_step"], 4)


if __name__ == "__main__":
    unittest.main()
