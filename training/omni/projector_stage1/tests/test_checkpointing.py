import tempfile
import unittest
from pathlib import Path

from training.omni.projector_stage1.checkpointing import resume_checkpoint_path, should_save_last


class Stage1CheckpointingTest(unittest.TestCase):
    def test_last_checkpoint_schedule(self) -> None:
        self.assertTrue(should_save_last(1_000, total_steps=10_731, every_steps=1_000, steps_per_epoch=3_577))
        self.assertTrue(should_save_last(3_577, total_steps=10_731, every_steps=1_000, steps_per_epoch=3_577))
        self.assertTrue(should_save_last(10_731, total_steps=10_731, every_steps=1_000, steps_per_epoch=3_577))
        self.assertFalse(should_save_last(999, total_steps=10_731, every_steps=1_000, steps_per_epoch=3_577))

    def test_resume_prefers_last_and_accepts_legacy_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "latest.pt").touch()
            self.assertEqual(resume_checkpoint_path(output).name, "latest.pt")
            (output / "last.pt").touch()
            self.assertEqual(resume_checkpoint_path(output).name, "last.pt")
