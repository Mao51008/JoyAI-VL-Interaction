import json
import tempfile
import unittest
from pathlib import Path

from training.omni.projector_stage1.observability import validation_is_due, write_loss_curve


class ObservabilityTest(unittest.TestCase):
    def test_validation_frequency_and_no_validation_compatibility(self) -> None:
        self.assertFalse(validation_is_due(3, 5, 4, False))
        self.assertFalse(validation_is_due(3, 5, 4, True))
        self.assertTrue(validation_is_due(5, 5, 4, True))
        self.assertTrue(validation_is_due(4, 5, 4, True))

    def test_loss_curve_contains_train_and_validation_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            curve = Path(directory) / "loss_curve.svg"
            metrics.write_text(
                "\n".join(
                    json.dumps(record)
                    for record in (
                        {"step": 1, "loss": 2.0},
                        {"step": 2, "loss": 1.5, "validation_loss": 1.8},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            write_loss_curve(metrics, curve)
            content = curve.read_text(encoding="utf-8")
        self.assertIn("stroke=\"#1f77b4\"", content)
        self.assertIn("stroke=\"#d62728\"", content)

    def test_loss_curve_supports_metrics_without_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            curve = Path(directory) / "loss_curve.svg"
            metrics.write_text('{"step": 1, "loss": 2.0}\n', encoding="utf-8")
            write_loss_curve(metrics, curve)
            self.assertTrue(curve.is_file())
            self.assertNotIn("#d62728", curve.read_text(encoding="utf-8"))
