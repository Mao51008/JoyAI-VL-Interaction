import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

from training.omni.projector_stage1.ablation import (
    sample_exchange_order,
    validate_exchange,
)
from training.omni.projector_stage1.compare_wer import compare_results
from training.omni.projector_stage1.evaluate_wer import has_eos, score_transcript
from training.omni.projector_stage1.plot_metrics import (
    best_validation_summary,
    load_metrics,
)


class EvaluationToolsTest(unittest.TestCase):
    def test_transcript_metrics_include_word_counts_and_exact_match(self) -> None:
        score = score_transcript("HELLO WORLD", "HELLO THERE")
        self.assertEqual(score["reference_words"], 2)
        self.assertEqual(score["hypothesis_words"], 2)
        self.assertFalse(score["exact_match"])
        self.assertEqual(score["word_errors"], 1)
        self.assertEqual(score["insertions"], 0)
        self.assertEqual(score["deletions"], 0)
        self.assertEqual(score["substitutions"], 1)
        self.assertEqual(score["wer"], 0.5)
        self.assertTrue(score_transcript("A B", "A B")["exact_match"])

    def test_transcript_metrics_separate_insertions_and_deletions(self) -> None:
        inserted = score_transcript("A B", "A B C")
        deleted = score_transcript("A B C", "A B")
        self.assertEqual((inserted["insertions"], inserted["deletions"], inserted["substitutions"]), (1, 0, 0))
        self.assertEqual((deleted["insertions"], deleted["deletions"], deleted["substitutions"]), (0, 1, 0))

    def test_eos_detection(self) -> None:
        class Tokens:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        self.assertTrue(has_eos(Tokens([1, 2, 3]), 2))
        self.assertFalse(has_eos(Tokens([1, 2, 3]), None))

    def test_comparison_reports_rates_deltas_and_typical_failure(self) -> None:
        normal = {"rows": [
            {"sample_id": "a", "hypothesis": "GOOD", "wer": 0.0, "cer": 0.0, "generated_tokens": 2, "generated_eos": True},
            {"sample_id": "b", "hypothesis": "OK", "wer": 0.2, "cer": 0.1, "generated_tokens": 4, "generated_eos": False},
        ]}
        zero = {"rows": [
            {"sample_id": "a", "hypothesis": "BAD", "wer": 1.0, "cer": 1.0, "generated_tokens": 3, "generated_eos": False},
            {"sample_id": "b", "hypothesis": "OK", "wer": 0.2, "cer": 0.1, "generated_tokens": 5, "generated_eos": True},
        ]}
        shuffle = {"rows": [
            {"sample_id": "a", "hypothesis": "BAD", "wer": 1.0, "cer": 1.0, "generated_tokens": 3, "generated_eos": True},
            {"sample_id": "b", "hypothesis": "OK", "wer": 0.2, "cer": 0.1, "generated_tokens": 5, "generated_eos": False},
        ]}
        result = compare_results(normal, zero, shuffle, typical_count=1)
        self.assertEqual(result["normal_shuffle_identical_rate"], 0.5)
        self.assertEqual(result["generation_summary"]["normal"]["average_generated_tokens"], 3.0)
        self.assertEqual(result["generation_summary"]["zero"]["eos_rate"], 0.5)
        self.assertEqual(result["rows"][0]["shuffle_minus_normal_wer"], 1.0)
        self.assertEqual(result["typical_failures"][0]["sample_id"], "a")

    def test_metrics_loading_and_best_validation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_text(
                '{"step": 1, "loss": 2.0, "learning_rate": 0.01}\n'
                '{"step": 2, "loss": 1.5, "learning_rate": 0.005, "validation_loss": 1.2}\n'
                '{"step": 3, "loss": 1.0, "learning_rate": 0.001, "validation_loss": 1.3}\n',
                encoding="utf-8",
            )
            summary = best_validation_summary(load_metrics(metrics))
        self.assertEqual(summary["best_validation_step"], 2)
        self.assertEqual(summary["best_validation_loss"], 1.2)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class AudioAblationTest(unittest.TestCase):
    def test_per_sample_token_losses_include_eos(self) -> None:
        import torch

        from training.omni.projector_stage1.evaluate import per_sample_token_losses

        logits = torch.tensor([[[0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
        labels = torch.tensor([[-100, 1, 2]])
        rows = per_sample_token_losses(logits, labels, eos_token_id=2)
        self.assertEqual(rows[0]["supervised_tokens"], 2)
        self.assertEqual(rows[0]["eos_tokens"], 1)
        self.assertIsNotNone(rows[0]["eos_loss"])

    def test_feature_zero_is_distinct_from_projected_zero(self) -> None:
        import torch

        from training.omni.projector_stage1.evaluate import apply_audio_ablation
        feature = torch.tensor([[1.0], [2.0], [3.0]])
        self.assertTrue(torch.equal(apply_audio_ablation(feature, "none"), feature))
        self.assertTrue(torch.equal(apply_audio_ablation(feature, "feature-zero"), torch.zeros_like(feature)))
        self.assertTrue(torch.equal(apply_audio_ablation(feature, "projected-zero"), feature))

    def test_sample_exchange_is_a_derangement(self) -> None:
        for seed in range(20):
            order = sample_exchange_order(5, random.Random(seed))
            validate_exchange(list(range(5)), order)
            self.assertEqual(sorted(order), list(range(5)))
            self.assertTrue(all(index != donor for index, donor in enumerate(order)))

    def test_sample_exchange_requires_two_samples(self) -> None:
        with self.assertRaises(ValueError):
            sample_exchange_order(1, random.Random(1))


if __name__ == "__main__":
    unittest.main()
