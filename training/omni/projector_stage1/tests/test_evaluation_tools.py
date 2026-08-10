import importlib.util
import pickle
import random
import tempfile
import unittest
from pathlib import Path

from training.omni.projector_stage1.ablation import (
    apply_temporal_shuffle,
    sample_exchange_order,
    sample_temporal_order,
    validate_exchange,
)
from training.omni.projector_stage1.compare_wer import compare_results
from training.omni.projector_stage1.evaluate_wer import canonical_ablation, has_eos, score_transcript
from training.omni.projector_stage1.evaluate_transcription import summarize_rows
from training.omni.projector_stage1.plot_metrics import (
    best_validation_summary,
    load_metrics,
)


class EvaluationToolsTest(unittest.TestCase):
    def test_paired_transcription_summary_uses_corpus_rates_and_latency_mean(self) -> None:
        rows = [
            {"projector_vlm_word_errors": 1, "projector_vlm_reference_words": 2,
             "projector_vlm_char_errors": 2, "projector_vlm_reference_chars": 10,
             "projector_vlm_latency_ms": 10.0, "qwen_asr_word_errors": 0,
             "qwen_asr_reference_words": 2, "qwen_asr_char_errors": 0,
             "qwen_asr_reference_chars": 10, "qwen_asr_latency_ms": 20.0},
            {"projector_vlm_word_errors": 1, "projector_vlm_reference_words": 8,
             "projector_vlm_char_errors": 1, "projector_vlm_reference_chars": 20,
             "projector_vlm_latency_ms": 30.0, "qwen_asr_word_errors": 2,
             "qwen_asr_reference_words": 8, "qwen_asr_char_errors": 2,
             "qwen_asr_reference_chars": 20, "qwen_asr_latency_ms": 40.0},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["projector_vlm_wer"], 0.2)
        self.assertEqual(summary["qwen_asr_wer"], 0.2)
        self.assertEqual(summary["projector_vlm_latency_ms_mean"], 20.0)
        self.assertAlmostEqual(summary["projector_vlm_minus_qwen_asr_cer"], 1 / 30)

    def test_canonical_ablation_accepts_waveform_zero_cli_value(self) -> None:
        self.assertEqual(canonical_ablation("waveform-zero"), "waveform-zero")

    def test_transcript_metrics_include_word_counts_and_exact_match(self) -> None:
        score = score_transcript("HELLO WORLD", "HELLO THERE")
        self.assertEqual(score["reference_words"], 2)
        self.assertEqual(score["reference_chars"], 10)
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
        self.assertEqual(result["generation_summary"]["waveform_zero"]["eos_rate"], 0.5)
        self.assertEqual(result["rows"][0]["cross_sample_shuffle_minus_normal_wer"], 1.0)
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

    def test_trusted_checkpoint_loader_uses_narrow_safe_load(self) -> None:
        from training.omni.projector_stage1.checkpoint_loading import load_trusted_checkpoint

        class FakeSerialization:
            class _Scope:
                def __enter__(self): return self
                def __exit__(self, *_): return False
            def safe_globals(self, values):
                self.values = values
                return self._Scope()

        class FakeTorch:
            serialization = FakeSerialization()

            @staticmethod
            def load(path, *, map_location, weights_only):
                self = FakeTorch.serialization
                assert map_location == "cpu"
                assert weights_only is True
                assert self.values == [__import__("pathlib").PosixPath]
                with path.open("rb") as handle:
                    return pickle.load(handle)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted.pt"
            with path.open("wb") as handle:
                pickle.dump({"value": 3}, handle)
            self.assertEqual(load_trusted_checkpoint(path, FakeTorch)["value"], 3)

    def test_temporal_shuffle_is_reproducible_and_preserves_layout_without_torch(self) -> None:
        from training.omni.projector_stage1.ablation import apply_temporal_shuffle, sample_temporal_order

        class Feature:
            shape = (3, 2)
            dtype = "bf16"

            def __init__(self, rows): self.rows = rows
            def __getitem__(self, order): return Feature([self.rows[index] for index in order])

        feature = Feature([[1, 2], [3, 4], [5, 6]])
        first = apply_temporal_shuffle(feature, random.Random(7))
        second = apply_temporal_shuffle(feature, random.Random(7))
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.shape, feature.shape)
        self.assertEqual(first.dtype, feature.dtype)
        self.assertCountEqual(first.rows, feature.rows)
        order = sample_temporal_order(3, random.Random(7))
        self.assertTrue(all(index != donor for index, donor in enumerate(order)))

    def test_temporal_shuffle_short_sequence_boundary_without_torch(self) -> None:
        from training.omni.projector_stage1.ablation import apply_temporal_shuffle

        class Feature:
            def __init__(self, rows): self.rows = rows; self.shape = (len(rows), 1)
            def __getitem__(self, order): return Feature([self.rows[index] for index in order])

        with self.assertRaises(ValueError):
            apply_temporal_shuffle(Feature([[1]]), random.Random(1))
        self.assertEqual(apply_temporal_shuffle(Feature([[1], [2]]), random.Random(1)).rows, [[2], [1]])


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

    def test_temporal_shuffle_is_reproducible_derangement_and_preserves_layout(self) -> None:
        import torch

        feature = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        first = apply_temporal_shuffle(feature, random.Random(7))
        second = apply_temporal_shuffle(feature, random.Random(7))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, feature.shape)
        self.assertEqual(first.dtype, feature.dtype)
        self.assertCountEqual(first.tolist(), feature.tolist())
        order = sample_temporal_order(3, random.Random(7))
        self.assertTrue(all(index != donor for index, donor in enumerate(order)))

    def test_temporal_shuffle_short_sequence_boundary(self) -> None:
        import torch

        with self.assertRaises(ValueError):
            apply_temporal_shuffle(torch.ones(1, 2), random.Random(1))
        swapped = apply_temporal_shuffle(torch.tensor([[1], [2]]), random.Random(1))
        self.assertEqual(swapped.tolist(), [[2], [1]])

    def test_checkpoint_loader_allowlists_only_posix_path(self) -> None:
        import torch

        from training.omni.projector_stage1.evaluate import _load_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted.pt"
            torch.save({"path": Path("sample.pt"), "value": 3}, path)
            loaded = _load_checkpoint(path, torch)
        self.assertEqual(loaded["path"], Path("sample.pt"))
        self.assertEqual(loaded["value"], 3)


if __name__ == "__main__":
    unittest.main()
