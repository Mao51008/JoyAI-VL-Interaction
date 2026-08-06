import json
import random
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from training.omni.projector_stage1.ablation import apply_temporal_shuffle
from training.omni.projector_stage1.merge_shards import merge_shard_results
from training.omni.projector_stage1.sharding import (
    global_exchange_order,
    load_sharded_samples,
    result_metadata,
    shard_indices,
    temporal_seed,
    validate_shard_args,
)


class ShardingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manifest = Path(self.temp.name) / "manifest.jsonl"
        self.manifest.write_text("{}\n" * 8, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _metadata(self, index, kind="tf"):
        indices = shard_indices(8, 4, index)
        metadata = result_metadata(
            manifest=self.manifest,
            manifest_sample_count=8,
            evaluation_sample_count=8,
            indices=indices,
            num_shards=4,
            shard_index=index,
            checkpoint=Path("best.pt"),
            ablation="none",
            seed=3407,
        )
        if kind == "tf":
            rows = [{"global_index": i, "sample_id": f"s{i}", "supervised_tokens": 2,
                     "loss": float(i + 1), "eos_tokens": 1, "eos_loss": float(i) / 10}
                    for i in indices]
        else:
            rows = [{"global_index": i, "sample_id": f"s{i}", "speaker": str(i % 2),
                     "duration_ms": 10000 + i, "word_errors": i % 3,
                     "reference_words": 5, "char_errors": i % 2,
                     "reference_chars": 10, "wer": (i % 3) / 5,
                     "cer": (i % 2) / 10, "generated_eos": i % 2 == 0,
                     "generated_tokens": i + 1, "exact_match": i == 0}
                    for i in indices]
        metadata.update({"samples": len(rows), "rows": rows})
        return metadata

    def test_four_shards_are_disjoint_and_cover_manifest(self):
        parts = [shard_indices(696, 4, index) for index in range(4)]
        self.assertEqual([len(part) for part in parts], [174] * 4)
        self.assertEqual(sorted(index for part in parts for index in part), list(range(696)))
        self.assertEqual(shard_indices(8, 1, 0), list(range(8)))

    def test_load_sharded_samples_selects_696_rows_by_global_index(self):
        all_samples = [SimpleNamespace(sample_id=f"s{i}") for i in range(696)]
        with patch("training.omni.projector_stage1.sharding.load_samples", return_value=all_samples):
            parts = [load_sharded_samples(self.manifest, None, 4, index) for index in range(4)]
        self.assertEqual([len(part[1]) for part in parts], [174] * 4)
        for manifest_samples, selected, indices in parts:
            self.assertIs(manifest_samples, all_samples)
            self.assertEqual([sample.sample_id for sample in selected], [f"s{i}" for i in indices])
        self.assertEqual(sorted(index for part in parts for index in part[2]), list(range(696)))

    def test_load_sharded_samples_max_prefix_and_default_one_zero(self):
        all_samples = [SimpleNamespace(sample_id=f"s{i}") for i in range(10)]
        with patch("training.omni.projector_stage1.sharding.load_samples", return_value=all_samples):
            default = load_sharded_samples(self.manifest, None, 1, 0)
            parts = [load_sharded_samples(self.manifest, 10, 4, index) for index in range(4)]
        self.assertEqual([sample.sample_id for sample in default[1]], [f"s{i}" for i in range(10)])
        self.assertEqual(default[2], list(range(10)))
        self.assertEqual(sorted(i for part in parts for i in part[2]), list(range(10)))
        self.assertEqual(
            sorted(sample.sample_id for part in parts for sample in part[1]),
            [f"s{i}" for i in range(10)],
        )

    def test_invalid_shard_parameters(self):
        with self.assertRaises(ValueError): validate_shard_args(0, 0)
        with self.assertRaises(ValueError): validate_shard_args(4, 4)
        with self.assertRaises(ValueError): shard_indices(-1, 1, 0)

    def test_global_shuffle_and_temporal_seed_are_shard_invariant(self):
        order = global_exchange_order(8, 3407)
        self.assertEqual(sorted(order), list(range(8)))
        self.assertTrue(all(index != donor for index, donor in enumerate(order)))
        self.assertEqual([temporal_seed(3407, i) for i in range(8)],
                         [temporal_seed(3407, i) for i in range(8)])
        feature = [[i] for i in range(3)]
        class Feature:
            shape = (3, 1)
            def __getitem__(self, indices):
                result = Feature()
                result.rows = [feature[index] for index in indices]
                return result
        value = Feature(); value.rows = feature
        self.assertEqual(apply_temporal_shuffle(value, random.Random(temporal_seed(3407, 3))).rows,
                         apply_temporal_shuffle(value, random.Random(temporal_seed(3407, 3))).rows)
        for shard in range(4):
            for index in shard_indices(8, 4, shard):
                self.assertEqual(order[index], global_exchange_order(8, 3407)[index])

    def test_tf_merge_is_token_weighted(self):
        parts = [self._metadata(i) for i in range(4)]
        merged = merge_shard_results(parts)
        single = self._metadata(0)
        single["num_shards"] = 1
        single["shard_index"] = 0
        single["global_indices"] = list(range(8))
        single["selected_samples"] = 8
        single["rows"] = [
            {"global_index": i, "sample_id": f"s{i}", "supervised_tokens": 2,
             "loss": float(i + 1), "eos_tokens": 1, "eos_loss": float(i) / 10}
            for i in range(8)
        ]
        single["shards"] = [0]
        single_merged = merge_shard_results([single])
        self.assertEqual(merged["samples"], 8)
        self.assertAlmostEqual(merged["loss"], 4.5)
        self.assertEqual(merged["supervised_tokens"], 16)
        self.assertAlmostEqual(merged["loss"], single_merged["loss"])
        self.assertEqual([row["global_index"] for row in merged["rows"]], list(range(8)))

    def test_wer_merge_is_micro_averaged_and_summarized(self):
        merged = merge_shard_results([self._metadata(i, "wer") for i in range(4)])
        self.assertEqual(merged["word_errors"], sum(i % 3 for i in range(8)))
        self.assertAlmostEqual(merged["wer"], sum(i % 3 for i in range(8)) / 40)
        self.assertEqual(merged["exact_match_count"], 1)
        self.assertIn("speaker_summary", merged)
        self.assertIn("duration_summary", merged)

    def test_old_wer_rows_recover_reference_chars_from_reference_text(self):
        parts = [self._metadata(i, "wer") for i in range(4)]
        for part in parts:
            for row in part["rows"]:
                row["reference"] = "AA BB"
                row.pop("reference_chars")
        merged = merge_shard_results(parts)
        self.assertEqual(merged["reference_chars"], 8 * 4)
        self.assertEqual(merged["duration_summary"]["10-15s"]["reference_chars"], 8 * 4)
        self.assertEqual(merged["speaker_summary"]["0"]["reference_chars"], 4 * 4)
        self.assertAlmostEqual(merged["cer"], merged["char_errors"] / (8 * 4))

    def test_old_wer_rows_without_reference_text_fail_clearly(self):
        parts = [self._metadata(i, "wer") for i in range(4)]
        for part in parts:
            for row in part["rows"]:
                row.pop("reference_chars")
        with self.assertRaisesRegex(ValueError, "reference text and reference_chars"):
            merge_shard_results(parts)

    def test_merge_rejects_missing_duplicate_and_inconsistent_shards(self):
        parts = [self._metadata(i) for i in range(4)]
        with self.assertRaises(ValueError): merge_shard_results(parts[:3])
        duplicate = json.loads(json.dumps(parts[1]))
        duplicate["shard_index"] = 0
        with self.assertRaises(ValueError): merge_shard_results([parts[0], duplicate, parts[2], parts[3]])
        inconsistent = json.loads(json.dumps(parts[1]))
        inconsistent["manifest_sha256"] = "bad"
        with self.assertRaises(ValueError): merge_shard_results([parts[0], inconsistent, parts[2], parts[3]])
        checkpoint_mismatch = json.loads(json.dumps(parts[1]))
        checkpoint_mismatch["checkpoint"] = "other.pt"
        with self.assertRaises(ValueError): merge_shard_results([parts[0], checkpoint_mismatch, parts[2], parts[3]])
        repeated_sample = json.loads(json.dumps(parts[1]))
        repeated_sample["rows"][0]["sample_id"] = parts[0]["rows"][0]["sample_id"]
        with self.assertRaises(ValueError): merge_shard_results([parts[0], repeated_sample, parts[2], parts[3]])


if __name__ == "__main__":
    unittest.main()
