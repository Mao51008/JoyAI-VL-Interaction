import json
import unittest

from training.omni.midasheng.prepare_phase1_manifest import prepare_manifest
from training.omni.midasheng.sampling import PHASE1_TASK_WEIGHTS, sample_task_rows, task_draw_counts


class Phase1SamplingTest(unittest.TestCase):
    def test_phase_one_counts_follow_explicit_weights(self) -> None:
        self.assertEqual(task_draw_counts(100, PHASE1_TASK_WEIGHTS), {
            "voiceassistant": 40, "librispeech": 40, "clotho_aqa": 20,
        })

    def test_task_sampling_is_seeded_and_uses_replacement(self) -> None:
        rows = {task: [{"sample_id": task}] for task in PHASE1_TASK_WEIGHTS}
        first = sample_task_rows(rows, total_examples=12, seed=7)
        self.assertEqual(first, sample_task_rows(rows, total_examples=12, seed=7))
        self.assertEqual({task: sum(task == value[0] for value in first) for task in rows}, {
            "voiceassistant": 5, "librispeech": 5, "clotho_aqa": 2,
        })

    def test_prepare_manifest_records_sampler_provenance_without_writing_sources(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_paths = {}
            for task in PHASE1_TASK_WEIGHTS:
                path = root / f"{task}.jsonl"
                path.write_text(json.dumps({"sample_id": task, "metadata": {"source": task}}) + "\n", encoding="utf-8")
                source_paths[task] = path
            output = root / "out" / "phase1.jsonl"
            result = prepare_manifest(
                voiceassistant_manifest=source_paths["voiceassistant"], librispeech_manifest=source_paths["librispeech"],
                clotho_aqa_manifest=source_paths["clotho_aqa"], output_manifest=output, total_examples=4, seed=13,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(result["task_counts"], {"voiceassistant": 2, "librispeech": 1, "clotho_aqa": 1})
            self.assertTrue(all("midasheng_phase1_sampler" in row["metadata"] for row in rows))
            self.assertTrue(all(row["sample_id"].startswith("midasheng-") for row in rows))
            self.assertTrue(all(path.read_text(encoding="utf-8").count("\n") == 1 for path in source_paths.values()))
