import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "audit_annotations.py"
SPEC = importlib.util.spec_from_file_location("audit_annotations", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def sample_record(
    video_name: str,
    source: str = "source-a",
    question_time: str = "1",
    response_time: str = "2",
) -> dict:
    return {
        "video_name": video_name,
        "task_type": "chat",
        "source": source,
        "question": [{"content": f"question-{video_name}", "time": question_time}],
        "response": [{"content": f"response-{video_name}", "time": response_time}],
    }


class AuditAnnotationsTest(unittest.TestCase):
    def test_parse_time_values_supports_converter_format(self) -> None:
        self.assertEqual(AUDIT.parse_time_values("5, 6.5,7"), [5.0, 6.5, 7.0])
        self.assertEqual(AUDIT.parse_time_values(""), [])

    def test_audit_aggregates_video_and_timing(self) -> None:
        records = [
            sample_record("video-1", response_time="3"),
            sample_record("video-1", question_time="4", response_time="4"),
            sample_record(
                "video-2", source="source-b", question_time="10", response_time="9"
            ),
        ]
        summary, videos = AUDIT.audit_records(records)

        self.assertEqual(summary["records"], 3)
        self.assertEqual(summary["unique_source_video_pairs"], 2)
        self.assertEqual(summary["issues"], {})
        self.assertEqual(
            summary["question_response_timing"]["response_after_question"], 1
        )
        self.assertEqual(summary["question_response_timing"]["response_same_time"], 1)
        self.assertEqual(
            summary["question_response_timing"]["response_before_question"], 1
        )
        first = next(row for row in videos if row["video_name"] == "video-1")
        self.assertEqual(first["annotation_count"], 2)
        self.assertEqual(first["max_annotation_time_s"], 4.0)

    def test_invalid_fields_are_reported_without_stopping_audit(self) -> None:
        record = sample_record("video-1")
        record["question"][0]["time"] = "not-a-time"
        summary, _ = AUDIT.audit_records([record])
        self.assertEqual(summary["issues"]["invalid_question_time"], 1)

    def test_stratified_sample_is_deterministic_and_covers_sources(self) -> None:
        records = [
            sample_record(f"a-{index}", source="source-a", response_time=str(index + 1))
            for index in range(8)
        ]
        records.extend(
            sample_record(
                f"b-{index}", source="source-b", response_time=str(70 + index)
            )
            for index in range(3)
        )
        _, videos = AUDIT.audit_records(records)
        first = AUDIT.stratified_sample(videos, sample_size=6, seed=42)
        second = AUDIT.stratified_sample(videos, sample_size=6, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual({row["source"] for row in first}, {"source-a", "source-b"})
        self.assertTrue(
            all(row["upstream_lookup_status"] == "pending" for row in first)
        )

    def test_sample_includes_count_and_timestamp_outliers(self) -> None:
        records = [sample_record(f"video-{index}") for index in range(30)]
        _, videos = AUDIT.audit_records(records)
        videos[0]["annotation_count"] = 999
        videos[1]["max_annotation_time_s"] = 999
        videos[1]["annotation_time_bucket"] = AUDIT.annotation_time_bucket(999)

        selected = AUDIT.stratified_sample(videos, sample_size=20, seed=42)
        by_name = {row["video_name"]: row for row in selected}

        self.assertIn(videos[0]["video_name"], by_name)
        self.assertIn(
            "high_annotation_count",
            by_name[videos[0]["video_name"]]["selection_reason"],
        )
        self.assertIn(videos[1]["video_name"], by_name)
        self.assertIn(
            "high_max_annotation_time",
            by_name[videos[1]["video_name"]]["selection_reason"],
        )

    def test_run_writes_report_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "annotations.json"
            output_dir = root / "output"
            input_path.write_text(
                json.dumps([sample_record("video-1")], ensure_ascii=False),
                encoding="utf-8",
            )

            summary = AUDIT.run(input_path, output_dir, sample_size=1, seed=7)

            self.assertEqual(summary["sample_size"], 1)
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "candidate_videos_1.csv").is_file())
            self.assertTrue((output_dir / "candidate_videos_1.jsonl").is_file())
            self.assertIn(
                "只能作为媒体时长下界", (output_dir / "report.md").read_text("utf-8")
            )


if __name__ == "__main__":
    unittest.main()
