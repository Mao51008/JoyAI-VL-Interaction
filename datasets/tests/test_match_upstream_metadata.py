import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "match_upstream_metadata.py"
SPEC = importlib.util.spec_from_file_location("match_upstream_metadata", MODULE_PATH)
assert SPEC and SPEC.loader
MATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATCHER)


def joy_record(question: str, answer: str) -> dict:
    return {
        "question": [{"content": question, "time": "1"}],
        "response": [{"content": answer, "time": "1"}],
    }


class MatchUpstreamMetadataTest(unittest.TestCase):
    def test_answer_variants_support_gui_multiple_choice(self) -> None:
        self.assertEqual(
            MATCHER.answer_variants("[[C]] Deletes the email"),
            {"[[C]] Deletes the email", "C", "Deletes the email"},
        )

    def test_gui_pair_collection_finds_nested_qa(self) -> None:
        value = {
            "MCQA": {
                "Question": "What happens?",
                "Correct Answer": "[[B]] It opens",
            }
        }
        pairs = MATCHER.collect_gui_pairs(value)
        self.assertIn(("What happens?", "B"), pairs)
        self.assertIn(("What happens?", "It opens"), pairs)

    def test_same_basename_is_preferred_over_generic_duplicate(self) -> None:
        matches = [
            {"video_path": "software/2.mp4"},
            {"video_path": "android/1.mp4"},
            {"video_path": "website/1.mp4"},
        ]
        preferred = MATCHER.prefer_matching_basename(matches, "1.mp4")
        self.assertEqual(
            {row["video_path"] for row in preferred},
            {"android/1.mp4", "website/1.mp4"},
        )

    def test_multiple_paths_confirm_name_collision(self) -> None:
        index = {
            ("Question", "Answer"): [
                {
                    "video_path": "android/1.mp4",
                    "metadata_file": "android.jsonl",
                    "metadata_id": "",
                },
                {
                    "video_path": "website/1.mp4",
                    "metadata_file": "website.jsonl",
                    "metadata_id": "",
                },
            ]
        }
        result = MATCHER.match_name_records(
            "1.mp4",
            [joy_record("Question", "Answer")],
            index,
        )
        self.assertEqual(result["metadata_match_status"], "confirmed_name_collision")
        self.assertEqual(result["matched_path_count"], 2)

    def test_one_path_is_unique_even_with_duplicate_metadata_rows(self) -> None:
        metadata = {
            "video_path": "STAR/ABC.mp4",
            "metadata_file": "data.json",
            "metadata_id": "one",
        }
        index = {("Question", "Answer"): [metadata, dict(metadata)]}
        result = MATCHER.match_name_records(
            "ABC.mp4",
            [joy_record("Question", "Answer")],
            index,
        )
        self.assertEqual(result["metadata_match_status"], "matched_unique_path")
        self.assertEqual(result["matched_path_count"], 1)

    def test_charades_official_action_join_is_marked_hypothetical(self) -> None:
        candidate = {
            "video_name": "ABC12EGO_action_0",
            "segment_index": 0,
            "max_annotation_time_s": 5,
        }
        index = {"ABC12EGO": {"actions": "c001 2.00 7.00;c002 8.00 9.00"}}
        result = MATCHER.match_charades_candidate(candidate, [{}], index)
        self.assertEqual(
            result["metadata_match_status"],
            "parent_metadata_found_action_index_plausible_unverified",
        )
        self.assertEqual(
            result["matched_upstream_paths"], ["CharadesEgo_v1_480/ABC12EGO.mp4"]
        )
        self.assertFalse(result["clip_derivation_verified"])
        self.assertEqual(result["hypothetical_action_class"], "c001")
        self.assertEqual(result["hypothetical_action_start_s"], 2)
        self.assertEqual(result["hypothetical_action_end_s"], 7)

    def test_charades_invalid_action_boundary_is_not_silently_accepted(self) -> None:
        candidate = {
            "video_name": "ABC12EGO_action_0",
            "segment_index": 0,
            "max_annotation_time_s": 5,
        }
        index = {"ABC12EGO": {"actions": "c001 8.40 8.17"}}
        result = MATCHER.match_charades_candidate(candidate, [{}], index)
        self.assertEqual(
            result["metadata_match_status"],
            "parent_metadata_found_action_boundary_invalid_unverified",
        )


if __name__ == "__main__":
    unittest.main()
