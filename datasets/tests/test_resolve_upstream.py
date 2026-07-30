import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "resolve_upstream.py"
SPEC = importlib.util.spec_from_file_location("resolve_upstream", MODULE_PATH)
assert SPEC and SPEC.loader
RESOLVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESOLVER)


class ResolveUpstreamTest(unittest.TestCase):
    def test_charades_parent_and_action(self) -> None:
        result = RESOLVER.resolve_charades("1G5SNEGO_action_9")
        self.assertEqual(result["parent_video_name"], "1G5SNEGO.mp4")
        self.assertEqual(result["segment_index"], 9)
        self.assertEqual(result["locator_status"], "prepared_clip_name_candidate")
        self.assertEqual(
            result["upstream_path_candidate"],
            "videos_pool/CharadesEgo/1G5SNEGO_action_9.mp4",
        )

    def test_egoprocel_components(self) -> None:
        cmu = RESOLVER.resolve_egoprocel(
            "embodied_datasets_2024_EgoProceL_videos_CMU-MMAC_Pizza_"
            "S35_Pizza_Video_S35_Pizza_6510211-726_action_3.mp4"
        )
        egtea = RESOLVER.resolve_egoprocel(
            "embodied_datasets_2024_EgoProceL_videos_"
            "EGTEA-Gaze+_OP01-R06-GreekSalad_action_9.mp4"
        )
        tents = RESOLVER.resolve_egoprocel(
            "embodied_datasets_2024_EgoProceL_videos_EPIC-Tents_"
            "2ite3tu1u53n42hjfh3886sa86_data_02_02.tent.120617.gopro_action_13.mp4"
        )
        meccano = RESOLVER.resolve_egoprocel(
            "embodied_datasets_2024_EgoProceL_videos_"
            "MECCANO_MECCANO_RGB_Videos_0011_action_28.mp4"
        )

        self.assertIn("CMU-MMAC/Pizza/S35_Pizza_Video", cmu["upstream_path_candidate"])
        self.assertEqual(egtea["lookup_key"], "OP01-R06-GreekSalad")
        self.assertEqual(tents["lookup_key"], "02.tent.120617.gopro")
        self.assertEqual(meccano["lookup_key"], "0011")

    def test_ambiguous_open_o3_split_is_not_treated_as_direct(self) -> None:
        result = RESOLVER.resolve_open_o3("split_3.mp4")
        self.assertEqual(result["locator_status"], "ambiguous_basename")
        self.assertFalse(result["remote_verified"])

    def test_ego4d_uuid_fields_are_extracted(self) -> None:
        result = RESOLVER.resolve_ego4d(
            "ego4d_vqa_56a3b49c-6979-4253-9ff5-2733c3e2f229_"
            "519579d3-4b63-426d-92a2-fb4cceb9ebe7_"
            "6f464a86-1bfd-494f-af22-8f54e9eec7c9_3.mp4"
        )
        self.assertEqual(result["lookup_key"], "56a3b49c-6979-4253-9ff5-2733c3e2f229")
        self.assertEqual(
            result["locator_status"], "controlled_access_metadata_required"
        )

    def test_shot2story_parent_and_shot(self) -> None:
        result = RESOLVER.resolve_shot2story("MPEqtTGDNoA.1.mp4")
        self.assertEqual(result["parent_video_name"], "MPEqtTGDNoA.mp4")
        self.assertEqual(result["segment_index"], 1)

    def test_run_keeps_remote_verification_false(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "candidates.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "source": "CharadesEgo",
                        "video_name": "1G5SNEGO_action_9",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = RESOLVER.run(input_path, root / "output")
            self.assertEqual(summary["candidates"], 1)
            self.assertEqual(summary["remote_verified"], 0)
            self.assertTrue((root / "output" / "upstream_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
