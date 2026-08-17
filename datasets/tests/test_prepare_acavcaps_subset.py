import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "prepare_acavcaps_subset.py"
SPEC = importlib.util.spec_from_file_location("prepare_acavcaps_subset", MODULE_PATH)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class PrepareAcavcapsSubsetTest(unittest.TestCase):
    def test_parse_sample_key_preserves_decimal_timestamps(self) -> None:
        self.assertEqual(
            PREPARE.parse_sample_key("wn17EHKNLoE_85_9075_95_9075"),
            ("wn17EHKNLoE", 85.9075, 95.9075),
        )

    def test_parse_sample_key_accepts_omitted_end_decimal(self) -> None:
        self.assertEqual(PREPARE.parse_sample_key("woMpG3UGsTs_51_0_61"), ("woMpG3UGsTs", 51.0, 61.0))

    def test_parse_sample_key_accepts_underscore_in_video_id(self) -> None:
        self.assertEqual(PREPARE.parse_sample_key("VtA8fjU_9xy_51_0_61"), ("VtA8fjU_9xy", 51.0, 61.0))

    def test_subset_uses_first_bounded_samples_per_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for category in ("00A", "SM0"):
                with gzip.open(root / f"{category}.jsonl.gz", "wt", encoding="utf-8") as handle:
                    for index in range(2):
                        key = f"abcdefghijk_{index}_0_{index + 1}_0"
                        handle.write(json.dumps({key: {"long": [category]}}) + "\n")
            rows = PREPARE.build_subset(root, 1, ("00A", "SM0"))
            self.assertEqual([row["category"] for row in rows], ["00A", "SM0"])
            self.assertEqual(rows[0]["start_seconds"], 0.0)
            self.assertEqual(rows[1]["annotations"]["long"], ["SM0"])


if __name__ == "__main__":
    unittest.main()
