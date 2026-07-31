import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "audit_media_archive.py"
SPEC = importlib.util.spec_from_file_location("audit_media_archive", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def write_tar(path: Path) -> None:
    with tarfile.open(path, "w") as archive:
        for name, content in (
            ("videos_pool/CharadesEgo/ABC_action_1.mp4", b"first-video"),
            ("videos_pool/CharadesEgo/XYZ_action_2.MP4", b"second-video"),
            ("README.txt", b"ignored"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


class AuditMediaArchiveTest(unittest.TestCase):
    def test_canonical_key_ignores_path_case_and_media_extension(self) -> None:
        self.assertEqual(
            AUDIT.canonical_video_key(r"folder\ABC_action_1.MP4"),
            "abc_action_1",
        )

    def test_media_member_recognizes_extensionless_joyai_video(self) -> None:
        self.assertTrue(
            AUDIT.is_media_member_name("videos_pool/CharadesEgo/ABC_action_1")
        )
        self.assertTrue(
            AUDIT.is_media_member_name("videos_pool/CharadesEgo/ABC_action_1.MP4")
        )
        self.assertFalse(AUDIT.is_media_member_name("ABC_action_1"))
        self.assertFalse(AUDIT.is_media_member_name("README.txt"))

    def test_scan_matches_extensionless_joyai_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "extensionless.tar"
            content = b"extensionless-video"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("videos_pool/CharadesEgo/ABC_action_1")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            annotations = {
                "abc_action_1": {
                    "video_name": "ABC_action_1",
                    "records": [{}],
                    "max_annotation_time_s": 1.0,
                }
            }
            matches, summary = AUDIT.scan_archive(
                archive_path,
                annotations,
            )
            self.assertEqual(summary["media_member_count"], 1)
            self.assertEqual(summary["matched_unique_video_names"], 1)
            extracted = AUDIT.extract_selected(
                archive_path,
                matches,
                root / "extracted",
            )
            self.assertEqual(
                extracted["ABC_action_1"].read_bytes(),
                content,
            )

    def test_scan_match_select_and_safe_extract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "sample.tar"
            write_tar(archive_path)
            annotations = {
                "abc_action_1": {
                    "video_name": "ABC_action_1",
                    "records": [{}, {}],
                    "max_annotation_time_s": 2.0,
                }
            }
            matches, summary = AUDIT.scan_archive(archive_path, annotations)
            self.assertEqual(summary["media_member_count"], 2)
            self.assertEqual(summary["matched_unique_video_names"], 1)
            selected = AUDIT.select_matches(matches, 1, seed="fixed")
            extracted = AUDIT.extract_selected(
                archive_path,
                selected,
                root / "extracted",
            )
            output = extracted["ABC_action_1"]
            self.assertEqual(output.read_bytes(), b"first-video")
            self.assertEqual(output.parent, root / "extracted")

    def test_priority_candidate_is_selected_first(self) -> None:
        matches = [
            AUDIT.ArchiveMatch("one", "one.mp4", 1, 1, 1),
            AUDIT.ArchiveMatch("two", "two.mp4", 1, 1, 1),
        ]
        selected = AUDIT.select_matches(
            matches,
            1,
            seed="fixed",
            priority_names={"two"},
        )
        self.assertEqual(selected[0].video_name, "two")
        self.assertEqual(selected[0].selection_reason, "priority_candidate")

    def test_extract_frames_uses_png_for_ffmpeg_8_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(AUDIT, "_tool_available", return_value=True),
                mock.patch.object(
                    AUDIT.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=1,
                        stderr="expected test failure",
                    ),
                ) as run,
            ):
                AUDIT.extract_frames(
                    root / "input.mp4",
                    root / "frames",
                )
            command = run.call_args.args[0]
            self.assertTrue(str(command[-1]).endswith("frame_%06d.png"))
            self.assertNotIn("-q:v", command)

    def test_extract_frames_falls_back_to_first_frame_for_short_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(AUDIT, "_tool_available", return_value=True),
                mock.patch.object(
                    AUDIT.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stderr=""),
                ) as run,
            ):
                status, frames, error = AUDIT.extract_frames(
                    root / "input.mp4",
                    root / "frames",
                )
            self.assertEqual(status, "failed")
            self.assertEqual(frames, [])
            self.assertEqual(error, "no frames extracted")
            self.assertEqual(run.call_count, 2)
            fallback_command = run.call_args_list[1].args[0]
            frame_count_index = fallback_command.index("-frames:v")
            self.assertEqual(
                fallback_command[frame_count_index + 1],
                "1",
            )

    def test_probe_payload_reports_audio_and_video_contract(self) -> None:
        parsed = AUDIT.parse_probe_payload(
            {
                "format": {"duration": "3.25"},
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 640,
                        "height": 360,
                        "avg_frame_rate": "30/1",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "16000",
                        "channels": 1,
                        "duration": "3.20",
                    },
                ],
            }
        )
        self.assertEqual(parsed["duration_s"], 3.25)
        self.assertEqual(parsed["audio_stream_count"], 1)
        self.assertEqual(parsed["audio_sample_rate"], 16000)

    def test_silence_ratio_is_derived_from_ffmpeg_events(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stderr="silence_duration: 1.5\nsilence_duration: 0.5\n",
            stdout="",
        )
        with (
            mock.patch.object(AUDIT, "_tool_available", return_value=True),
            mock.patch.object(AUDIT.subprocess, "run", return_value=completed),
        ):
            status, ratio, error = AUDIT.analyze_silence(
                Path("fake.mp4"),
                4.0,
            )
        self.assertEqual(status, "ok")
        self.assertEqual(ratio, 0.5)
        self.assertEqual(error, "")

    def test_run_works_with_mocked_media_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "sample.tar"
            write_tar(archive_path)
            annotations_path = root / "annotations.json"
            annotations_path.write_text(
                json.dumps(
                    [
                        {
                            "source": "CharadesEgo",
                            "video_name": "ABC_action_1",
                            "question": [{"content": "Q", "time": "1"}],
                            "response": [{"content": "A", "time": "2"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            probe = {
                "probe_status": "ok",
                "probe_error": "",
                "duration_s": 3.0,
                "video_stream_count": 1,
                "video_codec": "h264",
                "width": 640,
                "height": 360,
                "frame_rate": "30/1",
                "audio_stream_count": 1,
                "audio_codec": "aac",
                "audio_sample_rate": 16000,
                "audio_channels": 1,
                "audio_duration_s": 3.0,
            }
            with (
                mock.patch.object(AUDIT, "probe_media", return_value=probe),
                mock.patch.object(AUDIT, "decode_media", return_value=("ok", "")),
                mock.patch.object(
                    AUDIT,
                    "extract_frames",
                    return_value=(
                        "ok",
                        [{"path": "frame.jpg", "timestamp_ms": 0}],
                        "",
                    ),
                ),
            ):
                summary = AUDIT.run(
                    archive_path,
                    annotations_path,
                    root / "output",
                    source="CharadesEgo",
                    sample_size=1,
                    seed="fixed",
                    priority_path=None,
                    decode=True,
                    analyze_audio_silence=False,
                    extract_frame_images=True,
                    frame_fps=1.0,
                    ffprobe="ffprobe",
                    ffmpeg="ffmpeg",
                    expected_size=archive_path.stat().st_size,
                )
            self.assertEqual(summary["media_usable_count"], 1)
            self.assertEqual(summary["audio_usable_count"], 1)
            self.assertTrue((root / "output" / "media_audit.jsonl").is_file())

    def test_wrong_expected_archive_size_stops_before_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "sample.tar"
            write_tar(archive_path)
            with self.assertRaisesRegex(ValueError, "archive size mismatch"):
                AUDIT.run(
                    archive_path,
                    root / "missing.json",
                    root / "output",
                    source="CharadesEgo",
                    sample_size=1,
                    seed="fixed",
                    priority_path=None,
                    decode=False,
                    analyze_audio_silence=False,
                    extract_frame_images=False,
                    frame_fps=1.0,
                    ffprobe="ffprobe",
                    ffmpeg="ffmpeg",
                    expected_size=1,
                )


if __name__ == "__main__":
    unittest.main()
