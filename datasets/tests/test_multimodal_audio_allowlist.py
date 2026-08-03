import csv
from pathlib import Path

from datasets.build_multimodal_audio_allowlist import build


def test_build_marks_known_silent_sample(tmp_path: Path) -> None:
    report = tmp_path / "report.csv"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "-BCoVGruruc.mp4").write_bytes(b"fixture")
    (media_dir / "good.mp4").write_bytes(b"fixture2")
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "audio", "rate", "channels", "audio_s"])
        writer.writeheader()
        writer.writerow({"file": "-BCoVGruruc", "audio": "opus", "rate": "48000", "channels": "2", "audio_s": "1"})
        writer.writerow({"file": "good", "audio": "aac", "rate": "44100", "channels": "2", "audio_s": "2"})

    rows = build(report, "Kinetics-400", media_dir)

    assert rows[0]["allowlist_status"] == "exclude_silent"
    assert rows[1]["allowlist_status"] == "confirmed_audio_stream"
