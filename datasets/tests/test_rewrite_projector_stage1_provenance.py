import json

from datasets.rewrite_projector_stage1_provenance import rewrite


def test_rewrite_updates_only_librispeech_version(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"sample_id": "x", "provenance": {"dataset": "LibriSpeech", "version": "dev-clean"}, "metadata": {"dataset_version": "train-clean-100"}}) + "\n", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    result = rewrite(source, output, version="train-clean-100")
    assert result["samples"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["provenance"]["version"] == "train-clean-100"


def test_rewrite_rejects_source_version_mismatch(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "sample_id": "x", "provenance": {"dataset": "LibriSpeech", "version": "dev-clean"},
        "metadata": {"dataset_version": "dev-clean"},
    }) + "\n", encoding="utf-8")

    try:
        rewrite(source, tmp_path / "output.jsonl", version="train-clean-100")
    except ValueError as exc:
        assert "source/version mismatch" in str(exc)
    else:
        raise AssertionError("expected source/version mismatch")
