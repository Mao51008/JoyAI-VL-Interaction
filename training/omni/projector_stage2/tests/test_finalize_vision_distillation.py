from pathlib import Path

from training.omni.projector_stage2 import finalize_vision_distillation as finalize


def test_cli_maps_repeated_shard_argument(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_finalize(**kwargs):
        captured.update(kwargs)
        return {"written_samples": 1}

    monkeypatch.setattr(finalize, "finalize", fake_finalize)
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_vision_distillation.py", "--source-manifest", "source.jsonl",
            "--shard", "first.jsonl", "--shard", "second.jsonl",
            "--output-manifest", "output.jsonl", "--teacher-model", "JoyAI",
        ],
    )
    finalize.main()
    assert captured["source_manifest"] == Path("source.jsonl")
    assert captured["shards"] == [Path("first.jsonl"), Path("second.jsonl")]
    assert captured["output_manifest"] == Path("output.jsonl")
    assert '"written_samples": 1' in capsys.readouterr().out
