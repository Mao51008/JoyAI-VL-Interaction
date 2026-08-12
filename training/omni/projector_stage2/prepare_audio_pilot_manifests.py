"""Convert the audio-understanding pilot sources into formal Stage2 manifests.

The source manifests remain untouched.  This command writes new train/dev JSONL files
only after every selected audio file has been found and fingerprinted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration_ms(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to prepare audio manifests") from exc
    info = sf.info(path)
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"invalid or empty audio file: {path}")
    return round(1000.0 * info.frames / info.samplerate, 3)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _normalized_audio_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _clotho_audio_paths(audio_root: Path) -> dict[str, Path]:
    paths = {path.name: path for path in audio_root.glob("*.wav")}
    if not paths:
        raise FileNotFoundError(f"Clotho audio directory is empty: {audio_root}")
    normalized: dict[str, Path | None] = {}
    for name, path in paths.items():
        key = _normalized_audio_name(name)
        normalized[key] = path if key not in normalized else None
    resolved = dict(paths)
    for key, path in normalized.items():
        if path is not None:
            resolved.setdefault(key, path)
    return resolved


def _progress(rows: Iterable[dict[str, Any]], description: str, no_progress: bool) -> Iterable[dict[str, Any]]:
    try:
        from tqdm import tqdm
    except ImportError:
        return rows
    return tqdm(rows, desc=description, unit="sample", disable=no_progress)


def _formal_row(
    source: dict[str, Any],
    *,
    audio_path: Path,
    split: str,
    training_task: str,
    dialogue_history: list[dict[str, str]],
    user_text: str,
    source_manifest: Path,
) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio for {source['sample_id']} does not exist: {audio_path}")
    clip_sha256 = _sha256(audio_path)
    provenance = dict(source.get("provenance", {}))
    provenance.update({"source_manifest": str(source_manifest.resolve())})
    return {
        "sample_id": str(source["sample_id"]),
        "dialogue_id": str(source["sample_id"]),
        "turn_id": "0",
        "split": split,
        "audio_path": str(audio_path.resolve()),
        "clip_duration_ms": _duration_ms(audio_path),
        "clip_sha256": clip_sha256,
        "source_audio_sha256": clip_sha256,
        "user_text": user_text,
        "assistant_response": str(source["assistant_response"]).strip(),
        "dialogue_history": dialogue_history,
        "training_task": training_task,
        "provenance": provenance,
    }


def _pilot_rows(
    source_rows: Iterable[dict[str, Any]],
    source_manifest: Path,
    *,
    clotho_audio_root: Path,
    clotho_audio_paths: dict[str, Path],
    voice_audio_root: Path,
    no_progress: bool,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for source in _progress(source_rows, "prepare pilot audio", no_progress):
        task = str(source.get("task", ""))
        split = str(source.get("split", ""))
        if split not in {"train", "dev"}:
            continue
        if task == "environment_audio_qa":
            question = str(source.get("question", "")).strip()
            if not question:
                raise ValueError(f"Clotho question is empty: {source['sample_id']}")
            source_name = str(source["source_audio_path"])
            audio_path = clotho_audio_paths.get(source_name)
            if audio_path is None:
                audio_path = clotho_audio_paths.get(_normalized_audio_name(source_name))
            if audio_path is None:
                raise FileNotFoundError(f"Clotho audio is missing or ambiguous: {source_name}")
            converted.append(
                _formal_row(
                    source,
                    audio_path=audio_path,
                    split=split,
                    training_task="dialogue_response",
                    dialogue_history=[{"role": "user", "text": question}],
                    user_text="",
                    source_manifest=source_manifest,
                )
            )
        elif task == "audio_dialogue_response":
            converted.append(
                _formal_row(
                    source,
                    audio_path=voice_audio_root / str(source["source_audio_path"]),
                    split=split,
                    training_task="dialogue_response",
                    dialogue_history=[],
                    user_text="",
                    source_manifest=source_manifest,
                )
            )
        else:
            raise ValueError(f"unsupported pilot task {task!r}: {source['sample_id']}")
    return converted


def _asr_rows(
    source_manifest: Path, split: str, count: int, no_progress: bool
) -> list[dict[str, Any]]:
    source_rows = sorted(
        _read_jsonl(source_manifest),
        key=lambda row: hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest(),
    )
    if count > len(source_rows):
        raise ValueError(
            f"ASR replay needs {count} {split} samples but only {len(source_rows)} are available"
        )
    converted: list[dict[str, Any]] = []
    for source in _progress(source_rows[:count], "prepare ASR replay", no_progress):
        audio = source.get("audio")
        metadata = source.get("metadata", {})
        if not isinstance(audio, list) or len(audio) != 1 or not isinstance(metadata, dict):
            raise ValueError(f"unsupported LibriSpeech row: {source.get('sample_id')}")
        target = str(metadata.get("assistant_target_text", "")).strip()
        if not target:
            raise ValueError(f"LibriSpeech target is empty: {source.get('sample_id')}")
        synthetic = {
            "sample_id": str(source["sample_id"]),
            "assistant_response": target,
            "provenance": dict(source.get("provenance", {})),
        }
        converted.append(
            _formal_row(
                synthetic,
                audio_path=Path(str(audio[0]["path"])),
                split=split,
                training_task="asr_transcription",
                dialogue_history=[],
                user_text=target,
                source_manifest=source_manifest,
            )
        )
    return converted


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_manifests(
    *,
    clotho_manifest: Path | None,
    voice_manifest: Path | None,
    asr_train_manifest: Path,
    asr_dev_manifest: Path,
    clotho_audio_root: Path,
    voice_audio_root: Path,
    output_dir: Path,
    asr_replay_ratio: float = 0.4,
    no_progress: bool = False,
) -> dict[str, int]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    if not 0 < asr_replay_ratio < 1:
        raise ValueError("asr_replay_ratio must be in (0, 1)")
    if clotho_manifest is None and voice_manifest is None:
        raise ValueError("at least one pilot manifest is required")
    clotho_paths = _clotho_audio_paths(clotho_audio_root) if clotho_manifest is not None else {}
    clotho_rows = [] if clotho_manifest is None else _pilot_rows(
        _read_jsonl(clotho_manifest), clotho_manifest, clotho_audio_root=clotho_audio_root,
        clotho_audio_paths=clotho_paths,
        voice_audio_root=voice_audio_root, no_progress=no_progress,
    )
    voice_rows = [] if voice_manifest is None else _pilot_rows(
        _read_jsonl(voice_manifest), voice_manifest, clotho_audio_root=clotho_audio_root,
        clotho_audio_paths=clotho_paths,
        voice_audio_root=voice_audio_root, no_progress=no_progress,
    )
    pilot_rows = [*clotho_rows, *voice_rows]
    train_rows = [row for row in pilot_rows if row["split"] == "train"]
    dev_rows = [row for row in pilot_rows if row["split"] == "dev"]
    replay_scale = asr_replay_ratio / (1.0 - asr_replay_ratio)
    train_rows.extend(
        _asr_rows(asr_train_manifest, "train", round(len(train_rows) * replay_scale), no_progress)
    )
    dev_rows.extend(
        _asr_rows(asr_dev_manifest, "dev", round(len(dev_rows) * replay_scale), no_progress)
    )
    sample_ids = [str(row["sample_id"]) for row in [*train_rows, *dev_rows]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample_id across formal audio manifests")
    output_dir.mkdir(parents=True)
    _write(output_dir / "train.jsonl", train_rows)
    _write(output_dir / "dev.jsonl", dev_rows)
    return {"train": len(train_rows), "dev": len(dev_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clotho-manifest", type=Path)
    parser.add_argument("--voice-manifest", type=Path)
    parser.add_argument("--asr-train-manifest", type=Path, required=True)
    parser.add_argument("--asr-dev-manifest", type=Path, required=True)
    parser.add_argument("--clotho-audio-root", type=Path, required=True)
    parser.add_argument("--voice-audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asr-replay-ratio", type=float, default=0.4)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_manifests(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
