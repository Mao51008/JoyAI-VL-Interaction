"""Prepare deterministic SpokenWOZ train/dev manifests and audit leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable

from .spokenwoz_turn_converter import convert_spokenwoz_turns


TEXT_SUFFIXES = {".json", ".jsonl"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
AUDIT_FIELDS = (
    "sample_id",
    "dialogue_id",
    "speaker_id",
    "audio_path",
    "text_audio_sha256",
    "provenance",
)


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_text_records(root: Path, split: str) -> list[dict[str, Any]]:
    files = sorted(
        path for path in (root / split).rglob("*") if path.suffix.lower() in TEXT_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"no JSON/JSONL text files under {root / split}")
    records: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            values = json.loads(path.read_text(encoding="utf-8"))
        records.extend(_flatten_records(values, path))
    return records


def _flatten_records(value: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item, _source_file=str(source)) for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        raise ValueError(f"unsupported JSON shape in {source}")
    if "dialogue_id" in value or "sample_id" in value:
        return [dict(value, _source_file=str(source))]
    flattened: list[dict[str, Any]] = []
    for dialogue_id, dialogue in value.items():
        if not isinstance(dialogue, dict):
            continue
        turns = dialogue.get("log") or dialogue.get("turns") or [dialogue]
        if not isinstance(turns, list):
            turns = [turns]
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            flattened.append(
                dict(
                    turn,
                    dialogue_id=str(dialogue_id),
                    turn_id=turn.get("turn_id", turn_index),
                    _source_file=str(source),
                )
            )
    return flattened


def _audio_index(root: Path, split: str) -> dict[str, Path]:
    paths = sorted(
        path for path in (root / split).rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES
    )
    index: dict[str, Path] = {}
    for path in paths:
        key = path.stem
        if key in index:
            raise ValueError(f"ambiguous audio basename {key!r} in {root / split}")
        index[key] = path
    return index


def _first(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if record.get(name) not in (None, ""):
            return record[name]
    return None


def _make_rows(
    audio_root: Path, text_root: Path, split: str, dataset: str, version: str
) -> list[dict[str, Any]]:
    audio = _audio_index(audio_root, split)
    rows: list[dict[str, Any]] = []
    for record in _load_text_records(text_root, split):
        dialogue_id = str(_first(record, "dialogue_id", "dialogue", "id") or "").strip()
        if not dialogue_id:
            raise ValueError(f"record in {record['_source_file']} has no dialogue_id")
        audio_path = record.get("audio_path")
        path = Path(audio_path) if audio_path else audio.get(dialogue_id)
        if path is None:
            raise FileNotFoundError(
                f"no audio for dialogue {dialogue_id!r} in split {split}"
            )
        if not path.is_absolute():
            path = audio_root / split / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"audio path does not exist: {path}")
        turn_id = str(_first(record, "turn_id", "utterance_id") or 0)
        sample_id = str(_first(record, "sample_id") or f"{split}:{dialogue_id}:{turn_id}")
        text = _normalise_text(_first(record, "response", "assistant", "text", "user_text"))
        speaker = _first(record, "speaker_id", "speaker", "channel_id")
        speaker_id = str(speaker).strip() if speaker not in (None, "") else None
        audio_hash = _sha256_bytes(path)
        text_hash = _sha256_text(text)
        source_file = str(Path(record["_source_file"]).resolve())
        provenance = {
            "dataset": dataset,
            "version": version,
            "split": split,
            "source_file": source_file,
        }
        rows.append(
            {
                "sample_id": sample_id,
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "speaker_id": speaker_id,
                "split": split,
                "audio_path": path.relative_to((audio_root / split).resolve()).as_posix(),
                "text": text,
                "text_sha256": text_hash,
                "audio_sha256": audio_hash,
                "text_audio_sha256": _sha256_text(f"{text}\n{audio_hash}"),
                "provenance": provenance,
            }
        )
    rows.sort(key=lambda row: (row["dialogue_id"], row["turn_id"], row["sample_id"]))
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate sample_id in {split}")
    return rows


def _field_key(row: dict[str, Any], field: str) -> Any:
    if field == "provenance":
        provenance = row.get("provenance") or {}
        return (
            provenance.get("dataset"),
            provenance.get("version"),
            provenance.get("source_file"),
            provenance.get("audio_member"),
        )
    return row.get(field)


def audit_manifests(
    train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    leaked = False
    for field in AUDIT_FIELDS:
        train_values = {
            _field_key(row, field) for row in train_rows if _field_key(row, field) not in (None, "")
        }
        dev_values = {
            _field_key(row, field) for row in dev_rows if _field_key(row, field) not in (None, "")
        }
        intersection = train_values & dev_values
        examples = [
            list(value) if isinstance(value, tuple) else value
            for value in sorted(intersection, key=str)[:10]
        ]
        result[field] = {"intersection_count": len(intersection), "examples": examples}
        leaked = leaked or bool(intersection)
    result["leakage"] = leaked
    if leaked:
        fields = [field for field in AUDIT_FIELDS if result[field]["intersection_count"]]
        raise ValueError(f"train/dev leakage detected in: {', '.join(fields)}")
    return result


def _manifest_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _load_official_dialogues(data_json: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(data_json.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SpokenWOZ data.json must be a dialogue mapping")
    return value


def _load_dialogue_ids(path: Path) -> set[str]:
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError(f"SpokenWOZ split list is empty: {path}")
    return values


def _archive_audio_index(archive: Path) -> dict[str, tuple[str, tarfile.TarInfo]]:
    if not archive.is_file():
        raise FileNotFoundError(f"audio archive does not exist: {archive}")
    index: dict[str, tuple[str, tarfile.TarInfo]] = {}
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle:
            if not member.isfile() or Path(member.name).suffix.lower() not in AUDIO_SUFFIXES:
                continue
            stem = Path(member.name).stem
            if stem in index:
                raise ValueError(f"ambiguous archive audio basename {stem!r}")
            index[stem] = (member.name, member)
    return index


def _official_rows(
    data_json: Path,
    dev_list: Path,
    audio_archive: Path,
    dataset: str,
    version: str,
) -> dict[str, list[dict[str, Any]]]:
    dialogues = _load_official_dialogues(data_json)
    dev_ids = _load_dialogue_ids(dev_list)
    unknown_dev = dev_ids - dialogues.keys()
    if unknown_dev:
        raise ValueError(f"dev split contains unknown dialogue IDs: {sorted(unknown_dev)[:5]}")
    audio_index = _archive_audio_index(audio_archive)
    rows = {"train": [], "dev": []}
    for dialogue_id in sorted(dialogues):
        dialogue = dialogues[dialogue_id]
        turns = dialogue.get("log") if isinstance(dialogue, dict) else None
        if not isinstance(turns, list):
            raise ValueError(f"dialogue {dialogue_id!r} has no list log")
        split = "dev" if dialogue_id in dev_ids else "train"
        audio = audio_index.get(dialogue_id)
        if audio is None:
            raise FileNotFoundError(f"archive has no audio for dialogue {dialogue_id!r}")
        member_name, member = audio
        history: list[dict[str, Any]] = []
        pair_count = 0
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ValueError(f"dialogue {dialogue_id!r} turn {turn_index} is not an object")
            tag = str(turn.get("tag", turn.get("speaker", ""))).lower()
            text = _normalise_text(turn.get("text"))
            if tag in {"user", "human"}:
                if turn_index + 1 >= len(turns):
                    raise ValueError(f"dialogue {dialogue_id!r} user turn has no adjacent assistant turn")
                response_turn = turns[turn_index + 1]
                response_tag = str(response_turn.get("tag", response_turn.get("speaker", ""))).lower()
                if response_tag not in {"assistant", "system"}:
                    raise ValueError(
                        f"dialogue {dialogue_id!r} user turn {turn_index} is not followed by assistant"
                    )
                response = _normalise_text(response_turn.get("text"))
                if not response:
                    raise ValueError(f"dialogue {dialogue_id!r} assistant turn is empty")
                sample_id = f"{split}:{dialogue_id}:{turn_index}"
                combined_text = f"{text}\n{response}".strip()
                provenance = {
                    "dataset": dataset,
                    "version": version,
                    "split": split,
                    "source_file": str(data_json.resolve()),
                    "audio_archive": str(audio_archive.resolve()),
                    "audio_member": member_name,
                }
                rows[split].append(
                    {
                        "sample_id": sample_id,
                        "dialogue_id": dialogue_id,
                        "turn_id": str(turn_index),
                        "speaker_id": None,
                        "split": split,
                        "audio_path": member_name,
                        "audio_source": "tar.gz",
                        "audio_size_bytes": member.size,
                        "audio_sha256": None,
                        "user_text": text,
                        "assistant_response": response,
                        "text": response,
                        "text_sha256": _sha256_text(combined_text),
                        "text_audio_sha256": None,
                        "dialogue_history": list(history),
                        "provenance": provenance,
                    }
                )
                pair_count += 1
            history.append({"role": tag or "unknown", "text": text})
        if pair_count == 0:
            raise ValueError(f"dialogue {dialogue_id!r} has no user/assistant pair")
    for split_rows in rows.values():
        split_rows.sort(key=lambda row: (row["dialogue_id"], row["turn_id"], row["sample_id"]))
    return rows


def prepare_official_manifests(
    data_json: Path,
    dev_list: Path,
    audio_archive: Path,
    output_root: Path,
    dataset: str = "SpokenWOZ",
    version: str = "official-main",
) -> dict[str, Any]:
    """Build formal manifests from single-channel user-turn audio clips."""
    return convert_spokenwoz_turns(
        data_json,
        dev_list,
        audio_archive,
        output_root,
        dataset=dataset,
        version=version,
    )


def prepare_manifests(
    audio_root: Path,
    text_root: Path,
    output_root: Path,
    dataset: str = "SpokenWOZ",
    version: str = "official-main",
) -> dict[str, Any]:
    audio_root = audio_root.resolve()
    text_root = text_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_root}")
    for root, label in ((audio_root, "audio"), (text_root, "text")):
        if not root.is_dir():
            raise FileNotFoundError(f"{label} root does not exist: {root}")
        for split in ("train", "dev"):
            if not (root / split).is_dir():
                raise FileNotFoundError(f"missing {label}/{split} directory under {root}")
    rows = {
        split: _make_rows(audio_root, text_root, split, dataset, version)
        for split in ("train", "dev")
    }
    audit = audit_manifests(rows["train"], rows["dev"])
    output_root.mkdir(parents=True)
    manifest_hashes: dict[str, str] = {}
    for split, split_rows in rows.items():
        data = _manifest_bytes(split_rows)
        (output_root / f"{split}.jsonl").write_bytes(data)
        manifest_hashes[split] = hashlib.sha256(data).hexdigest()
    provenance = {
        "schema_version": 1,
        "dataset": dataset,
        "version": version,
        "audio_root": str(audio_root),
        "text_root": str(text_root),
        "splits": {
            split: {
                "samples": len(split_rows),
                "dialogues": len({row["dialogue_id"] for row in split_rows}),
                "speakers": len(
                    {row["speaker_id"] for row in split_rows if row["speaker_id"]}
                ),
                "manifest": f"{split}.jsonl",
                "manifest_sha256": manifest_hashes[split],
            }
            for split, split_rows in rows.items()
        },
        "leakage_audit": audit,
    }
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "leakage_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and audit deterministic SpokenWOZ train/dev manifests."
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=False,
        help="Root containing train/ and dev/ audio directories",
    )
    parser.add_argument(
        "--text-root",
        type=Path,
        required=False,
        help="Root containing train/ and dev/ JSON/JSONL directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory for manifests and audit outputs",
    )
    parser.add_argument("--data-json", type=Path, help="Official SpokenWOZ data.json")
    parser.add_argument("--dev-list", type=Path, help="Official SpokenWOZ valListFile.json")
    parser.add_argument("--audio-archive", type=Path, help="Official audio_5700_train_dev.tar.gz")
    parser.add_argument("--dataset", default="SpokenWOZ")
    parser.add_argument("--version", default="official-main")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    official = (args.data_json, args.dev_list, args.audio_archive)
    if any(value is not None for value in official):
        if any(value is None for value in official):
            raise ValueError("--data-json, --dev-list and --audio-archive must be supplied together")
        provenance = prepare_official_manifests(
            args.data_json,
            args.dev_list,
            args.audio_archive,
            args.output_root,
            args.dataset,
            args.version,
        )
    else:
        if args.audio_root is None or args.text_root is None:
            raise ValueError("legacy mode requires --audio-root and --text-root")
        provenance = prepare_manifests(
            args.audio_root, args.text_root, args.output_root, args.dataset, args.version
        )
    print(json.dumps(provenance["splits"], ensure_ascii=False, sort_keys=True))
    print("leakage: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
