"""Match JoyAI annotations to cached official upstream metadata.

Only JSON/JSONL/CSV metadata is read.  The script never opens or downloads media.
Matches prefer official paths whose basename equals JoyAI's flattened name,
then use exact normalized question-and-answer pairs for disambiguation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

CHARADES_CSV_MEMBERS = (
    "CharadesEgo/CharadesEgo_v1_train_only1st.csv",
    "CharadesEgo/CharadesEgo_v1_test_only1st.csv",
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def answer_variants(value: Any) -> set[str]:
    """Return useful equivalents for GUI multiple-choice answer formats."""
    normalized = normalize_text(value)
    variants = {normalized}
    bracketed = re.match(r"^\[\[([A-Z])\]\]\s*(.*)$", normalized)
    if bracketed:
        variants.add(bracketed.group(1))
        variants.add(bracketed.group(2).strip())
    labelled = re.match(r"^([A-Z])[\).:]?\s+(.+)$", normalized)
    if labelled:
        variants.add(labelled.group(1))
        variants.add(labelled.group(2).strip())
    return {variant for variant in variants if variant}


def collect_gui_pairs(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(value, dict):
        question = value.get("Question")
        if isinstance(question, str):
            for field in ("Answer", "Correct Answer", "answer", "correct_answer"):
                answer = value.get(field)
                if isinstance(answer, str):
                    pairs.extend(
                        (question, variant) for variant in answer_variants(answer)
                    )
        for child in value.values():
            pairs.extend(collect_gui_pairs(child))
    elif isinstance(value, list):
        for child in value:
            pairs.extend(collect_gui_pairs(child))
    return pairs


def prefer_matching_basename(
    matches: list[dict[str, Any]],
    video_name: str,
) -> list[dict[str, Any]]:
    same_basename = [
        row
        for row in matches
        if os.path.basename(str(row.get("video_path", ""))) == video_name
    ]
    return same_basename or matches


def load_open_o3_index(
    metadata_dir: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(metadata_dir.glob("*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            key = (
                normalize_text(record.get("question", "")),
                normalize_text(record.get("answer", "")),
            )
            index[key].append(
                {
                    "video_path": str(record.get("video_path", "")),
                    "metadata_file": path.name,
                    "metadata_id": str(record.get("id", "")),
                }
            )
    return index


def load_gui_world_index(
    metadata_dir: Path,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(metadata_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for question, answer in collect_gui_pairs(record):
                    key = (normalize_text(question), normalize_text(answer))
                    index[key].append(
                        {
                            "video_path": str(record.get("video_path", "")),
                            "metadata_file": path.name,
                            "metadata_id": "",
                        }
                    )
    return index


def load_charades_index(zip_path: Path) -> dict[str, dict[str, str]]:
    """Load first-person parent-video rows from the official annotation archive."""
    index: dict[str, dict[str, str]] = {}
    with ZipFile(zip_path) as archive:
        for member in CHARADES_CSV_MEMBERS:
            with archive.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                for row in reader:
                    index[row["id"]] = row
    return index


def parse_charades_action(value: str) -> tuple[str, float, float]:
    action_class, start, end = value.split()
    return action_class, float(start), float(end)


def match_charades_candidate(
    candidate: dict[str, Any],
    joy_records: list[dict[str, Any]],
    index: dict[str, dict[str, str]],
) -> dict[str, Any]:
    video_name = str(candidate.get("video_name", ""))
    name_match = re.fullmatch(r"(?P<parent>.+)_action_(?P<action>\d+)", video_name)
    parent_id = name_match.group("parent") if name_match else video_name
    parent = index.get(parent_id)
    common = {
        "joy_annotation_records_checked": len(joy_records),
        "exact_qa_matched_records": 0,
        "unique_path_annotation_records": 0,
        "ambiguous_path_annotation_records": 0,
        "matched_metadata_files": ["CharadesEgo.zip"] if parent else [],
        "matched_metadata_id_count": 1 if parent else 0,
        "clip_derivation_verified": False,
        "hypothetical_action_start_s": "",
        "hypothetical_action_end_s": "",
        "hypothetical_action_class": "",
    }
    if parent is None:
        return {
            **common,
            "metadata_match_status": "charades_parent_not_found",
            "metadata_verified": False,
            "matched_path_count": 0,
            "matched_upstream_paths": [],
        }

    upstream_path = f"CharadesEgo_v1_480/{parent_id}.mp4"
    segment_index = candidate.get("segment_index")
    if segment_index in ("", None):
        return {
            **common,
            "metadata_match_status": "parent_metadata_found_prepared_clip_unverified",
            "metadata_verified": True,
            "matched_path_count": 1,
            "matched_upstream_paths": [upstream_path],
        }

    actions = [value for value in parent.get("actions", "").split(";") if value]
    try:
        action = actions[int(segment_index)]
    except (ValueError, IndexError):
        return {
            **common,
            "metadata_match_status": "charades_action_index_out_of_range",
            "metadata_verified": True,
            "matched_path_count": 1,
            "matched_upstream_paths": [upstream_path],
        }

    try:
        action_class, start, end = parse_charades_action(action)
    except (ValueError, TypeError):
        return {
            **common,
            "metadata_match_status": "charades_action_format_invalid",
            "metadata_verified": True,
            "matched_path_count": 1,
            "matched_upstream_paths": [upstream_path],
        }

    duration = end - start
    status = (
        "parent_metadata_found_action_boundary_invalid_unverified"
        if duration < 0
        else "parent_metadata_found_action_index_plausible_unverified"
    )
    return {
        **common,
        "metadata_match_status": status,
        "metadata_verified": True,
        "matched_path_count": 1,
        "matched_upstream_paths": [upstream_path],
        "hypothetical_action_start_s": start,
        "hypothetical_action_end_s": end,
        "hypothetical_action_class": action_class,
    }


def _first_content(record: dict[str, Any], branch: str) -> str:
    nodes = record.get(branch, [])
    if isinstance(nodes, list) and nodes and isinstance(nodes[0], dict):
        return normalize_text(nodes[0].get("content", ""))
    return ""


def match_name_records(
    video_name: str,
    joy_records: list[dict[str, Any]],
    index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    paths: set[str] = set()
    metadata_files: set[str] = set()
    metadata_ids: set[str] = set()
    matched_records = 0
    unique_path_records = 0
    ambiguous_path_records = 0

    for record in joy_records:
        key = (_first_content(record, "question"), _first_content(record, "response"))
        matches = prefer_matching_basename(index.get(key, []), video_name)
        row_paths = {row["video_path"] for row in matches if row["video_path"]}
        if matches:
            matched_records += 1
        if len(row_paths) == 1:
            unique_path_records += 1
        elif len(row_paths) > 1:
            ambiguous_path_records += 1
        paths.update(row_paths)
        metadata_files.update(row["metadata_file"] for row in matches)
        metadata_ids.update(row["metadata_id"] for row in matches if row["metadata_id"])

    if len(paths) == 1:
        status = "matched_unique_path"
    elif len(paths) > 1:
        status = "confirmed_name_collision"
    else:
        status = "not_found_exact_qa"
    return {
        "metadata_match_status": status,
        "metadata_verified": bool(paths),
        "joy_annotation_records_checked": len(joy_records),
        "exact_qa_matched_records": matched_records,
        "unique_path_annotation_records": unique_path_records,
        "ambiguous_path_annotation_records": ambiguous_path_records,
        "matched_path_count": len(paths),
        "matched_upstream_paths": sorted(paths),
        "matched_metadata_files": sorted(metadata_files),
        "matched_metadata_id_count": len(metadata_ids),
        "clip_derivation_verified": "",
        "hypothetical_action_start_s": "",
        "hypothetical_action_end_s": "",
        "hypothetical_action_class": "",
    }


def build_full_source_stats(
    joy_records: list[dict[str, Any]],
    source: str,
    index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    paths_by_name: dict[str, set[str]] = defaultdict(set)
    source_records = 0
    matched_records = 0
    all_paths: set[str] = set()
    for record in joy_records:
        if record.get("source") != source:
            continue
        source_records += 1
        name = str(record.get("video_name", ""))
        key = (_first_content(record, "question"), _first_content(record, "response"))
        matches = prefer_matching_basename(index.get(key, []), name)
        if matches:
            matched_records += 1
        paths = {row["video_path"] for row in matches if row["video_path"]}
        paths_by_name[name].update(paths)
        all_paths.update(paths)

    collisions = {
        name: len(paths) for name, paths in paths_by_name.items() if len(paths) > 1
    }
    return {
        "joy_records": source_records,
        "exact_qa_matched_records": matched_records,
        "exact_qa_match_rate": round(matched_records / max(source_records, 1), 6),
        "joy_unique_flat_names": len(paths_by_name),
        "flat_names_with_multiple_official_paths": len(collisions),
        "official_paths_recovered": len(all_paths),
        "largest_name_collisions": dict(
            sorted(collisions.items(), key=lambda item: (-item[1], item[0]))[:50]
        ),
    }


def build_charades_stats(
    joy_records: list[dict[str, Any]],
    index: dict[str, dict[str, str]],
) -> dict[str, Any]:
    clip_names: set[str] = set()
    parent_ids: set[str] = set()
    matched_parents: set[str] = set()
    valid_actions: set[str] = set()
    invalid_boundaries: set[str] = set()
    source_records = 0

    for record in joy_records:
        if record.get("source") != "CharadesEgo":
            continue
        source_records += 1
        video_name = str(record.get("video_name", ""))
        clip_names.add(video_name)
        match = re.fullmatch(r"(.+)_action_(\d+)", video_name)
        parent_id = match.group(1) if match else video_name
        parent_ids.add(parent_id)
        parent = index.get(parent_id)
        if parent is None:
            continue
        matched_parents.add(parent_id)
        if match is None:
            continue
        action_index = int(match.group(2))
        actions = [value for value in parent.get("actions", "").split(";") if value]
        if action_index >= len(actions):
            continue
        try:
            _action_class, start, end = parse_charades_action(actions[action_index])
        except (ValueError, TypeError):
            continue
        valid_actions.add(video_name)
        if end < start:
            invalid_boundaries.add(video_name)

    return {
        "joy_records": source_records,
        "joy_unique_clip_names": len(clip_names),
        "joy_unique_parent_ids": len(parent_ids),
        "parent_ids_found_in_official_metadata": len(matched_parents),
        "hypothetical_valid_action_indices": len(valid_actions),
        "official_invalid_boundaries_under_hypothesis": len(invalid_boundaries),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_row = dict(row)
        for field in ("matched_upstream_paths", "matched_metadata_files"):
            csv_row[field] = json.dumps(csv_row[field], ensure_ascii=False)
        csv_rows.append(csv_row)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_report(summary: dict[str, Any]) -> str:
    candidate_counts = summary["candidate_match_status"]
    charades_stats = summary["full_source_stats"]["CharadesEgo"]
    open_stats = summary["full_source_stats"]["Open-o3-Video"]
    gui_stats = summary["full_source_stats"]["gui_world"]
    return "\n".join(
        [
            "# 官方元数据连接审计",
            "",
            "本阶段只使用缓存的官方 JSON/JSONL/CSV，没有下载或读取视频内容。",
            "",
            "## 100 条候选",
            "",
            *[
                f"- `{status}`：{count}"
                for status, count in sorted(candidate_counts.items())
            ],
            "",
            "## 全分片键碰撞",
            "",
            (
                f"- CharadesEgo：{charades_stats['joy_unique_clip_names']} 个 JoyAI 切片名称"
                f"对应 {charades_stats['joy_unique_parent_ids']} 个父视频；官方元数据找到"
                f" {charades_stats['parent_ids_found_in_official_metadata']} 个父视频。"
            ),
            (
                f"- Open-o3：{open_stats['joy_unique_flat_names']} 个 JoyAI 扁平名称中，"
                f"{open_stats['flat_names_with_multiple_official_paths']} 个对应多个官方路径；"
                f"恢复 {open_stats['official_paths_recovered']} 条官方路径。"
            ),
            (
                f"- GUI-World：{gui_stats['joy_unique_flat_names']} 个 JoyAI 扁平名称中，"
                f"{gui_stats['flat_names_with_multiple_official_paths']} 个对应多个官方路径；"
                f"恢复 {gui_stats['official_paths_recovered']} 条官方路径。"
            ),
            "",
            (
                "因此不能在所有来源上统一把 `(source, video_name)` 当成上游媒体主键。"
                "Open-o3 和 GUI-World 必须使用完整路径；CharadesEgo 则优先使用与 "
                "JoyAI `video_name` 对应的预切片归档。"
            ),
            "",
            "## 匹配边界",
            "",
            "- 精确匹配率低于 100% 主要因为 JoyAI 同时包含翻译/改写版本。",
            "- `matched_unique_path` 表示官方元数据路径唯一，不表示媒体已下载或可解码。",
            (
                "- CharadesEgo 的 `_action_N` 与官方 CSV action 顺序的关系没有公开证据；"
                "报告中的 hypothetical 字段只记录曾检查过的假设，禁止据此裁剪。"
            ),
            (
                "- JoyAI 的 question/response `time` 是交互监督时间，不是 action 时长，"
                "二者的差值不能用于判断标注质量。"
            ),
            "- CharadesEgo 官方许可仅限非商业研究；商业用途需要另行联系授权。",
            "- `confirmed_name_collision` 必须拆成多条上游媒体记录，禁止按扁平名称下载。",
            "- 未缓存索引的来源保持 `not_checked_no_metadata_index`。",
            "",
        ]
    )


def run(
    candidate_path: Path,
    joy_path: Path,
    metadata_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    joy_records = json.loads(joy_path.read_text(encoding="utf-8"))
    candidate_keys = {(row["source"], row["video_name"]) for row in candidates}
    joy_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in joy_records:
        key = (record.get("source"), record.get("video_name"))
        if key in candidate_keys:
            joy_by_key[key].append(record)

    indexes = {
        "Open-o3-Video": load_open_o3_index(metadata_root / "open_o3"),
        "gui_world": load_gui_world_index(metadata_root / "gui_world"),
    }
    charades_index = load_charades_index(metadata_root / "charades" / "CharadesEgo.zip")
    matched_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        source = candidate["source"]
        if source == "CharadesEgo":
            match = match_charades_candidate(
                candidate,
                joy_by_key[(source, candidate["video_name"])],
                charades_index,
            )
        elif source in indexes:
            match = match_name_records(
                candidate["video_name"],
                joy_by_key[(source, candidate["video_name"])],
                indexes[source],
            )
        else:
            match = {
                "metadata_match_status": "not_checked_no_metadata_index",
                "metadata_verified": False,
                "joy_annotation_records_checked": len(
                    joy_by_key[(source, candidate["video_name"])]
                ),
                "exact_qa_matched_records": 0,
                "unique_path_annotation_records": 0,
                "ambiguous_path_annotation_records": 0,
                "matched_path_count": 0,
                "matched_upstream_paths": [],
                "matched_metadata_files": [],
                "matched_metadata_id_count": 0,
                "clip_derivation_verified": "",
                "hypothetical_action_start_s": "",
                "hypothetical_action_end_s": "",
                "hypothetical_action_class": "",
            }
        matched_candidates.append({**candidate, **match})

    summary = {
        "candidate_count": len(matched_candidates),
        "candidate_match_status": dict(
            Counter(row["metadata_match_status"] for row in matched_candidates)
        ),
        "full_source_stats": {
            "CharadesEgo": build_charades_stats(joy_records, charades_index),
            **{
                source: build_full_source_stats(joy_records, source, index)
                for source, index in indexes.items()
            },
        },
        "important_note": (
            "Media identity is source-specific. Open-o3 and GUI-World require recovered "
            "full paths; CharadesEgo should use the prepared-clip archive matching "
            "JoyAI video_name, not inferred official action boundaries."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output_dir / "candidate_upstream_matched_100.jsonl", matched_candidates
    )
    _write_csv(output_dir / "candidate_upstream_matched_100.csv", matched_candidates)
    (output_dir / "metadata_match_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata_match_report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("joy_annotations", type=Path)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("audit_output"))
    args = parser.parse_args()
    summary = run(
        args.candidates,
        args.joy_annotations,
        args.metadata_root,
        args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
