"""Create an offline upstream-location audit for sampled JoyAI video names.

This script performs structural filename parsing only.  It does not contact
remote services or download media, and every output row remains unverified
until an official metadata index or archive is checked.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

SOURCE_INFO = {
    "CharadesEgo": {
        "repository": "https://prior.allenai.org/projects/charades-ego",
        "access": "public_archive",
        "license": "official_non_commercial_research_license",
    },
    "EgoProceL": {
        "repository": "https://sid2697.github.io/egoprocel/",
        "access": "multi_source_download",
        "license": "review_each_component_dataset_license",
    },
    "Open-o3-Video": {
        "repository": "https://huggingface.co/datasets/marinero4972/Open-o3-Video",
        "access": "public_huggingface_metadata_and_mixed_video_sources",
        "license": "repository_apache_2_0_but_review_each_underlying_video_source",
    },
    "ego4d_vqa": {
        "repository": "https://ego4d-data.org/docs/start-here/",
        "access": "license_acceptance_and_credentials_required",
        "license": "ego4d_license_agreement_required",
    },
    "egoblind": {
        "repository": "https://github.com/doc-doc/EgoBlind",
        "access": "google_drive_research_release",
        "license": "strictly_research_only_and_source_link_required",
    },
    "gui_world": {
        "repository": "https://huggingface.co/datasets/ONE-Lab/GUI-World",
        "access": "public_huggingface",
        "license": "cc_by_4_0",
    },
    "shot2story": {
        "repository": "https://huggingface.co/ByteDance/shot2story",
        "access": "official_release_or_hd_vila_source",
        "license": "annotations_cc_by_nc_sa_4_0_and_underlying_video_terms",
    },
}


def _base_result(source: str, video_name: str) -> dict[str, Any]:
    info = SOURCE_INFO.get(
        source,
        {
            "repository": "",
            "access": "unknown",
            "license": "unknown",
        },
    )
    return {
        "locator_status": "unsupported_source",
        "resolver_confidence": "none",
        "upstream_repository": info["repository"],
        "upstream_access": info["access"],
        "license_note": info["license"],
        "lookup_key": video_name,
        "parent_video_name": "",
        "segment_index": "",
        "upstream_path_candidate": "",
        "remote_verified": False,
        "blocking_reason": "no_resolver_rule",
        "resolver_note": "",
    }


def resolve_charades(video_name: str) -> dict[str, Any]:
    result = _base_result("CharadesEgo", video_name)
    result.update(
        {
            "upstream_repository": (
                "https://huggingface.co/datasets/momo321654/Interaction-videos"
            ),
            "upstream_access": "public_prepared_clip_shards_16_49_gb",
            "license_note": "community_redistribution_requires_independent_review",
            "lookup_key": video_name,
            "upstream_path_candidate": f"videos_pool/CharadesEgo/{video_name}.mp4",
            "remote_verified": False,
            "blocking_reason": "prepared_clip_member_and_audio_verification_required",
        }
    )
    if re.fullmatch(r"[A-Za-z0-9]+EGO", video_name):
        parent = f"{video_name}.mp4"
        result.update(
            {
                "locator_status": "prepared_clip_name_candidate",
                "resolver_confidence": "high",
                "parent_video_name": parent,
                "resolver_note": (
                    "A community reproduction archive exposes a prepared clip whose "
                    "name corresponds to JoyAI video_name."
                ),
            }
        )
        return result
    match = re.fullmatch(
        r"(?P<parent>[A-Za-z0-9]+EGO)_action_(?P<action>\d+)", video_name
    )
    if not match:
        result["blocking_reason"] = "unexpected_charades_name"
        return result
    parent = f"{match.group('parent')}.mp4"
    result.update(
        {
            "locator_status": "prepared_clip_name_candidate",
            "resolver_confidence": "high",
            "parent_video_name": parent,
            "segment_index": int(match.group("action")),
            "resolver_note": (
                "Treat _action_N as a prepared clip identifier. The public JoyAI "
                "materials do not establish that N is the official CSV action order."
            ),
        }
    )
    return result


def _egoprocel_parent(stem: str) -> tuple[str, str, str]:
    if stem.startswith("CMU-MMAC_"):
        rest = stem.removeprefix("CMU-MMAC_")
        match = re.fullmatch(
            r"(?P<task>[^_]+)_(?P<folder>S\d+_[^_]+_Video)_(?P<video>S\d+_.+)",
            rest,
        )
        if match:
            path = (
                f"CMU-MMAC/{match.group('task')}/{match.group('folder')}/"
                f"{match.group('video')}"
            )
            return "CMU-MMAC", match.group("video"), path
    if stem.startswith("EGTEA-Gaze+_"):
        parent = stem.removeprefix("EGTEA-Gaze+_")
        task = parent.rsplit("-", 1)[-1]
        return "EGTEA-Gaze+", parent, f"EGTEA_Gaze+/{task}/{parent}"
    if stem.startswith("EPIC-Tents_"):
        match = re.search(
            r"_data_\d+_(?P<parent>\d+\.tent\.\d+\.gopro)$",
            stem,
        )
        if match:
            parent = match.group("parent")
            return "EPIC-Tents", parent, f"EPIC-Tents/{parent}"
    if stem.startswith("MECCANO_MECCANO_RGB_Videos_"):
        parent = stem.removeprefix("MECCANO_MECCANO_RGB_Videos_")
        return "MECCANO", parent, f"MECCANO/MECCANO_RGB_Videos/{parent}"
    if stem.startswith(("pc_assembly_", "pc_disassembly_")):
        component, parent = stem.rsplit("_", 1)
        return component, parent, f"{component}/{parent}"
    return "", stem, ""


def resolve_egoprocel(video_name: str) -> dict[str, Any]:
    result = _base_result("EgoProceL", video_name)
    prefix = "embodied_datasets_2024_EgoProceL_videos_"
    match = re.fullmatch(r"(?P<stem>.+)_action_(?P<action>\d+)\.mp4", video_name)
    if not match or not match.group("stem").startswith(prefix):
        result["blocking_reason"] = "unexpected_egoprocel_name"
        return result
    encoded_parent = match.group("stem").removeprefix(prefix)
    component, parent, path = _egoprocel_parent(encoded_parent)
    result.update(
        {
            "locator_status": (
                "parent_video_rule_ready" if component else "component_rule_required"
            ),
            "resolver_confidence": "high" if component else "low",
            "lookup_key": parent,
            "parent_video_name": parent,
            "segment_index": int(match.group("action")),
            "upstream_path_candidate": path,
            "blocking_reason": (
                "component_annotation_and_extension_join_required"
                if component
                else "unknown_egoprocel_component"
            ),
            "resolver_note": (
                f"Component={component}; locate source video and cut action by EgoProceL CSV."
                if component
                else "The encoded component name was not recognized."
            ),
        }
    )
    return result


def resolve_open_o3(video_name: str) -> dict[str, Any]:
    result = _base_result("Open-o3-Video", video_name)
    if re.fullmatch(r"split_\d+\.mp4", video_name):
        result.update(
            {
                "locator_status": "ambiguous_basename",
                "resolver_confidence": "low",
                "lookup_key": video_name,
                "parent_video_name": video_name,
                "blocking_reason": "generic_split_name_requires_question_and_video_path_join",
                "resolver_note": (
                    "Many annotations share this generic name; match official STGR JSON "
                    "video_path and question before any media lookup."
                ),
            }
        )
        return result

    youtube_match = re.fullmatch(
        r"(?:ytb_|v_)?(?P<id>[-_A-Za-z0-9]{11})\.mp4", video_name
    )
    result.update(
        {
            "locator_status": "official_metadata_join_required",
            "resolver_confidence": "medium",
            "lookup_key": video_name,
            "parent_video_name": video_name,
            "blocking_reason": "official_stgr_video_path_join_required",
            "resolver_note": (
                "Possible YouTube ID=" + youtube_match.group("id")
                if youtube_match
                else "Use exact basename plus question text against STGR JSON."
            ),
        }
    )
    return result


def resolve_ego4d(video_name: str) -> dict[str, Any]:
    result = _base_result("ego4d_vqa", video_name)
    match = re.fullmatch(
        r"ego4d_vqa_(?P<video>[0-9a-f-]{36})_(?P<clip>[0-9a-f-]{36})_"
        r"(?P<query>[0-9a-f-]{36})_(?P<index>\d+)\.mp4",
        video_name,
    )
    if not match:
        result["blocking_reason"] = "unexpected_ego4d_vqa_name"
        return result
    result.update(
        {
            "locator_status": "controlled_access_metadata_required",
            "resolver_confidence": "medium",
            "lookup_key": match.group("video"),
            "parent_video_name": match.group("video"),
            "segment_index": int(match.group("index")),
            "blocking_reason": "ego4d_license_and_vqa_metadata_join_required",
            "resolver_note": (
                f"video_uid={match.group('video')}; clip_uid={match.group('clip')}; "
                f"query_uid={match.group('query')}"
            ),
        }
    )
    return result


def resolve_egoblind(video_name: str) -> dict[str, Any]:
    result = _base_result("egoblind", video_name)
    if not re.fullmatch(r"\d{5}", video_name):
        result["blocking_reason"] = "unexpected_egoblind_name"
        return result
    result.update(
        {
            "locator_status": "direct_name_candidate",
            "resolver_confidence": "medium",
            "lookup_key": video_name,
            "parent_video_name": f"{video_name}.mp4",
            "upstream_path_candidate": f"videos/{video_name}.mp4",
            "blocking_reason": "official_drive_file_index_verification_required",
            "resolver_note": "Numeric ID is structurally compatible; exact Drive member is unverified.",
        }
    )
    return result


def resolve_gui_world(video_name: str) -> dict[str, Any]:
    result = _base_result("gui_world", video_name)
    result.update(
        {
            "locator_status": "renamed_media_metadata_join_required",
            "resolver_confidence": "low",
            "lookup_key": Path(video_name).stem,
            "parent_video_name": video_name,
            "blocking_reason": "joyai_numeric_mp4_does_not_encode_official_mov_path",
            "resolver_note": (
                "Official data uses category paths such as IOS/0.mov; join by QA/description "
                "metadata, not by numeric basename alone."
            ),
        }
    )
    return result


def resolve_shot2story(video_name: str) -> dict[str, Any]:
    result = _base_result("shot2story", video_name)
    match = re.fullmatch(r"(?P<video>.+)\.(?P<shot>\d+)\.mp4", video_name)
    if not match:
        result["blocking_reason"] = "unexpected_shot2story_name"
        return result
    parent = match.group("video")
    result.update(
        {
            "locator_status": "parent_id_rule_ready",
            "resolver_confidence": "high",
            "lookup_key": parent,
            "parent_video_name": f"{parent}.mp4",
            "segment_index": int(match.group("shot")),
            "blocking_reason": "shot_boundary_and_source_availability_check_required",
            "resolver_note": (
                "Filename encodes parent video ID and shot index; source availability and "
                "non-commercial terms still require review."
            ),
        }
    )
    return result


RESOLVERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "CharadesEgo": resolve_charades,
    "EgoProceL": resolve_egoprocel,
    "Open-o3-Video": resolve_open_o3,
    "ego4d_vqa": resolve_ego4d,
    "egoblind": resolve_egoblind,
    "gui_world": resolve_gui_world,
    "shot2story": resolve_shot2story,
}


def resolve_row(row: dict[str, Any]) -> dict[str, Any]:
    source = row["source"]
    resolver = RESOLVERS.get(source)
    resolved = _base_result(source, row["video_name"])
    if resolver:
        resolved = resolver(row["video_name"])
    return {**row, **resolved}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_report(rows: list[dict[str, Any]]) -> str:
    status_counts = Counter(row["locator_status"] for row in rows)
    source_status = Counter(f"{row['source']}|{row['locator_status']}" for row in rows)
    return "\n".join(
        [
            "# 100 条候选上游定位审计",
            "",
            "本报告只解析文件名和官方访问路径，不执行网络文件枚举或媒体下载。",
            "`remote_verified=false` 表示尚未在官方文件清单中确认该具体文件存在。",
            "",
            "## 状态汇总",
            "",
            *[
                f"- `{status}`：{count}"
                for status, count in sorted(status_counts.items())
            ],
            "",
            "## 来源与状态",
            "",
            *[f"- `{key}`：{count}" for key, count in sorted(source_status.items())],
            "",
            "## 关键结论",
            "",
            "- CharadesEgo 存在与 JoyAI `video_name` 对应的社区预切片归档；不要按官方 action CSV 自行裁剪。",
            "- EgoProceL 和 Shot2Story 可从名称还原父视频或片段序号。",
            "- Open-o3 的普通名称需要连接官方 STGR JSON；`split_*.mp4` 不能按文件名定位。",
            "- GUI-World 的 JoyAI 数字 MP4 与官方分目录 MOV 不同，必须按问答文本连接元数据。",
            "- Ego4D 需要先签署许可并获得凭据，当前 UUID 仍需连接 VQA 元数据。",
            "- EgoBlind 数字 ID 可形成候选文件名，但必须检查官方 Drive 清单，且仅限研究。",
            "",
            "## 下一步（仍不下载媒体）",
            "",
            "1. 获取或读取官方的小型 JSON/CSV 文件索引。",
            "2. 对候选执行精确 basename、ID、问题文本和回答文本连接。",
            "3. 将结果更新为 `matched_unique`、`matched_multiple` 或 `not_found`。",
            "4. 只有唯一匹配且许可允许的条目，才进入媒体下载审批清单。",
            "",
        ]
    )


def run(input_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved = [resolve_row(row) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidate_upstream_audit_100.csv", resolved)
    _write_jsonl(output_dir / "candidate_upstream_audit_100.jsonl", resolved)
    (output_dir / "upstream_report.md").write_text(
        _render_report(resolved), encoding="utf-8"
    )
    summary = {
        "candidates": len(resolved),
        "status_counts": dict(Counter(row["locator_status"] for row in resolved)),
        "source_counts": dict(Counter(row["source"] for row in resolved)),
        "remote_verified": sum(bool(row["remote_verified"]) for row in resolved),
    }
    (output_dir / "upstream_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="candidate JSONL from audit_annotations.py"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("audit_output"))
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
