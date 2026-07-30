"""Audit JoyAI-VL-Interaction annotation shards without downloading media.

The script intentionally uses only the Python standard library.  It validates
annotation structure, summarizes sources/timestamps/repetition, and creates a
deterministic source-and-time-stratified video review manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("video_name", "task_type", "source", "question", "response")
TIME_BUCKETS = (
    (10, "00_0-10s"),
    (30, "01_11-30s"),
    (60, "02_31-60s"),
    (120, "03_61-120s"),
    (math.inf, "04_121s+"),
)


def parse_time_values(raw: Any) -> list[float]:
    """Parse the comma-separated seconds format used by convert_data.py."""
    if raw is None or isinstance(raw, bool):
        return []
    values: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    """Return interpolated percentiles for a numeric sequence."""
    ordered = sorted(values)
    if not ordered:
        return {}
    result: dict[str, float] = {}
    for percentile in (0, 25, 50, 75, 90, 95, 99, 100):
        position = (len(ordered) - 1) * percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        result[f"p{percentile}"] = round(value, 3)
    return result


def annotation_time_bucket(max_time: float) -> str:
    for upper_bound, label in TIME_BUCKETS:
        if max_time <= upper_bound:
            return label
    raise AssertionError("TIME_BUCKETS must end with infinity")


def _stable_rank(seed: int, *parts: str) -> str:
    joined = "\x1f".join((str(seed), *parts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _content_nodes(value: Any) -> list[dict[str, Any]]:
    """Return nested dictionaries that carry annotation content."""
    nodes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "content" in value:
            nodes.append(value)
        for child in value.values():
            nodes.extend(_content_nodes(child))
    elif isinstance(value, list):
        for child in value:
            nodes.extend(_content_nodes(child))
    return nodes


def _allocate_quotas(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    sample_size: int,
) -> dict[tuple[str, str, str], int]:
    """Allocate proportional quotas while retaining small strata when possible."""
    target = min(sample_size, sum(len(rows) for rows in groups.values()))
    keys = sorted(groups)
    if not target:
        return {key: 0 for key in keys}
    if len(keys) > target:
        selected = sorted(keys, key=lambda key: (-len(groups[key]), key))[:target]
        return {key: int(key in selected) for key in keys}

    total = sum(len(groups[key]) for key in keys)
    exact = {key: target * len(groups[key]) / total for key in keys}
    quotas = {
        key: min(len(groups[key]), max(1, math.floor(exact[key]))) for key in keys
    }

    while sum(quotas.values()) > target:
        reducible = [key for key in keys if quotas[key] > 1]
        key = max(
            reducible, key=lambda item: (quotas[item] - exact[item], quotas[item], item)
        )
        quotas[key] -= 1

    while sum(quotas.values()) < target:
        expandable = [key for key in keys if quotas[key] < len(groups[key])]
        key = max(
            expandable,
            key=lambda item: (exact[item] - quotas[item], len(groups[item]), item),
        )
        quotas[key] += 1
    return quotas


def stratified_sample(
    videos: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select deterministic strata plus high-count and long-timestamp outliers."""
    target = min(sample_size, len(videos))
    outlier_quota = min(10, target // 10)
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    mandatory: dict[tuple[str, str], dict[str, Any]] = {}

    ranked_by_count = sorted(
        videos,
        key=lambda row: (-row["annotation_count"], row["source"], row["video_name"]),
    )
    ranked_by_time = sorted(
        videos,
        key=lambda row: (
            -row["max_annotation_time_s"],
            row["source"],
            row["video_name"],
        ),
    )
    for reason, ranked in (
        ("high_annotation_count", ranked_by_count[:outlier_quota]),
        ("high_max_annotation_time", ranked_by_time[:outlier_quota]),
    ):
        for row in ranked:
            identity = (row["source"], row["video_name"])
            mandatory[identity] = row
            reasons[identity].append(reason)

    remaining = [
        row for row in videos if (row["source"], row["video_name"]) not in mandatory
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for video in remaining:
        key = (
            video["source"],
            video["task_type"],
            video["annotation_time_bucket"],
        )
        groups[key].append(video)

    quotas = _allocate_quotas(groups, target - len(mandatory))
    selected = [dict(row) for row in mandatory.values()]
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda row: _stable_rank(seed, row["source"], row["video_name"]),
        )
        for row in ranked[: quotas[key]]:
            selected.append(dict(row))
            reasons[(row["source"], row["video_name"])].append(
                "source_task_time_stratified"
            )

    selected.sort(
        key=lambda row: (
            row["source"],
            row["annotation_time_bucket"],
            row["video_name"],
        )
    )
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"audit-{index:03d}"
        identity = (row["source"], row["video_name"])
        row["selection_reason"] = "+".join(reasons[identity])
        row["upstream_lookup_status"] = "pending"
        row["media_download_status"] = "not_started"
        row["annotation_review_status"] = "pending"
        row["audio_review_status"] = "pending"
    return selected


def audit_records(records: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build an audit summary and one aggregate row per source/video."""
    sources: Counter[str] = Counter()
    task_types: Counter[str] = Counter()
    field_types: dict[str, Counter[str]] = defaultdict(Counter)
    issue_counts: Counter[str] = Counter()
    issue_examples: dict[str, list[int]] = defaultdict(list)
    question_texts: Counter[str] = Counter()
    response_texts: Counter[str] = Counter()
    exact_records: Counter[str] = Counter()
    record_max_times: list[float] = []
    response_delays: list[float] = []
    timing_relation: Counter[str] = Counter()
    question_lengths: list[float] = []
    response_lengths: list[float] = []
    videos: dict[tuple[str, str], dict[str, Any]] = {}
    video_name_sources: dict[str, set[str]] = defaultdict(set)
    record_identity: dict[str, dict[str, Any]] = {}

    def add_issue(name: str, index: int) -> None:
        issue_counts[name] += 1
        if len(issue_examples[name]) < 10:
            issue_examples[name].append(index)

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            add_issue("record_not_object", index)
            continue
        for field in REQUIRED_FIELDS:
            field_types[field][type(record.get(field)).__name__] += 1
            if field not in record:
                add_issue(f"missing_{field}", index)

        source = record.get("source")
        video_name = record.get("video_name")
        task_type = record.get("task_type")
        if not isinstance(source, str) or not source.strip():
            add_issue("invalid_source", index)
        if not isinstance(video_name, str) or not video_name.strip():
            add_issue("invalid_video_name", index)
        if not isinstance(task_type, str) or not task_type.strip():
            add_issue("invalid_task_type", index)
        if not all(isinstance(value, str) for value in (source, video_name, task_type)):
            continue

        sources[source] += 1
        task_types[task_type] += 1
        video_name_sources[video_name].add(source)
        key = (source, video_name)
        if key not in videos:
            videos[key] = {
                "source": source,
                "video_name": video_name,
                "task_types": set(),
                "annotation_count": 0,
                "max_annotation_time_s": 0.0,
                "min_annotation_time_s": None,
                "question_example": "",
                "response_example": "",
            }
        video = videos[key]
        video["task_types"].add(task_type)
        video["annotation_count"] += 1

        questions = _content_nodes(record.get("question"))
        responses = _content_nodes(record.get("response"))
        if not isinstance(record.get("question"), list):
            add_issue("question_not_list", index)
        if not isinstance(record.get("response"), list):
            add_issue("response_not_list", index)
        if not questions:
            add_issue("question_without_content_node", index)
        if not responses:
            add_issue("response_without_content_node", index)

        question_times: list[float] = []
        response_times: list[float] = []
        for branch, nodes, text_counter, lengths, times in (
            ("question", questions, question_texts, question_lengths, question_times),
            ("response", responses, response_texts, response_lengths, response_times),
        ):
            for node in nodes:
                content = node.get("content")
                if not isinstance(content, str) or not content.strip():
                    add_issue(f"invalid_{branch}_content", index)
                else:
                    normalized = " ".join(content.split())
                    text_counter[normalized] += 1
                    lengths.append(float(len(content)))
                    example_field = f"{branch}_example"
                    if not video[example_field]:
                        video[example_field] = normalized[:240]
                if "time" not in node:
                    add_issue(f"missing_{branch}_time", index)
                    continue
                try:
                    parsed = parse_time_values(node["time"])
                except (TypeError, ValueError):
                    add_issue(f"invalid_{branch}_time", index)
                    continue
                if not parsed:
                    add_issue(f"empty_{branch}_time", index)
                if any(value < 0 for value in parsed):
                    add_issue(f"negative_{branch}_time", index)
                times.extend(parsed)

        row_times = question_times + response_times
        if row_times:
            maximum = max(row_times)
            minimum = min(row_times)
            record_max_times.append(maximum)
            video["max_annotation_time_s"] = max(
                video["max_annotation_time_s"], maximum
            )
            current_minimum = video["min_annotation_time_s"]
            video["min_annotation_time_s"] = (
                minimum if current_minimum is None else min(current_minimum, minimum)
            )
        else:
            add_issue("record_without_valid_time", index)

        if question_times and response_times:
            delay = min(response_times) - min(question_times)
            response_delays.append(delay)
            if delay < 0:
                timing_relation["response_before_question"] += 1
            elif delay == 0:
                timing_relation["response_same_time"] += 1
            else:
                timing_relation["response_after_question"] += 1

        digest = hashlib.sha256(
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        exact_records[digest] += 1
        record_identity.setdefault(
            digest,
            {
                "source": source,
                "video_name": video_name,
                "question": video["question_example"],
                "response": video["response_example"],
                "question_times_s": question_times,
                "response_times_s": response_times,
            },
        )

    video_rows: list[dict[str, Any]] = []
    per_source_video_times: dict[str, list[float]] = defaultdict(list)
    per_source_annotation_counts: dict[str, list[float]] = defaultdict(list)
    time_buckets: Counter[str] = Counter()
    for video in videos.values():
        task_values = sorted(video.pop("task_types"))
        video["task_type"] = ",".join(task_values)
        if video["min_annotation_time_s"] is None:
            video["min_annotation_time_s"] = ""
        maximum = video["max_annotation_time_s"]
        video["annotation_time_bucket"] = annotation_time_bucket(maximum)
        time_buckets[video["annotation_time_bucket"]] += 1
        per_source_video_times[video["source"]].append(maximum)
        per_source_annotation_counts[video["source"]].append(video["annotation_count"])
        video_rows.append(video)

    source_summary: dict[str, Any] = {}
    for source in sorted(sources):
        source_videos = [row for row in video_rows if row["source"] == source]
        source_summary[source] = {
            "records": sources[source],
            "unique_videos": len(source_videos),
            "records_per_video": percentile_summary(
                per_source_annotation_counts[source]
            ),
            "video_max_annotation_time_s": percentile_summary(
                per_source_video_times[source]
            ),
            "sum_video_max_annotation_time_hours": round(
                sum(per_source_video_times[source]) / 3600, 3
            ),
        }

    duplicate_record_instances = sum(count - 1 for count in exact_records.values())
    duplicate_groups = [
        {**record_identity[digest], "count": count}
        for digest, count in exact_records.most_common()
        if count > 1
    ]
    videos_by_annotation_count = sorted(
        video_rows,
        key=lambda row: (-row["annotation_count"], row["source"], row["video_name"]),
    )
    videos_by_max_time = sorted(
        video_rows,
        key=lambda row: (
            -row["max_annotation_time_s"],
            row["source"],
            row["video_name"],
        ),
    )
    summary = {
        "records": len(records),
        "unique_source_video_pairs": len(videos),
        "unique_video_names": len(video_name_sources),
        "video_names_in_multiple_sources": sum(
            len(source_set) > 1 for source_set in video_name_sources.values()
        ),
        "sources": dict(sources.most_common()),
        "task_types": dict(task_types.most_common()),
        "field_types": {
            field: dict(counts.most_common())
            for field, counts in sorted(field_types.items())
        },
        "issues": dict(issue_counts.most_common()),
        "issue_example_record_indexes": dict(sorted(issue_examples.items())),
        "exact_duplicate_record_instances": duplicate_record_instances,
        "exact_duplicate_record_groups": sum(
            count > 1 for count in exact_records.values()
        ),
        "top_exact_duplicate_groups": duplicate_groups[:50],
        "unique_question_texts": len(question_texts),
        "unique_response_texts": len(response_texts),
        "question_text_duplicate_rate": round(
            1 - len(question_texts) / max(sum(question_texts.values()), 1), 6
        ),
        "response_text_duplicate_rate": round(
            1 - len(response_texts) / max(sum(response_texts.values()), 1), 6
        ),
        "top_repeated_questions": [
            {"text": text, "count": count}
            for text, count in question_texts.most_common(20)
        ],
        "top_repeated_responses": [
            {"text": text, "count": count}
            for text, count in response_texts.most_common(20)
        ],
        "question_character_count": percentile_summary(question_lengths),
        "response_character_count": percentile_summary(response_lengths),
        "record_max_annotation_time_s": percentile_summary(record_max_times),
        "video_max_annotation_time_s": percentile_summary(
            row["max_annotation_time_s"] for row in video_rows
        ),
        "sum_video_max_annotation_time_s": round(
            sum(row["max_annotation_time_s"] for row in video_rows), 3
        ),
        "sum_video_max_annotation_time_hours": round(
            sum(row["max_annotation_time_s"] for row in video_rows) / 3600, 3
        ),
        "video_annotation_count": percentile_summary(
            row["annotation_count"] for row in video_rows
        ),
        "top_videos_by_annotation_count": videos_by_annotation_count[:20],
        "top_videos_by_max_annotation_time": videos_by_max_time[:20],
        "video_time_buckets": dict(sorted(time_buckets.items())),
        "question_response_timing": dict(timing_relation),
        "response_delay_s": percentile_summary(response_delays),
        "per_source": source_summary,
        "important_note": (
            "Annotation timestamps are event positions and only lower bounds on media duration; "
            "actual duration, decodability, audio presence, and A/V alignment require media files."
        ),
    }
    video_rows.sort(key=lambda row: (row["source"], row["video_name"]))
    return summary, video_rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    summary: dict[str, Any], input_path: Path, sample_size: int
) -> str:
    source_lines = [
        (
            f"| {source} | {details['records']} | {details['unique_videos']} | "
            f"{details['video_max_annotation_time_s'].get('p50', '-')} | "
            f"{details['video_max_annotation_time_s'].get('p95', '-')} |"
        )
        for source, details in summary["per_source"].items()
    ]
    issue_text = (
        "未发现结构或时间字段异常"
        if not summary["issues"]
        else "；".join(f"{name}: {count}" for name, count in summary["issues"].items())
    )
    same_time = summary["question_response_timing"].get("response_same_time", 0)
    same_time_rate = same_time / max(summary["records"], 1)
    highest_count_video = summary["top_videos_by_annotation_count"][0]
    return "\n".join(
        [
            "# JoyAI 标注分片本地审计报告",
            "",
            f"- 输入文件：`{input_path.as_posix()}`",
            f"- 标注记录：{summary['records']}",
            f"- 唯一视频（source + video_name）：{summary['unique_source_video_pairs']}",
            f"- 任务类型：{json.dumps(summary['task_types'], ensure_ascii=False)}",
            f"- 结构检查：{issue_text}",
            f"- 精确重复记录实例：{summary['exact_duplicate_record_instances']}",
            f"- 问题文本重复率：{summary['question_text_duplicate_rate']:.2%}",
            f"- 回答文本重复率：{summary['response_text_duplicate_rate']:.2%}",
            "",
            "## 重要发现",
            "",
            (
                f"- {same_time} 条（{same_time_rate:.2%}）记录的问题与回答时间戳相同。"
                "这更接近在指定时刻注入文本问题的视觉问答数据，不是原生语音全双工样本。"
            ),
            (
                f"- 聚合标注最多的视频是 `{highest_count_video['source']}/"
                f"{highest_count_video['video_name']}`，共有 "
                f"{highest_count_video['annotation_count']} 条标注，但最大标注时间仅 "
                f"{highest_count_video['max_annotation_time_s']} 秒。应优先检查这是否为合法的"
                "密集问答，或不同上游目录中的同名文件发生了键碰撞。"
            ),
            (
                "- 问题和回答文本重复率包含模板化提问、动作短标签以及选择题的 A/B/C/D，"
                "不能等同于精确重复样本；完整记录的精确重复数量单独统计。"
            ),
            "",
            "## 来源分布",
            "",
            "| 来源 | 标注数 | 唯一视频数 | 视频最大标注时间 P50/秒 | P95/秒 |",
            "|---|---:|---:|---:|---:|",
            *source_lines,
            "",
            "## 时间和抽样解释",
            "",
            (
                f"视频级最大标注时间分位数："
                f"`{json.dumps(summary['video_max_annotation_time_s'], ensure_ascii=False)}`。"
            ),
            (
                "所有唯一视频的最大标注时间之和为 "
                f"{summary['sum_video_max_annotation_time_hours']} 小时。这只是媒体总时长的"
                "下界，且仍可能受同名文件键碰撞影响，不能直接用于计算下载空间。"
            ),
            "",
            (
                "时间戳表示问题或回答在媒体中的位置，只能作为媒体时长下界，不能据此断言"
                "原视频总时长。真实时长、是否可解码、是否包含音轨以及音画是否匹配，必须"
                "在后续取得媒体文件后用 ffprobe 和人工抽检确认。"
            ),
            "",
            (
                f"已生成 {sample_size} 条候选：固定纳入标注数和最大时间戳的极端项，"
                "其余名额按来源、任务类型和最大标注时间区间分层。"
                "清单中的上游定位、下载、标注和音频状态默认均为 pending/not_started，"
                "不代表已经找到或验证视频。"
            ),
            "",
            "## 下一步",
            "",
            "1. 为每个来源编写上游文件名解析规则，只做 URL/文件名 dry-run。",
            "2. 先定位候选，不批量下载；记录无法匹配、需授权或已失效的条目。",
            "3. 获得确认后下载小样本并检查时长、解码、音轨和音画同步。",
            "4. 人工审核问题/回答是否被画面和音频支持，再决定是否扩展到 1000 条。",
            "",
        ]
    )


def run(
    input_path: Path, output_dir: Path, sample_size: int, seed: int
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError("input JSON must be an array of annotation objects")

    summary, video_rows = audit_records(records)
    candidates = stratified_sample(video_rows, sample_size, seed)
    summary["input_file"] = str(input_path)
    summary["input_bytes"] = input_path.stat().st_size
    summary["sample_size"] = len(candidates)
    summary["sample_seed"] = seed
    summary["candidate_strata"] = dict(
        Counter(
            f"{row['source']}|{row['task_type']}|{row['annotation_time_bucket']}"
            for row in candidates
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / f"candidate_videos_{len(candidates)}.jsonl", candidates)
    _write_csv(output_dir / f"candidate_videos_{len(candidates)}.csv", candidates)
    report = _render_markdown(summary, input_path, len(candidates))
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JoyAI annotation JSON array")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audit_output"),
        help="directory for summary, report, and candidate manifests",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")

    summary = run(args.input, args.output_dir, args.sample_size, args.seed)
    print(
        json.dumps(
            {
                "records": summary["records"],
                "unique_videos": summary["unique_source_video_pairs"],
                "issues": summary["issues"],
                "sample_size": summary["sample_size"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
