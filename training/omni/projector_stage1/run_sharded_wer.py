"""Run one WER condition across explicit GPUs and merge its shard results."""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence


def parse_progress_line(line: str) -> dict[str, str | int] | None:
    """Parse the latest tqdm line emitted by evaluate_wer."""
    match = re.search(
        r"shard\s+(?P<shard>\d+)/(?:\d+).*?(?P<completed>\d+)/(?P<total>\d+)\s+\[(?P<timing>[^\]]+)\]",
        line.replace("\r", ""),
    )
    if match is None:
        return None
    timing = match.group("timing")
    elapsed, _, eta = timing.partition("<")
    return {
        "shard": int(match.group("shard")),
        "completed": int(match.group("completed")),
        "total": int(match.group("total")),
        "elapsed": elapsed.strip(),
        "eta": eta.split(",", 1)[0].strip() if eta else "?",
    }


def latest_progress(log_path: Path) -> dict[str, str | int] | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(re.split(r"[\r\n]", text)):
        parsed = parse_progress_line(line)
        if parsed is not None:
            return parsed
    return None


def _build_child_command(args: argparse.Namespace, shard: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "training.omni.projector_stage1.evaluate_wer",
        "--manifest",
        str(args.manifest),
        "--feature-dir",
        str(args.feature_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--joyai-model",
        args.joyai_model,
        "--device",
        "cuda:0",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--audio-ablation",
        args.audio_ablation,
        "--seed",
        str(args.seed),
        "--num-shards",
        "4",
        "--shard-index",
        str(shard),
        "--output",
        str(output),
    ]
    if args.zero_feature_dir is not None:
        command.extend(("--zero-feature-dir", str(args.zero_feature_dir)))
    return command


def _validate_args(args: argparse.Namespace) -> list[str]:
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise ValueError("--gpus must contain four distinct GPU ids")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.output_dir.exists() or args.merged_output.exists():
        raise FileExistsError("output-dir or merged-output already exists; refusing to overwrite")
    return [str(gpu) for gpu in args.gpus]


def run_sharded(
    args: argparse.Namespace,
    *,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    merge_runner: Callable[[Sequence[Path], Path], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    print_fn: Callable[..., None] = print,
) -> dict:
    """Run four WER shards, then merge only after every child succeeds."""
    gpus = _validate_args(args)
    args.output_dir.mkdir(parents=True)
    children: list[subprocess.Popen] = []
    log_handles = []
    stopping = False

    def stop_children(signum: int = signal.SIGTERM) -> None:
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass
        for child in children:
            child.wait()

    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    previous_handlers = {sig: signal.getsignal(sig) for sig in signals}
    for sig in previous_handlers:
        signal.signal(sig, stop_children)
    try:
        shard_outputs = []
        for shard, gpu in enumerate(gpus):
            output = args.output_dir / f"shard-{shard}.json"
            log_path = args.output_dir / f"shard-{shard}.log"
            handle = log_path.open("w", encoding="utf-8")
            log_handles.append(handle)
            command = _build_child_command(args, shard, output)
            environment = dict(getattr(args, "environment", {}))
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            children.append(popen_factory(command, stdout=handle, stderr=subprocess.STDOUT, env=environment))
            shard_outputs.append(output)

        failed = False
        while True:
            statuses = [child.poll() for child in children]
            for shard in range(4):
                progress = latest_progress(args.output_dir / f"shard-{shard}.log")
                if progress:
                    print_fn(
                        f"shard {shard + 1}/4: {progress['completed']}/{progress['total']} "
                        f"elapsed {progress['elapsed']} ETA {progress['eta']}"
                    )
            if all(status is not None for status in statuses):
                failed = any(status != 0 for status in statuses)
                break
            sleep_fn(5)

        for child in children:
            if child.poll() is None:
                child.wait()
        if stopping or failed:
            raise RuntimeError("one or more WER shards failed; refusing to merge")
        if args.merged_output.exists():
            raise FileExistsError("merged-output appeared during evaluation; refusing to overwrite")
        if merge_runner is None:
            from .merge_shards import merge_shard_results

            def merge_runner(paths: Sequence[Path], output: Path) -> None:
                results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
                merged = merge_shard_results(results)
                output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        merge_runner(shard_outputs, args.merged_output)
        result = json.loads(args.merged_output.read_text(encoding="utf-8"))
        print_fn(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
        return result
    except BaseException:
        stop_children()
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for handle in log_handles:
            handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--joyai-model", required=True)
    parser.add_argument("--audio-ablation", default="none")
    parser.add_argument("--zero-feature-dir", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpus", type=int, nargs=4, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.environment = dict()
    run_sharded(args)


if __name__ == "__main__":
    main()
