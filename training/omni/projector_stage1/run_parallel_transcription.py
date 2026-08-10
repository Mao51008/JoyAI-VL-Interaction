"""Run two Qwen-ASR and two projector-VLM transcription shards, then join them."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .evaluate_transcription import merge_role_results


def build_command(args: argparse.Namespace, mode: str, shard_index: int, output: Path) -> list[str]:
    """Build one single-GPU worker command; CUDA visibility is set by the parent."""
    command = [
        sys.executable, "-m", "training.omni.projector_stage1.evaluate_transcription",
        "--manifest", str(args.manifest), "--checkpoint", str(args.checkpoint),
        "--audio-model", args.audio_model, "--joyai-model", args.joyai_model,
        "--device", "cuda:0", "--mode", mode, "--num-shards", "2", "--shard-index",
        str(shard_index), "--max-new-tokens", str(args.max_new_tokens), "--output", str(output),
    ]
    if args.qwen_language is not None:
        command.extend(("--qwen-language", args.qwen_language))
    if args.max_samples is not None:
        command.extend(("--max-samples", str(args.max_samples)))
    return command


def run_parallel(args: argparse.Namespace) -> dict:
    """Run fixed four-card data parallel evaluation without overwriting outputs."""
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise ValueError("--gpus must contain four distinct GPU ids")
    if args.output_dir.exists() or args.output.exists():
        raise FileExistsError("output-dir or output already exists; refusing to overwrite")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    tasks = [
        ("qwen-asr", 0, args.gpus[0]), ("qwen-asr", 1, args.gpus[1]),
        ("projector-vlm", 0, args.gpus[2]), ("projector-vlm", 1, args.gpus[3]),
    ]
    args.output_dir.mkdir(parents=True)
    children: list[subprocess.Popen] = []
    handles = []
    outputs: dict[str, list[Path]] = {"qwen-asr": [], "projector-vlm": []}
    try:
        for mode, shard_index, gpu in tasks:
            output = args.output_dir / f"{mode}-shard-{shard_index}.json"
            log = args.output_dir / f"{mode}-shard-{shard_index}.log"
            handle = log.open("w", encoding="utf-8")
            handles.append(handle)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            children.append(subprocess.Popen(
                build_command(args, mode, shard_index, output), stdout=handle,
                stderr=subprocess.STDOUT, env=environment,
            ))
            outputs[mode].append(output)
        statuses = [child.wait() for child in children]
        if any(status != 0 for status in statuses):
            raise RuntimeError("one or more transcription workers failed; refusing to merge")
        missing = [path for paths in outputs.values() for path in paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing worker outputs; refusing to merge: {missing}")
        result = merge_role_results(
            [json.loads(path.read_text(encoding="utf-8")) for path in outputs["projector-vlm"]],
            [json.loads(path.read_text(encoding="utf-8")) for path in outputs["qwen-asr"]],
        )
        result["workers"] = {
            "qwen_asr_gpus": args.gpus[:2], "projector_vlm_gpus": args.gpus[2:],
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return result
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for child in children:
            child.wait()
        raise
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio-model", required=True)
    parser.add_argument("--joyai-model", required=True)
    parser.add_argument("--gpus", type=int, nargs=4, required=True)
    parser.add_argument("--qwen-language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run_parallel(parser.parse_args())


if __name__ == "__main__":
    main()
