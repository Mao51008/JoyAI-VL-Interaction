#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/maoyy/JoyAI-VL-Interaction}"
PYTHON_BIN="${PYTHON_BIN:-/data/maoyy/joyai-runtime/omni-training/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/maoyy/datasets/projector_stage2/spokenwoz_turns_20260807}"
AUDIO_MODEL="${AUDIO_MODEL:-/data/maoyy/models/Qwen3-ASR-1.7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/maoyy/datasets/projector_stage2/spokenwoz_feature_cache_4gpu_20260807}"
WORKERS=4

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/parts" "$OUTPUT_ROOT/logs"
cd "$REPO_ROOT"

pids=()
for shard_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard_index" "$PYTHON_BIN" -m training.omni.projector_stage2.cache_features \
    --manifest "$DATA_ROOT/train.jsonl" --manifest "$DATA_ROOT/dev.jsonl" \
    --audio-model "$AUDIO_MODEL" --output-dir "$OUTPUT_ROOT/parts/worker-$shard_index" \
    --device cuda:0 --num-shards "$WORKERS" --shard-index "$shard_index" >"$OUTPUT_ROOT/logs/worker-$shard_index.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON_BIN" -m training.omni.projector_stage2.cache_features \
  --manifest "$DATA_ROOT/train.jsonl" --manifest "$DATA_ROOT/dev.jsonl" \
  --output-dir "$OUTPUT_ROOT/merged" \
  --merge-part "$OUTPUT_ROOT/parts/worker-0" --merge-part "$OUTPUT_ROOT/parts/worker-1" \
  --merge-part "$OUTPUT_ROOT/parts/worker-2" --merge-part "$OUTPUT_ROOT/parts/worker-3"
