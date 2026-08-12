#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/maoyy/JoyAI-VL-Interaction}"
PYTHON_BIN="${PYTHON_BIN:-/data/maoyy/joyai-runtime/omni-training/.venv/bin/python}"
AUDIO_MODEL="${AUDIO_MODEL:-/data/maoyy/models/Qwen3-ASR-1.7B}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:?set TRAIN_MANIFEST}"
DEV_MANIFEST="${DEV_MANIFEST:?set DEV_MANIFEST}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
TRAINING_TASK="${TRAINING_TASK:-}"
WORKERS=4

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/parts" "$OUTPUT_ROOT/logs"
cd "$REPO_ROOT"

pids=()
task_args=()
if [ -n "$TRAINING_TASK" ]; then
  task_args+=(--training-task "$TRAINING_TASK")
fi
for shard_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard_index" "$PYTHON_BIN" -m training.omni.projector_stage2.cache_features \
    --manifest "$TRAIN_MANIFEST" --manifest "$DEV_MANIFEST" \
    --audio-model "$AUDIO_MODEL" --output-dir "$OUTPUT_ROOT/parts/worker-$shard_index" \
    --device cuda:0 --num-shards "$WORKERS" --shard-index "$shard_index" \
    --progress-file "$OUTPUT_ROOT/parts/worker-$shard_index/progress.json" \
    "${task_args[@]}" \
    >"$OUTPUT_ROOT/logs/worker-$shard_index.log" 2>&1 &
  pids+=("$!")
done

while :; do
  completed=0 total=0 active=0
  for shard_index in 0 1 2 3; do
    progress_file="$OUTPUT_ROOT/parts/worker-$shard_index/progress.json"
    if [ -f "$progress_file" ]; then
      completed=$((completed + $(grep -o '"completed": [0-9]*' "$progress_file" | awk '{print $2}')))
      total=$((total + $(grep -o '"total": [0-9]*' "$progress_file" | awk '{print $2}')))
    fi
    kill -0 "${pids[$shard_index]}" 2>/dev/null && active=1 || true
  done
  [ "$total" -gt 0 ] && printf '\rfeature cache: %d/%d (%.1f%%)' "$completed" "$total" "$(awk -v c="$completed" -v t="$total" 'BEGIN {print 100*c/t}')"
  [ "$active" -eq 0 ] && break
  sleep 5
done
printf '\n'
for pid in "${pids[@]}"; do wait "$pid"; done

if [ -n "$TRAINING_TASK" ]; then
  echo "task-filtered cache parts are ready; merge them with the complementary cache separately"
  exit 0
fi

"$PYTHON_BIN" -m training.omni.projector_stage2.cache_features \
  --manifest "$TRAIN_MANIFEST" --manifest "$DEV_MANIFEST" --output-dir "$OUTPUT_ROOT/merged" \
  --merge-part "$OUTPUT_ROOT/parts/worker-0" --merge-part "$OUTPUT_ROOT/parts/worker-1" \
  --merge-part "$OUTPUT_ROOT/parts/worker-2" --merge-part "$OUTPUT_ROOT/parts/worker-3"
