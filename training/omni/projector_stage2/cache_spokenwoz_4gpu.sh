#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/maoyy/JoyAI-VL-Interaction}"
PYTHON_BIN="${PYTHON_BIN:-/data/maoyy/joyai-runtime/omni-training/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/maoyy/datasets/projector_stage2/spokenwoz_turns_20260807}"
AUDIO_MODEL="${AUDIO_MODEL:-/data/maoyy/models/Qwen3-ASR-1.7B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/maoyy/datasets/projector_stage2/spokenwoz_feature_cache_4gpu_20260807}"
WORKERS=4

format_duration() {
  local seconds="$1"
  printf '%02d:%02d:%02d' "$((seconds / 3600))" "$(((seconds % 3600) / 60))" "$((seconds % 60))"
}

test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/parts" "$OUTPUT_ROOT/logs"
cd "$REPO_ROOT"

pids=()
for shard_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard_index" "$PYTHON_BIN" -m training.omni.projector_stage2.cache_features \
    --manifest "$DATA_ROOT/train.jsonl" --manifest "$DATA_ROOT/dev.jsonl" \
    --audio-model "$AUDIO_MODEL" --output-dir "$OUTPUT_ROOT/parts/worker-$shard_index" \
    --device cuda:0 --num-shards "$WORKERS" --shard-index "$shard_index" \
    --progress-file "$OUTPUT_ROOT/parts/worker-$shard_index/progress.json" \
    >"$OUTPUT_ROOT/logs/worker-$shard_index.log" 2>&1 &
  pids+=("$!")
done

while :; do
  completed=0
  total=0
  active=0
  for shard_index in 0 1 2 3; do
    progress_file="$OUTPUT_ROOT/parts/worker-$shard_index/progress.json"
    if [ -f "$progress_file" ]; then
      worker_completed="$(grep -o '"completed": [0-9]*' "$progress_file" | awk '{print $2}' || true)"
      worker_total="$(grep -o '"total": [0-9]*' "$progress_file" | awk '{print $2}' || true)"
      completed=$((completed + ${worker_completed:-0}))
      total=$((total + ${worker_total:-0}))
    fi
    if kill -0 "${pids[$shard_index]}" 2>/dev/null; then
      active=1
    fi
  done
  if [ "$total" -gt 0 ]; then
    if [ "$completed" -gt 0 ]; then
      eta_seconds=$((SECONDS * (total - completed) / completed))
      printf '\rfeature cache: %d/%d (%.1f%%), elapsed %s, ETA %s' \
        "$completed" "$total" \
        "$(awk -v completed="$completed" -v total="$total" 'BEGIN { print 100 * completed / total }')" \
        "$(format_duration "$SECONDS")" "$(format_duration "$eta_seconds")"
    else
      printf '\rfeature cache: 0/%d (loading models...)' "$total"
    fi
  fi
  [ "$active" -eq 0 ] && break
  sleep 5
done
printf '\n'

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$PYTHON_BIN" -m training.omni.projector_stage2.cache_features \
  --manifest "$DATA_ROOT/train.jsonl" --manifest "$DATA_ROOT/dev.jsonl" \
  --output-dir "$OUTPUT_ROOT/merged" \
  --merge-part "$OUTPUT_ROOT/parts/worker-0" --merge-part "$OUTPUT_ROOT/parts/worker-1" \
  --merge-part "$OUTPUT_ROOT/parts/worker-2" --merge-part "$OUTPUT_ROOT/parts/worker-3"
