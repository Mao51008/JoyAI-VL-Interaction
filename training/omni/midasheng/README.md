# MiDasheng 音频对齐训练

本目录仅保存 MiDasheng 实验的代码、版本化配置与操作说明。训练 manifest、特征缓存、日志和 projector checkpoint 一律写入主力服务器的：

```text
/data/maoyy/joyai-runtime/omni-training/midasheng/
```

不得写入 `datasets/`，也不得提交运行产物或基础模型权重。

## Phase 1 固定合同

- checkpoint：`/data/maoyy/models/midashenglm-7b-1021-bf16`，revision `76c3019`；
- 只使用 checkpoint 的 `audio_encoder`，不使用其官方 `audio_projector`；
- 输入为单声道 16 kHz waveform，缓存其带有效 mask 截断的原始 1280 维特征；
- 项目 projector 配置为 `LayerNorm(1280) → Linear(1280, 4096) → GELU → Dropout(0) → Linear(4096, 4096)`；
- 仅训练该 projector；MiDasheng、JoyAI 和视觉路径均冻结；不加载 LoRA；
- task sampler 固定为 VoiceAssistant 50%、LibriSpeech 25%、Clotho-AQA 25%，以有放回抽样实现，绝不按 manifest 自然行数混合。

## 操作顺序

下面命令草案都可能加载模型并使用 GPU；获得授权后由用户在主力服务器手动执行。运行前先核对分支和工作树。

```bash
/data/maoyy/miniforge3/bin/git -C /data/maoyy/JoyAI-VL-Interaction branch --show-current
/data/maoyy/miniforge3/bin/git -C /data/maoyy/JoyAI-VL-Interaction status --short

ROOT=/data/maoyy/joyai-runtime/omni-training/midasheng
mkdir -p "$ROOT"/manifests "$ROOT"/features "$ROOT"/runs

python -m training.omni.midasheng.prepare_phase1_manifest \
  --voiceassistant-manifest /path/to/voiceassistant_train.jsonl \
  --librispeech-manifest /path/to/librispeech_train.jsonl \
  --clotho-aqa-manifest /path/to/clotho_aqa_train.jsonl \
  --total-examples 100000 \
  --output-manifest "$ROOT"/manifests/phase1_train_100k.jsonl

python -m training.omni.midasheng.cache_features \
  --manifest "$ROOT"/manifests/phase1_train_100k.jsonl \
  --model-dir /data/maoyy/models/midashenglm-7b-1021-bf16 \
  --output-dir "$ROOT"/features/phase1_train_100k \
  --device cuda:0
```

缓存产生的是 Stage 2 formal-manifest 兼容格式，不能交给旧的 ASR-only `projector_stage1.train`：后者会把 VoiceAssistant 和 Clotho-AQA 错误地改写成转录提示。Phase 1 训练入口必须使用 `training.omni.midasheng` 的任务感知 collator，并将 `--output-dir` 限定在 `$ROOT/runs/`。
