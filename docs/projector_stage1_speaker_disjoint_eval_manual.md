# Projector 阶段一 speaker-disjoint 评测手动方案

本文档只提供命令草案，不由 Codex 自动执行。执行前必须由用户在主力服务器端手动启动并确认 GPU 授权。

## 固定输入

```bash
CHECKOUT=/data/maoyy/joyai-runtime/eval-checkouts/JoyAI-VL-Interaction-9757a1c
ROOT=/data/maoyy/datasets/projector_stage1
OUT=$ROOT/speaker_disjoint_eval_20260805
MAN=$ROOT/librispeech_train100_20260804/validation_samples_train-clean-100.jsonl
FEAT=$ROOT/librispeech_train100_20260804/features-validation-20260804
STEP0=$ROOT/first10_diverse_overfit_20260805/step-0.pt
BEST=$ROOT/diverse10_controlled_lr3e-5_2000steps_20260805/best.pt
MODEL=/data/maoyy/models/jdopensource/JoyAI-VL-Interaction
ASR_MODEL=/data/maoyy/models/Qwen3-ASR-1.7B
PY=/data/maoyy/joyai-runtime/omni-training/.venv/bin/python
```

固定校验项：validation manifest 696 条；diverse10 训练 manifest 10 条；sample_id 和 speaker 必须均无交集。记录两个 manifest 的 SHA-256、样本数和 speaker 数。

## 阶段一：固定 32 条子集

先用 validation manifest 的原始顺序取前 32 条，生成 `$OUT/validation_subset32.jsonl`。分别对 `step-0.pt` 和 `best.pt` 运行 normal teacher-forced loss 与自由生成：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m training.omni.projector_stage1.evaluate \
  --manifest $OUT/validation_subset32.jsonl --feature-dir $FEAT \
  --checkpoint $CHECKPOINT --joyai-model $MODEL --device cuda:0 \
  --batch-size 1 --audio-ablation none --seed 3407 \
  --output $OUT/subset32_${LABEL}_tf.json --no-progress

CUDA_VISIBLE_DEVICES=0 $PY -m training.omni.projector_stage1.evaluate_wer \
  --manifest $OUT/validation_subset32.jsonl --feature-dir $FEAT \
  --checkpoint $CHECKPOINT --joyai-model $MODEL --device cuda:0 \
  --max-samples 32 --max-new-tokens 128 --audio-ablation none --seed 3407 \
  --output $OUT/subset32_${LABEL}_wer.json
```

其中分别设置 `CHECKPOINT=$STEP0 LABEL=step0` 和 `CHECKPOINT=$BEST LABEL=best`。若 best 的 normal TF loss 与 WER/CER 均没有可测改善，立即停止，不扩展 696 条。

## 阶段二：完整 696 条消融

只有阶段一 best 相对 step-0 有改善时，才对两个 checkpoint 分别运行以下条件：

| CLI 参数 | canonical 输出名 | 语义 |
| --- | --- | --- |
| `none` | `none` | 正常音频特征 |
| `waveform-zero` | `waveform-zero` | 使用零波形重新提取的 ASR 特征 |
| `projected-zero` | `projected-zero` | projector 输出置零 |
| `shuffle` | `cross-sample-shuffle` | 不同样本之间交换完整音频特征，确定性无自交换 |
| `temporal-shuffle` | `within-sample-temporal-shuffle` | 每条样本内部沿时间轴确定性无固定点置换，保持 shape/dtype |

normal、projected-zero、跨样本交换和样本内时间打乱不需要新增 cache；waveform-zero 需要先单独提取 cache：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m training.omni.projector_stage1.cache_features \
  --manifest $MAN --audio-model $ASR_MODEL \
  --output-dir $OUT/cache-validation-waveform-zero \
  --device cuda:0 --shard-size 256 --waveform-zero
```

随后每个 checkpoint/条件运行：

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m training.omni.projector_stage1.evaluate \
  --manifest $MAN --feature-dir $FEAT --checkpoint $CHECKPOINT \
  --joyai-model $MODEL --device cuda:0 --batch-size 1 \
  --audio-ablation $ABLATION --zero-feature-dir $OUT/cache-validation-waveform-zero \
  --seed 3407 --output $OUT/${LABEL}_${ABLATION}_tf.json --no-progress

CUDA_VISIBLE_DEVICES=0 $PY -m training.omni.projector_stage1.evaluate_wer \
  --manifest $MAN --feature-dir $FEAT --checkpoint $CHECKPOINT \
  --joyai-model $MODEL --device cuda:0 --max-samples 696 \
  --max-new-tokens 128 --audio-ablation $ABLATION \
  --zero-feature-dir $OUT/cache-validation-waveform-zero --seed 3407 \
  --output $OUT/${LABEL}_${ABLATION}_wer.json
```

`--zero-feature-dir` 只在 `waveform-zero` 时需要；为避免歧义，不要把旧的 `zero` 或 `shuffle` 结果描述成时序打乱。

## 汇总字段与停止条件

每个 TF JSON 应记录：`checkpoint`、canonical `audio_ablation`、`samples`、`supervised_tokens`、总体 `loss`、`perplexity` 及逐样本 `speaker`、`duration_ms`、`loss`、`eos_loss`。

每个 WER JSON 应记录：`checkpoint`、canonical `audio_ablation`、`samples`、总体 `wer`/`cer`，以及逐样本 `speaker`、`duration_ms`、`generated_tokens`、`generated_eos`、`exact_match`、典型失败文本。

可用 `compare_wer.py` 汇总 normal、waveform-zero、projected-zero、跨样本交换和样本内时间打乱的逐样本差值、EOS 率、平均生成长度及典型失败。

满足以下任一条件立即停止：best 在固定子集上没有 normal 改善；GPU OOM 或出现其他用户进程；输出目录已存在且可能覆盖历史结果；manifest/cache fingerprint 不匹配；任何评测进程无法可靠停止。此任务不训练 95 条、不进行全量训练、不解冻基础模型。
