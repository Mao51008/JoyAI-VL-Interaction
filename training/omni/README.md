# Omni B2：无 GPU 数据协议与训练骨架

## Projector 第一阶段

真实音频-文本 projector-only 训练代码统一位于 `projector_stage1/`：训练入口为
`training.omni.projector_stage1.train`，完整人工操作说明见
`projector_stage1/Projector第一阶段训练手册.md`。该目录与本 README 说明的 CPU B2
原型相互独立。

该目录实现里程碑 B2 的 CPU 原型，用来验证多模态样本格式、时间监督、批处理、假音频
Encoder、可训练 bridge、动作 loss 和断点恢复。它不加载 JoyAI、Qwen3-ASR 或任何
媒体文件，也不能用于评价真实模型效果。

## 已实现

- `omni-training-v1` JSONL 协议和机器可读 JSON Schema；
- 支持纯音频且没有文本输入；
- 支持音频、视频、文本任意组合；
- ASR 转写可以放入 `text`，但必须标记为辅助通道，不能替代原始音频；
- 稀疏动作标注转换为每秒 `silence/response/delegate/interrupt` 监督；
- 假音频 Encoder 约产生 13 token/秒；
- 1 秒稳定提交、8 秒活动窗口、历史 token 计数和 reset；
- 混合长度 padding、attention mask、token 时间和无音频 batch；
- 小维度 NumPy bridge 和动作 head；
- 不覆盖历史 step checkpoint，并保存数据指纹、数据游标、优化器、scheduler、
  scaler 占位状态、随机种子、配置和代码 commit；
- checkpoint 恢复时拒绝已经变化的训练文件。

## 样本协议

每一行是一个独立 JSON 对象：

```json
{
  "schema_version": "omni-training-v1",
  "sample_id": "audio-only-user-question",
  "duration_ms": 3000,
  "provenance": {
    "dataset": "example",
    "version": "1",
    "source_uri": "local://example",
    "license_name": "CC0-1.0",
    "license_tier": "redistributable",
    "allows_training": true,
    "allows_modification": true,
    "allows_redistribution": true,
    "allows_commercial_use": true
  },
  "audio": [
    {
      "path": "/data/audio.wav",
      "start_ms": 0,
      "end_ms": 3000,
      "sample_rate": 16000,
      "num_samples": 48000,
      "channel": "user_audio"
    }
  ],
  "video": [],
  "text": [],
  "targets": [
    {
      "timestamp_ms": 2000,
      "action": "response",
      "text": "我听到了你的问题。"
    }
  ]
}
```

`audio`、`video`、`text` 至少有一种非空，但任何一种都可以缺失。未显式写出 target 的
每秒时间步自动监督为 `silence`。同一秒只能有一个动作 target。

每条样本必须记录许可证和来源。`allows_training=false` 的样本会在加载时被拒绝。
`research_only` 与 `redistributable` 可以进入不同实验，但正式 checkpoint 必须保留
使用过的数据指纹和 manifest；本原型暂不负责把不同许可层自动混训。

完整字段约束见：

- `schema/omni-training-v1.schema.json`
- `schema.py`

## 运行 CPU 假训练

在仓库根目录执行：

```powershell
.\.venv-local\Scripts\python.exe -m training.omni.train_fake `
  training\omni\examples\b2_fake_samples.jsonl `
  --config training\omni\configs\b2_cpu_fake.json `
  --output-dir training\omni\tmp\b2-fake-run
```

继续训练：

```powershell
.\.venv-local\Scripts\python.exe -m training.omni.train_fake `
  training\omni\examples\b2_fake_samples.jsonl `
  --config training\omni\configs\b2_cpu_fake.json `
  --output-dir training\omni\tmp\b2-fake-run `
  --resume
```

配置中的 `total_steps` 是目标总步数，不是本次追加步数。提高后再使用 `--resume`。

## 测试

```powershell
.\.venv-local\Scripts\python.exe -m unittest `
  training.omni.tests.test_schema `
  training.omni.tests.test_b2_fake_training
```

## 与真实模型的边界

假 Encoder 默认只用 8 维，假 bridge 默认投到 6 维，以便 CPU 测试快速完成。真实 B3
接口仍按已调查的候选维度设计：

```text
Qwen3-ASR Encoder 1024
  -> Qwen3-ASR projector 2048
  -> JoyAI audio bridge 4096
```

B2 已证明的是数据和状态流可以工作，不是以下事项：

- Qwen3-ASR Encoder 的真实环境声能力；
- 音频窗口编码与完整离线编码的数值等价性；
- JoyAI `inputs_embeds`、MRoPE 和 tokenizer 的真实兼容性；
- 2×4090 的峰值显存或吞吐；
- 真实 loss 下降、视觉语言遗忘或最终效果。

这些仍属于 B3/B4 GPU 验证。
