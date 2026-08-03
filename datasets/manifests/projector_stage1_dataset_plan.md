# Projector 第一阶段数据准备

第一阶段只做音频-文本对齐。JoyAI 的视频问答文本不是音频转写，因此 ActivityNet、YouCook2、Kinetics、WebVid、DiDeMo、NExT-QA 只保留给后续多模态阶段，不能直接作为这一阶段的正向音频文本监督。环境声也不纳入当前目标。

## 当前优先级

- `[√]` 建立只接受本地媒体的转换入口：`datasets/prepare_projector_stage1_manifest.py`
- `[√]` 固定单个媒体文件上限为 1,000,000,000 bytes；工具不会下载任何文件
- `[√]` 记录 provenance、split、原始记录标识和媒体 SHA-256
- `[ ]` P0：取得并审核 SpokenWOZ 小规模完整对话 pilot
- `[ ]` P0：取得 LibriSpeech dev-clean 小分片，作为最小可运行 audio-text smoke test
- `[ ]` P1：取得 AISHELL-4 的获准小分片，补充中文语音活动与重叠
- `[ ]` P1：用 MInDS-14 做小规模意图/音频管线回归，不把它当自然对话主集

## 输入格式

准备一个本地 `metadata.jsonl`，每行至少包含：

```json
{"audio_path":"C:/data/sample.wav","target_text":"transcript","duration_ms":1200,"sample_rate":16000,"num_samples":19200,"split":"train","source_record":"id"}
```

`target_text` 是模型要生成的文字；`input_text` 是可选的用户/上下文文字，不能把转写误放进 `input_text`。转换结果把目标写入 `metadata.assistant_target_text`，供 tokenizer/batch builder 构造 assistant labels；它不是模型输入的 user 文本。

再运行：

```powershell
python datasets/prepare_projector_stage1_manifest.py `
  --input datasets/manifests/projector_stage1_metadata.jsonl `
  --output datasets/manifests/projector_stage1_samples.jsonl `
  --provenance datasets/manifests/projector_stage1_provenance.json
```

输出可以直接由 `training.omni.schema.load_samples` 校验。缺失媒体、超过 1GB、无转写或字段不一致的记录会被跳过并写入 summary，不会进入训练集。

## 约束

本阶段不下载服务器文件、不使用 JoyAI 原始视频作为语音标签、不把未知停顿自动标成 `silence`，也不提交原始媒体、模型权重或凭据。真正开始 GPU 训练前，必须先用 50--500 条已核验样本完成单样本过拟合和 schema 回归。
