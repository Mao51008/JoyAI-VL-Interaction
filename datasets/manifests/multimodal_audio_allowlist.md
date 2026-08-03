# JoyAI 多模态音频逐文件清单

生成文件：`datasets/audit_output/multimodal_audio_allowlist.jsonl`。

## 已核验样本

| 来源 | 抽样文件 | 有音频流 | 排除静音 | 结论 |
|---|---:|---:|---:|---|
| ActivityNet | 8 | 8 | 0 | 可作为多模态辅助候选 |
| YouCook2 | 10 | 10 | 0 | 可作为多模态辅助候选 |
| Kinetics-400 | 10 | 9 | 1 | 9 条进入正向 allowlist，1 条仅保留为静音负样本 |
| **合计** | **28** | **27** | **1** | **27 条已确认有音频流** |

每条记录包含来源、`video_name`、绝对媒体路径、文件大小、SHA-256、音频 codec、采样率、声道数、音频时长以及 `allowlist_status`。

## 语义限制

这些视频的音频是 `original_scene_audio_not_user_speech`：不能把它们当成用户语音转写，也不能用来替代 SpokenWOZ/AISHELL 等 audio-text 数据。它们适合后续视觉+场景音频联合输入、音视频同步和缺失模态鲁棒性训练。

WebVid、DiDeMo、NExT-QA 当前仍是候选，不进入本清单：必须先完成各自官方媒体索引映射，再对 JoyAI 实际 `source + video_name` 逐文件核验。
