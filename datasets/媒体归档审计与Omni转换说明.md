# 媒体归档审计与 Omni 转换说明

## 目标

这套工具解决“JoyAI 只有 JSON、媒体以大 tar 提供”的衔接问题：

```text
媒体 tar + JoyAI JSON
  -> 文件名连接
  -> 选择性抽取
  -> ffprobe / ffmpeg 验收
  -> 每秒审计帧
  -> omni-training-v1.jsonl
```

它不会完整解压 tar，不加载模型，不使用 GPU，也不会把 MP4 中的原始音轨误认为用户
说出的 JoyAI 问题。

## 验收状态

每个媒体记录分别保存：

- tar 内成员名称和大小；
- 本地抽取路径和实际大小；
- ffprobe 成功或失败；
- 视频流数量、编码、分辨率和帧率；
- 音频流数量、编码、采样率、声道和时长；
- 音视频时长差和静音比例；
- JoyAI 最大标注时间是否位于实际媒体时长内；
- 全量解码是否通过；
- 每秒抽帧是否通过；
- `media_usable` 和 `audio_usable`。

其中：

- `media_usable` 表示媒体结构、视频流、时间戳和可选解码检查通过；
- `audio_usable` 只表示可探测音频流存在；
- 音频语义、静音比例、音画同步和问题回答正确性仍需后续自动分析或人工抽查。

## 安全边界

- 只抽取与 JoyAI 标注匹配且被选中的普通媒体文件；
- 不使用 tar 中的原始路径作为写入路径；
- 不调用 `extractall`；
- 抽取后核对字节数；
- 同名多成员不会自动选择，避免拿错媒体；
- 生成数据保留 `research_only` 与许可明细；
- 原始媒体和生成输出由 `.gitignore` 排除。

## 转换语义

CharadesEgo 转换后的模态含义为：

```text
video
  每秒审计帧

audio
  MP4 原始音轨，channel=environment_audio

text
  JoyAI 合成或整理的问题，channel=user_text

targets
  JoyAI 回答发生时刻，action=response
```

因此这种数据可以用于视觉交互和环境音探索，但不能直接作为“用户通过音频说出问题”
的语音对话训练数据。纯音频用户对话仍需要独立的语音数据构造流程。

## 本机依赖

脚本仅依赖 Python 标准库完成 tar 扫描、匹配和选择性抽取。真实音视频验收还需要：

```text
ffprobe
ffmpeg
```

当前本地 Windows 环境尚未检测到这两个命令。视频下载完成后，可以安装 FFmpeg，
或在执行时通过 `--ffprobe` 和 `--ffmpeg` 指向已有可执行文件。
