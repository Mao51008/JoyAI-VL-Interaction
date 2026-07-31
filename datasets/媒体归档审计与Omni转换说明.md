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

归档成员既可以是常见媒体扩展名，也可以是社区 CharadesEgo 归档采用的
`videos_pool/<source>/<video_name>` 无扩展名文件。无扩展名识别仅在 `videos_pool`
目录契约内启用；选择性抽取仍使用由标注名称生成的安全本地文件名。

## 转换语义

CharadesEgo 转换后的模态含义为：

```text
video
  每秒审计帧

audio
  媒体存在原始音轨时，channel=environment_audio

text
  JoyAI 合成或整理的问题，channel=user_text

targets
  JoyAI 回答发生时刻，action=response
```

因此这种数据可以用于视觉交互和环境音探索，但不能直接作为“用户通过音频说出问题”
的语音对话训练数据。纯音频用户对话仍需要独立的语音数据构造流程。

`CharadesEgo__001.tar` 的首轮 20 条抽样均没有音轨；这些条目只能提供视频和
`user_text`，转换器不能凭空构造 `environment_audio`。

## 服务器抽样验收结果

2026-07-31 在 `/data/maoyy` 范围内完成了 `CharadesEgo__001.tar` 的低优先级 CPU
审计，全程显式屏蔽 CUDA：

- 归档大小与 SHA-256 校验通过；
- 10,609 个媒体成员，10,590 个与当前标注连接，无名称碰撞；
- 20/20 `ffprobe` 成功；
- 20/20 完整解码成功；
- 20/20 PNG 抽帧成功；
- 16/20 标注时间位于视频时长内；
- 0/20 存在可探测音轨。

4 条时间戳越界样本为 `V51RVEGO_action_2`、`ZZROGEGO_action_1`、
`O0628EGO_action_1` 和 `V5IL7EGO_action_5`。其中
`O0628EGO_action_1` 仅 0.467 秒而标注到 24 秒，需要优先人工复核。

这只是单分片、20 条媒体的自动抽样验收。人工画面/问题/回答语义核对和许可审查仍未
完成，不能标记为正式训练数据全部验收通过。

## 本机依赖

脚本仅依赖 Python 标准库完成 tar 扫描、匹配和选择性抽取。真实音视频验收还需要：

```text
ffprobe
ffmpeg
```

当前本地 Windows 已安装 FFmpeg 8.1.2；服务器使用
`/data/maoyy/miniforge3/bin/ffmpeg` 和对应的 `ffprobe`。也可以在执行时通过
`--ffprobe`、`--ffmpeg` 指向其他已有可执行文件。

审计帧默认写为 PNG，避免 FFmpeg 8 对部分非全范围 YUV 输入的 MJPEG 输出限制。
当每秒采样对不足一秒的极短视频没有产生帧时，工具会回退抽取首帧。
