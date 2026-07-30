# JoyVL Interaction WebUI

> 原文档: [README.md](./README.md)

实时视觉语言模型交互 WebUI。默认情况下，它连接到本地 OpenAI 兼容 VLM 服务，用于本地摄像头或视频流交互预览。

## 环境设置

仓库级安装入口位于 `install/`，仓库级运行时入口是 `services/scripts/run.sh`。本 README 只说明单组件 WebUI 开发安装和启动。

需要 Python 3.12。

```bash
# 从仓库根目录运行
cd services/webui
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

默认后端地址为：

```text
http://127.0.0.1:8070/v1
```

请确保对应的 VLM 后端服务已经先启动。

## 启动

```bash
source ../.venv/bin/activate
./scripts/start_server.sh
```

在浏览器中打开：

```text
https://localhost:8099
```

如果浏览器提示自签名证书警告，请继续访问该站点。如果证书文件缺失，请先生成：

```bash
./scripts/generate_cert.sh
```

## 常用端口

```bash
# 默认脚本：WebUI 8099，后端 8070
source ../.venv/bin/activate
./scripts/start_server.sh

# WebUI 8090，后端 8070
./scripts/start_server.sh --port 8090 --api-base http://127.0.0.1:8070/v1

# WebUI 8091，后端 8071
./scripts/start_server.sh --port 8091 --api-base http://127.0.0.1:8071/v1
```

## 停止

```bash
./scripts/stop_server.sh
```

## 持续音频入口

启动视频流后，浏览器会自动采集单声道 PCM16 音频，并以 40 ms 小包发送到：

```text
GET /ws/audio-ingress?session_id=<session-id>
```

每包包含 session ID、序号、浏览器单调时钟时间戳、采样率、sample 数量、
`voice_active` 标志和 PCM16 数据。服务端默认只保留最近 30 秒音频，避免长时间
运行导致内存持续增长。

查看单个会话统计：

```text
GET /api/audio-ingress/status?session_id=<session-id>
```

## Omni 多模态时间轴

A4 将音频转写、环境声音、视频采样帧、用户问题、模型动作和 TTS 播放区间统一
映射到服务器单调时钟。查询最近事件：

```text
GET /api/timeline?session_id=<session-id>&lookback_seconds=10
```

查询固定 tick 或高优先级事件产生的多模态快照：

```text
GET /api/omni/snapshots?session_id=<session-id>&limit=20
```

默认每秒生成一次最近 10 秒快照，危险声音会立即生成 `priority` 快照。时间轴
默认最多保留 120 秒/2000 个事件，快照历史最多 256 份。配置项：

```bash
OMNI_TICK_SECONDS=1
OMNI_LOOKBACK_SECONDS=10
OMNI_URGENT_PRIORITY=80
OMNI_TIMELINE_SECONDS=120
OMNI_TIMELINE_MAX_EVENTS=2000
OMNI_MAX_SNAPSHOTS=256
```

## Omni 无 GPU 决策模式

A5 提供本地假模型模式，用于验证非回合制决策链路而不启动 GPU 或真实模型：

```bash
OMNI_DECISION_MODE=fake
```

该模式按多模态快照组织视频、环境声音、ASR、用户问题、TTS 播放状态和最近动作。
普通无事件场景保持沉默；用户请求或稳定语音触发响应；火警、烟雾警报、爆炸和玻璃破碎
等高优先级事件可绕过普通冷却；TTS 播放期间检测到用户开口会产生
`</interrupt>` 决策。响应具备冷却与重复内容抑制，决策记录会写回统一时间线，
并通过 WebSocket 的 `omni_decision` 消息发送给浏览器。

此模式只完成软件侧和假模型验收。真实 JoyAI-VL 的效果和 GPU 延迟仍待 GPU0 可用后验证。

## TTS generation 与打断

A6 为每次浏览器 TTS 播放分配独立 `generation_id`。服务端按会话维护当前活动生成，
`stop` 消息只取消匹配的 generation，避免迟到的旧请求误停新语音。Omni 决策产生
`</interrupt>` 时，会分别执行以下动作：

- 通知浏览器立即停止当前音源并清空排队的 PCM；
- 取消服务端正在转发的 TTS generation；
- 取消该会话当前正在执行的 VLM 请求。

浏览器在停止前会回报已播放毫秒数，并根据已接收音频进度估算已播放文本前缀；
这些信息写入统一时间线，后续决策上下文优先使用已播放内容，而不是把完整计划文本
误认为已经播出。该功能已有无 GPU 自动化测试，真实浏览器打断延迟和外放插话仍待
里程碑 A 集中验收。

## TTS 回声参考过滤

A7 在浏览器 acoustic echo cancellation 之外增加服务端文本参考过滤。连续 ASR 的
partial、stable 和 final 文本会与当前或刚结束的 TTS 文本比较，高相似结果携带
`likely_tts_echo`、相似度和参考 `generation_id`，并写入时间线。决策门控忽略纯回声，
但与 TTS 不同的新增用户内容仍会触发 barge-in；麦克风始终保持开启。

可调参数：

```bash
TTS_ECHO_SIMILARITY_THRESHOLD=0.82
TTS_ECHO_MIN_TEXT_CHARS=4
TTS_ECHO_TAIL_SECONDS=1.5
```

这是无需 GPU 的文本参考抑制，不替代 WebRTC AEC 或声学参考信号消除。扬声器、房间混响
和耳机条件下的真实效果仍需在里程碑 A 设备验收中测试。

## A8 无 GPU 场景验收

`tests/test_a8_scenarios.py` 使用假响应模型和合成时间线事件覆盖安静办公室、纯视觉火灾、
纯音频火警、烟雾与警报融合、变化中的 ASR partial、用户插话、危险事件抢占、回声中
真实插话、重叠说话不确定性，以及虚拟 30 分钟长流。视觉假事件采用
`modality="video"`、`kind="visual_event"` 和 `label=fire|flame|smoke|explosion`。

这套测试不启动 GPU，也不代表真实 JoyAI-VL 已能从视频识别火灾。真实视频、麦克风、
扬声器、ASR、CLAP、JoyAI-VL 和完整 30 分钟实时稳定性将在里程碑 A 集中验收。

可以用环境变量调整缓冲长度：

```bash
export AUDIO_INGRESS_BUFFER_SECONDS=30
```

持续入口的环形缓冲现在可以按最近若干秒构造带时间戳和 VAD 状态的音频窗口。
A3.1 已加入 400 ms 滑动窗口调度器、stable prefix、`speech_start`、
`speech_partial`、`speech_stable`、`speech_final`、`speech_end` 事件，以及可插拔的环境声音检测接口。
浏览器能够直接显示调度器发回的实时转写。

真实 Qwen3-ASR 客户端已在 A3.2 接入。启动 WebUI 前设置
`CONTINUOUS_ASR_ENABLED=1` 后，它会把滑动窗口封装为 WAV 并请求
`/v1/audio/transcriptions`；Qwen 返回的语言元数据会被清理，异常重复输出也会
被过滤。ASR 暂时不可用时会产生 `asr_error`，后续窗口继续自动重试，不影响
视频和手动语音功能。环境声音分类器将在 A3.3 注入同一事件流。

```bash
CONTINUOUS_ASR_ENABLED=1 \
CONTINUOUS_ASR_URL=http://127.0.0.1:8993/v1/audio/transcriptions \
CONTINUOUS_ASR_MODEL=Qwen/Qwen3-ASR-1.7B \
bash services/webui/scripts/start_server.sh
```

同时启用 A3.3 CLAP 环境声音检测：

```bash
AUDIO_EVENT_DETECTOR=clap \
AUDIO_EVENT_CLAP_MODEL=/data/maoyy/models/laion/clap-htsat-unfused \
AUDIO_EVENT_CLAP_DEVICE=cpu \
AUDIO_EVENT_CONFIDENCE_THRESHOLD=0.4 \
AUDIO_EVENT_COOLDOWN_SECONDS=2 \
CONTINUOUS_ASR_ENABLED=1 \
bash services/webui/scripts/start_server.sh
```

CLAP 支路会把 16 kHz 输入重采样到 48 kHz。危险事件会携带相对置信度；
0.4 是小规模火警/普通语音样本上的原型阈值，不应直接作为生产安全阈值。
可先查看一个 WAV 对全部候选类别的评分：

```bash
PYTHONPATH=services/webui/src services/.venv/bin/python \
  services/webui/tools/audio_event_smoke.py <test.wav> \
  --model /data/maoyy/models/laion/clap-htsat-unfused
```

用 16 kHz、单声道、PCM16 WAV 验证连续链路：

```bash
PYTHONPATH=services/webui/src services/.venv/bin/python \
  services/webui/tools/continuous_asr_smoke.py <test.wav> \
  --url wss://127.0.0.1:8099/ws/audio-ingress
```
