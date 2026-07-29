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
