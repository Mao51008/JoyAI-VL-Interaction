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

该持续入口在 A2 阶段只负责采集、缓存和统计，不会把无限音频直接送入当前
final-only ASR；滑动窗口 partial ASR 将在 A3 接入。
