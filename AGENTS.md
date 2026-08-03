# 仓库指南

## 项目结构与模块组织

- `services/webui/` 包含 Python WebRTC/WebUI 应用；主要 Omni 编排代码位于 `src/joy_interaction_webui/`，测试位于 `services/webui/tests/`。
- `services/webinfer/` 提供兼容 OpenAI API 的 VLM 适配器和 vLLM 启动脚本。`services/asr/`、`services/tts/` 和 `services/background-agent/` 是可独立运行的可选服务。
- `training/omni/` 包含 `omni-training-v1` schema、Fake Encoder/Projector 原型、示例配置和 CPU 训练测试。
- `datasets/` 包含数据审计、归档抽取、格式转换和来源追踪工具。生成的审计结果不要放入源码模块。
- `install/` 管理环境安装和模型下载脚本；`container/`、`docs/`、`doc/` 和 `img/` 保存部署、文档与图片资源。

## 构建、测试与开发命令

项目使用 Python 3.12 和 `uv`。除非组件 README 另有说明，否则从仓库根目录执行命令。

```bash
./install/install.sh --with-all       # 创建 services/.venv 并安装全部适配器
./services/scripts/run.sh minimal     # 启动 webinfer 和 WebUI
./services/scripts/run.sh all         # 启动完整服务栈
./services/scripts/stop.sh            # 停止仓库管理的服务
uv run --project services/webui pytest -q
uv run --project services/asr pytest -q
uv run --project services/tts pytest -q
```

运行 Omni CPU 原型测试：

```bash
python -m unittest training.omni.tests.test_schema training.omni.tests.test_b2_fake_training
```

模型下载和真实 GPU 验证必须显式执行，不应成为轻量单元测试的前置条件。

## 代码风格与命名约定

Python 使用 4 空格缩进；公共接口应提供类型标注，非直观逻辑使用简短 docstring。Black 和 Ruff 的行宽均为 100。模块和函数使用 `snake_case`，类使用 `PascalCase`，环境变量使用大写形式。保持现有 `src/` 包布局。Shell 脚本应遵循现有约定，例如 `set -euo pipefail` 和可由环境变量覆盖的默认配置。

Markdown 任务清单中，未完成项使用 `[ ]`，已完成项统一使用 `[√]`，不得使用 `[x]` 或 `[X]`；延期项可使用 `[延后]` 并说明恢复条件。

## 测试规范

Pytest 按 `test_*.py`、`Test*` 和 `test_*` 规则发现测试；异步测试使用 `pytest-asyncio` 的 auto 模式。测试应放在受影响服务旁边。修改时间线、决策、打断或回声抑制逻辑时，必须增加对应的场景回归测试。涉及 GPU、模型权重、麦克风、扬声器或长时间运行的验证，应明确标记为真实环境测试，不能用来替代单元测试。

## 提交与合并请求规范

近期提交采用简短、祈使式中文摘要，例如 `实现A6无GPU语音打断链路`。每个提交只处理一项明确行为。合并请求应说明改动内容、受影响服务、执行的测试及结果、GPU/设备假设，以及模型或数据集要求。WebUI 改动应附截图，并关联相关 Issue。禁止提交模型权重、凭据、未经许可的原始媒体或运行日志；数据来源和许可证信息必须写入 provenance manifest。

## 用户补充

不要改驱动，不要改硬件，服务器端不能动data/maoyy目录以外的内容
主力服务器IP：maoyy@10.11.12.30,CUDA13,4张4090
备用服务器IP: maoyy@10.11.12.96,CUDA12.4,4张3090
每次完成新增的功能或者修复了bug请务必同步到github。

## 服务器与工作分支唯一映射

| 服务器 | 硬件与 CUDA | 唯一部署分支 | 后端用途 |
| --- | --- | --- | --- |
| 主力服务器 `maoyy@10.11.12.30` | `4×RTX 4090`、CUDA 13 | `feature/continuous-omni` | 原始 vLLM/vLLM-Omni；真实流式、取消、吞吐、设备与训练验收 |
| 备用服务器 `maoyy@10.11.12.96` | `4×RTX 3090`、CUDA 12.4 | `codex/native-transformers-audio` | Transformers 兼容后端；只做兼容部署和非原生流式验证 |

- 每次连接服务器后、同步代码或启动服务前，必须先运行 `git branch --show-current` 和 `git status --short` 核对分支与工作树。
- 4090 主力服务器不得切换、拉取或部署 `codex/native-transformers-audio`；3090 备用服务器不得用 `feature/continuous-omni` 的 CUDA 13/vLLM-Omni 环境覆盖其 CUDA 12.4 兼容环境。
- 需要共享修复时，应先提交到 GitHub，再通过 cherry-pick 或合并把提交同步到目标分支；禁止通过在服务器上直接切换到另一服务器专用分支来共享代码。
- 如果服务器实际分支与上表不一致，立即停止部署，先保留工作树改动并恢复正确分支，不得继续启动模型服务。

## 原生 Transformers 兼容分支

- `codex/native-transformers-audio` 仅用于备用 3090 服务器在驱动版本无法运行当前 vLLM/vLLM-Omni wheel 时部署 Transformers 兼容后端；不得在主力 4090 服务器上切换或部署该分支。
- 主力 4090 服务器使用 `feature/continuous-omni` 及其已验证的原始 vLLM/vLLM-Omni 部署基线；3090 兼容分支的实现和环境不得作为主力服务器部署依据。
- 备用 3090 服务器使用 `MAIN_BACKEND=transformers`、`ASR_BACKEND=transformers`、`TTS_BACKEND=transformers`；默认规划为主模型 GPU 0、ASR 和 TTS 共用 GPU 1，GPU 2 若被其他用户占用则禁止使用。
- 三套隔离环境分别为 `services/.venv-cu124`、`services/asr/.venv-cu124` 和 `services/tts/.venv-cu124`，由 `install/install-native-transformers-runtime.sh` 创建。不得把 CUDA 13、CUDA 12.9 nightly 或 vLLM-Omni 依赖混入这些环境。
- 备用服务器的 FFmpeg、FFprobe 和 SoX 位于 `/data/maoyy/miniforge3/bin`；启动音频服务时显式加入 `PATH`，不要修改系统安装。
- Transformers ASR 是整段请求转写；Transformers TTS 会在完整生成后分块发送 PCM，且不能中止已进入 `generate()` 的计算。不得将两者宣称为模型原生流式或完整打断验收结果。
- 主力 4090 服务器仍负责 vLLM/vLLM-Omni 的真实流式、取消、吞吐和长时间稳定性验收；兼容分支不得改变驱动或硬件。

## 服务器部署约束

- 服务器部署时最多使用两张 GPU，只允许使用空闲的服务器，以 GPU 0 和 GPU 1 为例。
- GPU 0 仅部署主视觉模型；不要部署或启动摘要模型。
- ASR 和 TTS 必须共同部署在 GPU 1，不得分别占用额外 GPU。
- 启动前先检查 GPU 0、GPU 1 的实时占用，并按单请求验证结果控制显存预算；不得默认启动仓库中会占用摘要模型 GPU 的完整 WebInfer 配置。

## Omni 第一阶段训练约束

- 第一阶段定义为音频对齐的 `projector-only` 训练：冻结 Qwen3-ASR Encoder、其原生 projector 以及 JoyAI-VL 全部参数，只训练连接 ASR 音频特征和 JoyAI token embedding 的 `audio_projector`。
- 当前候选维度 `2048 -> 4096` 仅用于配置占位。开始真实训练前必须运行 `training.omni.probe_models` 并依据服务器本地模型的 config 和真实 forward shape 确认维度，禁止仅凭文档硬编码后直接长训。
- 冻结 JoyAI-VL 参数时，不得用 `torch.no_grad()` 包裹语言模型 forward；必须保留从语言模型 loss 到 `audio_projector` 输入的梯度图。冻结 Audio Encoder 的特征提取可以使用 `torch.no_grad()` 或离线特征缓存。
- 第一阶段监督目标是 assistant 文本 token 的生成 loss。system、user、音频占位符和 padding 的 label 必须为 `-100`；暂不把 `silence/response/delegate/interrupt` 动作分类 head 混入该阶段。
- optimizer 只能接收 `requires_grad=True` 的 projector 参数。训练启动和单元测试必须校验可训练参数名、projector 非零有限梯度、冻结参数无梯度且训练步后权重不变。
- checkpoint 默认只保存 projector 权重以及 optimizer、scheduler、scaler、数据指纹、基础模型路径/revision/config hash；不得复制或提交基础模型权重。
- GPU 验证必须先通过单样本过拟合，再执行小批量 BF16 显存与稳定性测试。模型下载、真实权重加载和 GPU 测试必须显式执行，不得成为 CPU 单元测试前置条件。

## Benchmark 评测例外与准备

- Benchmark/离线性能评测不受“服务器部署最多两张 GPU”的限制，但必须确认服务器四张卡均为空闲，并在评测记录中写明 GPU 型号、驱动、CUDA、vLLM/vLLM-Omni、模型 revision 和启动参数。
- 需要同时启用主模型和摘要模型时，默认使用 GPU 0、1、2 做主模型张量并行（`MAIN_GPU=0,1,2`、`TENSOR_PARALLEL_SIZE=3`），GPU 3 启动摘要模型（`SUMMARY_GPU=3`）。只有评测原始主模型且不启用摘要时，才使用四卡主模型张量并行（`MAIN_GPU=0,1,2,3`、`TENSOR_PARALLEL_SIZE=4`）。
- Benchmark 默认提高上下文预算：主模型 `MAX_MODEL_LEN=262144`，摘要模型 `SUMMARY_MAX_MODEL_LEN=131072`；若模型或显存不支持，必须记录实际值，不得静默回退。
- 评测前必须准备：固定模型路径和 revision、公开 benchmark 数据集及许可证/provenance、视频解码依赖（FFmpeg/FFprobe）、统一的图像分辨率/采样 FPS/最大帧数、统一 prompt 和 generation 参数、结果目录、原始请求/响应日志，以及可复现的评测脚本版本。
- 评测至少分离记录四种条件：原始单帧、原始多帧/视频、项目实时 prompt、启用 chunk/摘要/长期记忆的完整链路；不得用完整链路分数替代原始模型性能。
- 除准确率外，必须记录 TTFT、端到端延迟 P50/P95/P99、tokens/s、峰值显存、吞吐、请求取消延迟、错误率、上下文截断率，以及实时场景的静默误报率、漏报率和事件响应延迟。
- 评测输出不得写入 `data/maoyy` 以外的服务器目录，不得提交模型权重、凭据、原始媒体或运行日志；大体积结果只保留摘要和 provenance manifest。

## 主力服务器 RTSP 本地视频推流

- 主力 4090 服务器的 RTSP 测试运行目录为 `/data/maoyy/joyai-runtime/rtsp`，不得把 MediaMTX、配置、PID 或日志写到 `/data/maoyy` 之外。
- MediaMTX 使用独立 conda 前缀 `/data/maoyy/joyai-runtime/rtsp/mediamtx-env`，不要安装到模型使用的 `uv` 虚拟环境，也不要修改系统安装。当前验证包为 conda-forge `mediamtx=1.15.6`。
- FFmpeg 和 FFprobe 使用 `/data/maoyy/miniforge3/bin/ffmpeg` 与 `/data/maoyy/miniforge3/bin/ffprobe`。
- 当前现成测试视频为 `/data/maoyy/videos/-CM30ROUHEs.mp4`，启动脚本为 `/data/maoyy/joyai-runtime/rtsp/start.sh`，MediaMTX 配置为 `/data/maoyy/joyai-runtime/rtsp/mediamtx.yml`。
- 启动推流：`bash /data/maoyy/joyai-runtime/rtsp/start.sh`。脚本循环读取视频，转码为 H.264 + AAC，并通过 TCP 发布到 `rtsp://127.0.0.1:8554/joyai-cm30rouhes`。
- WebUI 与 MediaMTX 在同一主力服务器上运行时，RTSP 输入使用 `rtsp://127.0.0.1:8554/joyai-cm30rouhes`；其他机器直接拉流时使用 `rtsp://10.11.12.30:8554/joyai-cm30rouhes`。
- 验证命令：`/data/maoyy/miniforge3/bin/ffprobe -v error -rtsp_transport tcp -show_streams rtsp://127.0.0.1:8554/joyai-cm30rouhes`。
- 停止时读取 `/data/maoyy/joyai-runtime/rtsp/ffmpeg.pid` 和 `mediamtx.pid`，只终止对应 PID；禁止使用可能误伤其他用户进程的宽泛 `pkill ffmpeg` 或 `pkill mediamtx`。
- WebUI RTSP 模式必须让视频帧、文本输入、ASR 和 TTS 使用浏览器当前的同一个 session。测试连接按钮创建的临时 session 或独立 `/api/rtsp/start` session 不得与浏览器正式推理 session 混用。
