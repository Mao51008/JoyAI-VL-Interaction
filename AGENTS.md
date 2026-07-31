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

## 原生 Transformers 兼容分支

- `codex/native-transformers-audio` 用于驱动版本无法运行当前 vLLM/vLLM-Omni wheel 时的兼容后端。
- 备用 3090 服务器使用 `MAIN_BACKEND=transformers`、`ASR_BACKEND=transformers`、`TTS_BACKEND=transformers`；默认规划为主模型 GPU 0、ASR 和 TTS 共用 GPU 1，GPU 2 若被其他用户占用则禁止使用。
- 三套隔离环境分别为 `services/.venv-cu124`、`services/asr/.venv-cu124` 和 `services/tts/.venv-cu124`，由 `install/install-native-transformers-runtime.sh` 创建。不得把 CUDA 13、CUDA 12.9 nightly 或 vLLM-Omni 依赖混入这些环境。
- 备用服务器的 FFmpeg、FFprobe 和 SoX 位于 `/data/maoyy/miniforge3/bin`；启动音频服务时显式加入 `PATH`，不要修改系统安装。
- Transformers ASR 是整段请求转写；Transformers TTS 会在完整生成后分块发送 PCM，且不能中止已进入 `generate()` 的计算。不得将两者宣称为模型原生流式或完整打断验收结果。
- 主力 4090 服务器仍负责 vLLM/vLLM-Omni 的真实流式、取消、吞吐和长时间稳定性验收；兼容分支不得改变驱动或硬件。
