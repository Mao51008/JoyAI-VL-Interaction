# JoyAI-VL → 非回合制 Omni 改造步骤清单

## 0. 先明确目标

目标不是简单地给现有接口增加一个音频文件参数，而是让系统具备以下行为：

- [ ] 摄像头和麦克风始终处于连续输入状态。
- [ ] 没有明确用户提问时，系统仍会持续感知视频、语音和环境声音。
- [ ] 每个推理时刻自主选择 `沉默 / 发言 / 委托 / 打断当前输出`。
- [ ] 系统播放语音时仍继续听、继续看，而不是停止输入等待自己说完。
- [ ] 用户插话或出现更高优先级事件时，可以中止正在生成和播放的语音。
- [ ] 能理解环境声音，例如警报、爆炸、玻璃破碎、婴儿哭声，而不只是把人声转成文字。
- [ ] 音频、视频、文本和模型输出都使用同一条时间轴。

这里的“非回合制”仍然需要内部状态机和调度器，只是不再以“用户说完一句 → 模型回答一句”作为处理边界。

## 1. 重要结论与推荐路线

当前仓库已经包含：

- `services/webinfer`：持续视频理解、主动沉默/发言、chunk 和记忆。
- `services/webui`：WebRTC 视频、浏览器麦克风、ASR/TTS 桥接。
- `services/asr`：把一段 PCM16 音频转写成文本。
- `services/tts`：把模型文本回复流式合成为 PCM16。

但当前 ASR 路线只是：

```text
声音 → ASR 文本 → 当作用户问题 → VLM
```

它会丢失音色、情绪、语速、非语言声音和细粒度时间信息，因此只能算“带语音接口的多模态系统”，不能算声音原生的 Omni 模型。

推荐分三步推进：

1. **里程碑 A：模块化非回合制全双工系统**
   - 复用现有 VLM、ASR、TTS。
   - 先把持续监听、自主触发、可打断、回声处理和统一时间轴做正确。
   - 这是最优先、风险最低、最容易验证的一步。
2. **里程碑 B：声音原生输入**
   - 给 JoyAI-VL 增加音频编码器和投影/融合层。
   - 使用音频 embedding，而不是只使用 ASR 文本。
   - 需要训练，不是只改推理代码。
3. **里程碑 C：端到端语音输出**
   - 模型直接生成语音 codec token，或者采用文本与语音双头。
   - 在完成 A、B 之前，不建议优先做这一阶段。

---

## 2. 里程碑 A：先做出模块化非回合制全双工

### A1. 冻结当前可复现基线

- [√] 保存当前可工作的 vLLM、Adapter 和 WebUI 启动命令。

  启动 vLLM（仅使用已授权的 GPU 0）：

  ```bash
  VLLM_USE_FLASHINFER_SAMPLER=0 MAIN_GPU=0 MODEL_PATH=/data/maoyy/models/jdopensource/JoyAI-VL-Interaction PYTHON_BIN=/data/maoyy/JoyAI-VL-Interaction/services/.venv/bin/python MAX_MODEL_LEN=8192 MAIN_GPU_MEMORY_UTILIZATION=0.90 TENSOR_PARALLEL_SIZE=1 MAIN_SMOKE_ENABLE=0 bash services/webinfer/scripts/start_model.sh
  ```

  启动 Live Adapter（当前验证配置为 `--chunk 4`）：

  ```bash
  services/.venv/bin/python services/webinfer/live_adapter.py --main-api-base http://127.0.0.1:7060/v1 --main-model jdopensource/JoyAI-VL-Interaction --disable-summarizer --host 127.0.0.1 --port 8070 --allowed-local-image-roots /data/maoyy --chunk 4 --no-force-silence-before-query
  ```

  启动 WebUI：

  ```bash
  bash services/webui/scripts/start_server.sh
  ```

  启用 ffmpeg 并推流成 RTSP：

  ```bash
  ffmpeg -re -stream_loop -1 -i /data/maoyy/videos/Fire_montage.mp4 -vf "scale='min(1280,iw)':-2" -c:v libx264 -preset veryfast -tune zerolatency -b:v 2500k -an -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/fire1
  ```

- [√] 记录 Python、PyTorch、vLLM、驱动和模型版本。

  - 操作系统：Ubuntu 22.04
  - Python：3.12.13（Clang 22.1.3 构建）
  - PyTorch：2.11.0+cu130
  - PyTorch CUDA runtime：13.0
  - vLLM：0.22.0
  - Transformers：5.14.1
  - NVIDIA Driver：595.84
  - `nvidia-smi` 显示的驱动最高支持 CUDA：13.2（不等同于 PyTorch CUDA runtime）
  - GPU：4 × NVIDIA GeForce RTX 4090，每张 24564 MiB；当前仅使用已授权的 GPU 0
  - 主模型：`jdopensource/JoyAI-VL-Interaction`
  - 项目虚拟环境：`/data/maoyy/JoyAI-VL-Interaction/services/.venv`

- [√] 保存当前真实图片、RTSP 视频流冒烟测试。
- [√] 建立新的 Git 分支，例如 `feature/continuous-omni`。
- [√] 保持当前纯视频链路测试始终可运行，避免音频改动破坏已有功能。

验收：

- [√] 新分支开始修改前，纯视频自主响应仍能稳定运行。

### A2. 把麦克风改成真正的连续音频源

主要位置：

- `services/webui/src/joy_interaction_webui/static/index.html`
- `services/webui/src/joy_interaction_webui/asr.py`
- `services/asr/asr_adapter.py`

任务：

- [√] 浏览器持续发送单声道 PCM16，而不是只在用户按下按钮后录一段。
- [√] 使用固定小包，每包 40 ms（16 kHz、640 samples）。
- [√] 每个音频包携带 `session_id`、序号和单调递增时间戳。
- [√] 服务端建立有界环形缓冲区，默认只保留最近 30 秒原始音频。
- [√] 加入音频丢包、乱序、重连和客户端/服务端时钟漂移统计。
- [√] VAD 只作为 `voice_active` 包特征，不会因静音停止持续采集。

注意：当前 `services/asr/asr_adapter.py` 会累计音频并在 final 后执行一次完整转写。要实现连续系统，需要改为滑动窗口或真正的流式 ASR。

A2 实现说明：

- 持续音频入口：`GET /ws/audio-ingress?session_id=...`
- 运行状态：`GET /api/audio-ingress/status?session_id=...`
- 浏览器在视频流进入运行状态后自动启动麦克风，视频停止时关闭。
- 持续入口暂时不调用 ASR，避免当前 final-only ASR 无限累计 PCM；A3 再从环形缓冲区执行滑动窗口 partial ASR。
- 原有按键语音输入仍然保留，并通过 clone 复用持续麦克风轨道。
- 自动化测试：`services/webui/tests/test_audio_ingress.py`

验收：

- [√] 30 分钟数据量模拟测试确认环形缓冲始终只保留最后 30 秒。
- [√] 音频入口 WebSocket 断开后采用指数退避自动重连，并有集成测试覆盖。
- [ ] 在真实浏览器中连续运行 30 分钟，确认进程内存稳定。
- [ ] A3 接入流式 ASR 后，验证 ASR 服务重启时自动恢复。
- [ ] 音频时间戳与服务器时间轴没有明显漂移。

### A3. 增加流式 ASR 与非语言声音检测

先保留两个独立通道：

```text
人声 → 流式 ASR → partial/final transcript
环境声 → Audio Event Detector → alarm/fire/glass_break/...
```

任务：

- [ ] ASR 每 200–500 ms 给出 partial transcript。
- [ ] ASR 提供 stable prefix，避免每次重写整句话。
- [ ] 增加语音活动概率、说话开始和说话结束事件。
- [ ] 增加环境声音分类器；原型可先使用 BEATs、CLAP 或同类音频模型。
- [ ] 环境声音结果包含类别、置信度和起止时间。
- [ ] 不要把低置信度环境声音直接当成事实，应作为模型上下文中的不确定观测。

建议的统一事件格式：

```json
{
  "session_id": "demo",
  "modality": "audio",
  "start_sec": 12.40,
  "end_sec": 13.20,
  "kind": "audio_event",
  "label": "fire_alarm",
  "confidence": 0.91
}
```

验收：

- [ ] 用户仍在说话时，UI 能看到逐步更新的 partial transcript。
- [ ] 播放警报、玻璃破碎等无语音音频时，即使 ASR 为空也会产生音频事件。

### A4. 建立统一的多模态时间轴

建议新增一个独立模块：

```text
services/omni/orchestrator.py
```

或者先在 `live_adapter.py` 外建立一个小型服务，不要继续把所有逻辑堆进 Adapter。

统一维护：

- 最近视频帧及时间戳。
- 最近原始音频窗口。
- ASR partial/final 文本。
- 环境声音事件。
- 用户问题、模型动作和 TTS 播放区间。
- 哪些音频来自用户，哪些可能是系统自己的扬声器回声。

任务：

- [ ] 定义统一的 `TimelineEvent` 数据结构。
- [ ] 所有时间使用同一单调时钟，墙上时间只用于日志展示。
- [ ] 按固定 tick（建议先用 500 ms 或 1 秒）生成一次多模态快照。
- [ ] 允许突发高优先级事件绕过普通 tick，立即触发一次决策。
- [ ] 为过期事件设置窗口和淘汰策略，禁止上下文无限增长。

验收：

- [ ] 能回放一次会话，并按时间顺序重建视频、声音、文本和模型动作。

### A5. 将 `live_adapter.py` 从视频决策扩展为多模态决策

主要位置：

- `services/webinfer/live_adapter.py`
- 新增的 `services/omni/orchestrator.py`

每次决策输入建议包含：

```text
[Video observations: 10.0s–14.0s]
[Audio events: 10.0s–14.0s]
[Partial speech transcript]
[Stable speech transcript]
[System speaking state]
[Recent actions and memory]
```

任务：

- [ ] 保持 `--no-force-silence-before-query`，没有用户提问也执行推理。
- [ ] 将 ASR partial、final 和音频事件注入模型上下文。
- [ ] 把动作协议扩展为：
  - `</silence>`
  - `</response> ...`
  - `</delegate> ...`
  - `</interrupt>`（停止当前语音）
- [ ] 增加主动发言冷却时间和内容去重，避免每秒重复报警。
- [ ] 增加事件优先级，例如生命安全事件可以打断普通回答。
- [ ] 把“是否需要发言”和“具体说什么”拆成两个逻辑阶段进行实验。
- [ ] 所有决策记录输入快照、动作、置信度、延迟和触发原因。

验收：

- [ ] 没有人讲话时，看到火灾或听到警报能够主动输出。
- [ ] 普通稳定场景大部分时间保持沉默。
- [ ] 同一持续事件不会每秒重复完全相同的警告。

### A6. 实现真正的边听边说与打断

主要位置：

- `services/webui/src/joy_interaction_webui/tts.py`
- `services/tts/tts_adapter.py`
- WebUI 浏览器端音频播放代码
- 新增的 Omni 调度器

任务：

- [ ] TTS 播放期间麦克风和视频输入保持运行。
- [ ] 每次 TTS 生成分配 `generation_id`。
- [ ] TTS 服务支持 `cancel(generation_id)`。
- [ ] 浏览器收到取消信号后立即清空尚未播放的音频 buffer。
- [ ] 用户插话或高优先级事件出现时，在 200–500 ms 内停止旧回复。
- [ ] 新决策能够使用被打断前已经说出的文本，而不是把整段计划文本都当成已播放。
- [ ] 区分“停止播放”“停止 TTS 生成”“取消语言模型生成”三个动作。

验收：

- [ ] 系统说话时用户插话，旧语音可以被中止。
- [ ] 中止后系统能根据新输入继续，而不是重头重复旧回答。

### A7. 解决系统自声回流

全双工最容易失败的地方不是模型，而是系统把自己的 TTS 当成用户输入。

任务：

- [ ] 开启浏览器 WebRTC 的 acoustic echo cancellation。
- [ ] 保存当前正在播放的 TTS PCM 或文本及播放时间区间。
- [ ] 在服务端增加回声相似度检测或参考信号消除。
- [ ] TTS 播放期间降低与 TTS 内容高度相似的 ASR partial 的优先级。
- [ ] 不要简单地在 TTS 播放时关闭麦克风，否则会失去真正的 barge-in。
- [ ] 使用扬声器和耳机两种场景分别测试。

验收：

- [ ] 模型不会持续回答自己刚刚说出的内容。
- [ ] 戴耳机和外放时都能正常插话。

### A8. 里程碑 A 的测试集

- [ ] 无人说话、普通办公室视频：应基本沉默。
- [ ] 火灾视频、无声音：应主动警告。
- [ ] 黑屏、只有火灾警报声音：应主动警告。
- [ ] 视频出现烟雾，同时声音出现警报：应融合两个证据。
- [ ] 用户说话中途修改问题：partial transcript 不应触发大量错误回答。
- [ ] 系统说话时用户插话：应中止旧回复。
- [ ] 系统说话时出现高优先级警报：应中止普通回复并报警。
- [ ] 两个人重叠说话：记录当前能力边界，不把错误转写伪装成确定事实。
- [ ] 30 分钟长流：内存、上下文和延迟保持稳定。

推荐指标：

- 主动事件召回率。
- 每小时误报次数。
- 事件发生到首次发言的延迟。
- ASR partial 延迟和 stable prefix 延迟。
- 打断延迟。
- 自声误识别率。
- 实时系数和 GPU 峰值显存。

---

## 3. 里程碑 B：把声音变成模型原生输入

完成里程碑 A 后，再开始修改模型结构。

### B1. 先确认训练代码和许可证

- [ ] 确认 JoyAI-VL 权重目录中的 `config.json`、`auto_map` 和实际模型类。
- [ ] 确认仓库是否提供完整预训练/微调代码；部署代码不等于训练代码。
- [ ] 确认视觉主干、语言模型、对话模板和位置编码实现。
- [ ] 确认模型及训练数据许可证允许修改和再训练。
- [ ] 若官方没有训练代码，先基于 Transformers 自定义模型类做原型，不要先改 vLLM。

### B2. 选择音频表示

候选路线：

1. **音频 encoder + projector（推荐起点）**
   - 使用 Whisper encoder、BEATs、CLAP、Qwen Audio encoder 或同类模型。
   - 输出连续音频 embedding。
   - 经过 projector/resampler 后插入语言模型序列。
2. **离散音频 token**
   - 使用音频 codec tokenizer。
   - 更适合最终端到端语音输出，但训练和序列长度成本更高。
3. **ASR 文本 + 音频 embedding 双通道**
   - 保留文本可解释性，同时保留语气和环境声信息。
   - 推荐作为第一版原生输入结构。

待决定：

- [ ] 音频采样率。
- [ ] 每个音频 chunk 的长度和重叠。
- [ ] 每秒保留多少音频 token。
- [ ] 音频 projector 的输出维度如何对齐 LLM hidden size。
- [ ] 音频、图像 token 和文本 token 的排列及时间标记。

推荐输入结构：

```text
<time_12.0>
<vision>...</vision>
<audio>audio embeddings...</audio>
<asr_partial>...</asr_partial>
```

### B3. 修改模型结构

- [ ] 新增 `audio_encoder`。
- [ ] 新增 `audio_projector` 或 resampler。
- [ ] 新增 `<audio_start>`、`<audio_patch>`、`<audio_end>` 等特殊 token。
- [ ] 修改 processor，使其同时接收图像、音频和文本。
- [ ] 修改 forward，将音频 embedding 填入对应 token 位置。
- [ ] 定义流式音频 KV/cache 策略，避免每个 tick 重算全部历史音频。
- [ ] 保持纯文本、纯图像和图文输入向后兼容。
- [ ] 编写最小单元测试：相同输入维度、mask、position id 和 batch 拼接均正确。

### B4. 分阶段训练

不要一开始全参数训练。

阶段 1：音频-语言对齐

- [ ] 冻结视觉编码器和 LLM。
- [ ] 只训练 audio projector/resampler。
- [ ] 使用 ASR、音频描述、声音事件分类和音视频对齐数据。

阶段 2：多模态指令微调

- [ ] 解冻 projector，并对 LLM 使用 LoRA 或少量层解冻。
- [ ] 加入视频 + 音频 + 文本联合任务。
- [ ] 训练模型利用声音补充视觉，而不是机械复述 ASR。

阶段 3：流式动作训练

- [ ] 将现有数据格式扩展为每个 tick 同时包含图像和音频窗口。
- [ ] 标注 `</silence>`、`</response>`、`</delegate>`、`</interrupt>`。
- [ ] 大量加入“无事件，应沉默”的负样本。
- [ ] 加入事件开始、持续、消失和重复场景。
- [ ] 加入用户插话和系统被打断样本。

阶段 4：偏好优化（可选）

- [ ] 优化误报、漏报、重复发言和打断时机。
- [ ] 安全事件优先优化召回率，普通事件优先控制误报率。

### B5. 扩展训练数据格式

基于 `datasets/README.zh-CN.md` 的逐秒消息格式，增加音频字段：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<1.0 seconds>\n<image>\n<audio>"
    },
    {
      "role": "assistant",
      "content": "</response> 检测到火灾警报和可见烟雾，请立即撤离。"
    }
  ],
  "images": ["frame_000001.jpg"],
  "audio_chunks": ["audio_000001.wav"],
  "audio_events": [
    {"label": "fire_alarm", "start": 0.2, "end": 0.9}
  ]
}
```

必须包含：

- [ ] 只有视频有证据。
- [ ] 只有音频有证据。
- [ ] 音视频证据一致。
- [ ] 音视频证据冲突。
- [ ] ASR 错误或不完整。
- [ ] 系统自己的 TTS 回声。
- [ ] 长时间无事件的沉默负样本。

### B6. 推理服务适配

- [ ] 先使用 Transformers 自定义推理服务验证模型正确性。
- [ ] 再检查当前 vLLM 是否原生支持新增模型架构和音频输入。
- [ ] 若不支持，编写 vLLM multimodal model plugin/registry 适配。
- [ ] 增加音频 processor cache 和音频 embedding cache。
- [ ] 压测连续音视频输入时的 KV cache、显存和实时系数。
- [ ] 保留模块化 ASR 作为日志、回退和可解释通道。

验收：

- [ ] 不依赖 ASR 文本，模型也能区分警报、哭声、爆炸等声音。
- [ ] 去掉音频输入后，相关回答明显退化，证明模型确实使用了声音。
- [ ] 音频与视频冲突时，模型能表达不确定性而不是随机选择。

---

## 4. 里程碑 C：端到端语音输出

第一版建议继续使用现有流式 TTS。只有在确实需要更自然的情绪、低延迟和声音语义闭环时，再训练直接语音输出。

任务：

- [ ] 选择音频 codec tokenizer。
- [ ] 决定文本 token 与语音 codec token 是串行生成还是双头生成。
- [ ] 建立增量 codec decoder，边生成边播放。
- [ ] 保留可中止 generation id。
- [ ] 训练文本内容、韵律、情绪和说话人一致性。
- [ ] 处理语言模型文本已经生成、但音频尚未播放完时的状态差异。
- [ ] 定义被打断后 codec cache、LM KV cache 和会话记忆的清理规则。

验收：

- [ ] 首包语音延迟达到目标。
- [ ] 用户打断时不会播放取消后的残留音频。
- [ ] 文本记录与实际播放内容一致。
- [ ] 端到端语音输出相较独立 TTS 有明确收益，否则保留模块化 TTS。

---

## 5. 建议的代码边界

不要把所有功能继续加入 `live_adapter.py`。推荐边界：

```text
services/omni/
├── audio_ingress.py       # 连续 PCM、ring buffer、时间戳
├── timeline.py            # 音视频统一事件时间轴
├── event_detector.py      # 非语言声音事件
├── orchestrator.py        # tick、优先级、决策调度
├── duplex_controller.py   # speaking/listening/cancel/barge-in
├── echo_filter.py         # TTS 自声参考与过滤
├── schemas.py             # TimelineEvent、Action、SessionState
└── tests/
```

现有组件职责保持：

- `webui`：采集和播放。
- `asr`：流式语音转写。
- `tts`：可取消的流式语音合成。
- `webinfer`：主模型推理和长期记忆。
- `omni`：连续时间轴、融合调度和全双工控制。

---

## 6. 资源与部署注意事项

- [ ] 不要默认占用服务器上其他用户正在使用的 GPU。
- [ ] 在获得明确授权前，只使用已分配的 GPU。
- [ ] 24 GB 单卡已经难以同时容纳当前 VLM、原生音频 encoder、大 KV cache 和 TTS。
- [ ] 开发阶段可以分时启动组件，或将小型音频事件模型放在 CPU。
- [ ] 完整模块化部署通常需要把 VLM、ASR/音频模型、TTS 分配到不同的已授权 GPU。
- [ ] 端到端训练需要单独评估显存，不能按当前推理显存估算。
- [ ] 先以 `MAX_MODEL_LEN=8192` 做功能开发，并通过较小 chunk 和摘要控制上下文。

---

## 7. 推荐的实际执行顺序

当前最值得立即开始的是：

1. [ ] 创建 `feature/continuous-omni` 分支。
2. [ ] 连续采集麦克风 PCM，并建立带时间戳的 ring buffer。
3. [ ] 把当前 final-only ASR 改成滑动窗口 partial ASR。
4. [ ] 增加一个小型环境声音检测器。
5. [ ] 新建 `services/omni/timeline.py`，统一音频、视频和输出时间轴。
6. [ ] 新建 `services/omni/orchestrator.py`，每 1 秒进行一次自主决策。
7. [ ] 将音频事件和 ASR partial 注入 `live_adapter.py`。
8. [ ] 给 TTS 增加 cancel，完成用户插话测试。
9. [ ] 解决 TTS 自声回流。
10. [ ] 用火灾视频 + 火灾警报音频完成里程碑 A 验收。
11. [ ] 收集误报、漏报和打断数据。
12. [ ] 再决定音频 encoder 和训练方案，进入里程碑 B。

## 8. 最终完成定义

只有同时满足以下条件，才称为“非回合制全模态系统”：

- [ ] 音频和视频连续输入，不依赖按钮或一句话结束。
- [ ] 没有用户问题也会进行模型决策。
- [ ] 模型能根据视觉、语音和非语言声音主动发言。
- [ ] 模型说话时仍持续监听和观察。
- [ ] 用户或高优先级事件可以打断输出。
- [ ] 系统不会把自己的 TTS 当成新用户输入。
- [ ] 长时间运行时上下文、显存和内存保持有界。
- [ ] 原生 Omni 阶段中，移除 ASR 文本后模型仍具备可测量的声音理解能力。
