# 原生 VLM 基线评估操作说明

> 文档状态：待 GPU 可用后执行
> 适用分支：`feature/continuous-omni`
> 基线模型：`jdopensource/JoyAI-VL-Interaction`
> 编写日期：2026-07-30

## 1. 目的

在开始训练 Omni 模型之前，完整记录原生 JoyAI-VL-Interaction 的视觉语言能力，作为训练后模型的对照基线。

本评估主要回答两个问题：

1. Omni 训练是否保留了原生模型已有的图像、视频理解和视觉语言推理能力。
2. Omni 训练带来的音频和全双工能力，是否以明显损害原有视觉能力为代价。

本评估只测试模型自身，不测试 ASR、TTS、CLAP、WebUI、Omni 决策器或摘要模型。原生模型与训练后的 Omni 模型必须使用同一份样本、同一套提示词、相同的视觉采样和相同的生成参数。

README 中列出的 26 项公开基准成绩属于项目发布结果，不能替代本地实测基线。只有保存了本次实际请求、原始回答、评分结果和运行环境的记录，才可以作为训练前基线。

## 2. 评估原则

### 2.1 必须保持不变的条件

训练前后对比时固定以下项目：

- 测试集及其文件内容；
- 图像缩放和视频抽帧方式；
- 输入图像数量、顺序和时间戳；
- system prompt、用户问题及选项顺序；
- chat template；
- `max_tokens`、`temperature`、`top_p`；
- 答案抽取和评分程序；
- vLLM 与模型处理器版本；如果必须升级，要额外做版本影响复测；
- GPU 型号、并行方式和上下文长度；性能对比时尤其必须一致。

训练完成前，不得根据原生 VLM 的错误修改正式测试答案、问题或评分规则。发现数据错误时，应记录原因并发布一个新的测试集版本，同时保留旧版本结果。

### 2.2 数据隔离

- 正式对比集不得进入 Omni 训练集、验证集或数据合成提示词。
- 优先使用来源明确、许可允许评估的公开基准测试集。
- 项目自建样本要记录来源、许可、采集时间和 SHA-256。
- 不确定是否被原模型训练过的数据，只用于横向能力观察，不用于宣称严格泛化。
- 训练人员在训练结束前不应查看正式隐藏测试集答案。

### 2.3 公平比较 Omni 模型

复测 Omni 模型视觉语言能力时：

- 关闭或屏蔽音频输入；
- 不向 Omni 模型额外提供 ASR 文本、CLAP 标签或人工事件标签；
- 不让外部规则门替模型回答；
- 对同一问题只比较 Thinker/主模型的视觉语言输出；
- Talker 的音色、韵律和语音质量另行评估，不计入本基线；
- 如果 Omni 模型接口不同，应在适配层转换为相同的逻辑输入，不能增加额外视觉描述。

## 3. 评估范围

建议分为三层。第一次执行至少完成第 1、2 层；第 3 层可在获得相应评测工具和数据许可后补齐。

### 3.1 第 1 层：环境与小样本冒烟

目的：确认权重、服务、图片输入和生成参数正确。

建议 10 个不参与正式计分的样本，覆盖：

- 单图物体识别；
- 属性和颜色；
- OCR；
- 空间关系；
- 简单计数；
- 两帧前后变化；
- 短视频动作；
- 时间顺序；
- 一条应回答的场景；
- 一条应保持沉默的场景。

冒烟失败时不要开始正式评估。

### 3.2 第 2 层：项目固定回归集

建立一份不会进入训练集的固定回归集，建议初版不少于 200 题：

| 子集 | 建议题数 | 主要能力 |
| --- | ---: | --- |
| 单图感知 | 30 | 物体、属性、计数、空间关系、OCR |
| 多图状态变化 | 30 | 出现、消失、移动、状态前后变化 |
| 短视频理解 | 40 | 动作、事件、时序、因果 |
| 长视频与记忆 | 30 | 跨时间检索、人物/物体追踪、早期信息回忆 |
| 视频推理与知识 | 30 | 基于画面的推断、常识、过程理解 |
| 时间定位 | 20 | 事件开始、结束和高光区间 |
| 交互动作选择 | 20 | `silence`、`response`、`delegate` 的视觉触发判断 |

应同时包含：

- 正样本与负样本；
- 静态、快速变化和遮挡场景；
- 中文、英文和无需 OCR 的样本；
- 明确答案题与开放描述题；
- 日常安全场景和视觉危险场景；
- 容易诱发幻觉的反事实问题，例如询问画面中不存在的物体。

### 3.3 第 3 层：公开视频理解基准

项目 README 报告了以下 26 项基准，可优先复现与里程碑 B 最相关的子集：

- 通用感知：TOMATO、MotionBench、TVBench、MVBench、TempCompass、Perception Test；
- 长视频：VideoMME、VideoMME V2、LongVideoBench、VideoEval-Pro、EgoSchema；
- 视频推理：Video-Holmes、VRBench、VCRBench、LongVideoReason、MMR-VBench；
- 视频知识：MMVU、SciVideo、VideoMathQA；
- 时间定位：Charades-TL、ActivityNet-TL、QVHighlight-TL、STVG；
- 流式视频：OVOBench、OVBench、ODVBench。

公开基准必须使用各自官方版本和官方评分规则。不要把不同基准的原始分数直接混成一个总分；如需展示宏平均，必须同时保留每项独立成绩。

资源有限时，建议首轮优先级为：

1. MVBench 或 VideoMME：一般视频理解；
2. TempCompass：时间理解；
3. LongVideoBench：长视频；
4. QVHighlight-TL 或 ActivityNet-TL：时间定位；
5. OVBench 或 ODVBench：流式视频。

## 4. 固定回归集格式

建议使用 JSONL 清单，一行一个问题。媒体文件放在清单目录下，以相对路径引用。

```json
{"id":"image_0001","subset":"single_image","media":[{"path":"media/image_0001.jpg","timestamp_s":0.0}],"question":"图中桌面上有几个杯子？","answer_type":"short","reference":["2","两个"],"expected_action":"response","tags":["counting"],"source":"internal","license":"internal-eval-only","split":"hidden_test"}
{"id":"video_0001","subset":"short_video","media":[{"path":"frames/video_0001/frame_000000.jpg","timestamp_s":0.0},{"path":"frames/video_0001/frame_000001.jpg","timestamp_s":1.0}],"question":"人物先拿起了什么，随后放到了哪里？","answer_type":"open","reference":"人物先拿起杯子，随后把它放到水槽旁。","expected_action":"response","tags":["temporal_order","action"],"source":"dataset-name","license":"license-name","split":"hidden_test"}
{"id":"silent_0001","subset":"interaction","media":[{"path":"frames/silent_0001/frame_000000.jpg","timestamp_s":0.0}],"question":"判断现在是否需要主动发言。只输出动作标签。","answer_type":"action","reference":"</silence>","expected_action":"silence","tags":["negative","stable_scene"],"source":"internal","license":"internal-eval-only","split":"hidden_test"}
```

每条记录至少包含：

- `id`：永久稳定且唯一；
- `subset`：计分子集；
- `media`：按时间排序的图片或抽帧；
- `question`：固定问题；
- `answer_type`：`choice`、`short`、`open`、`temporal` 或 `action`；
- `reference`：标准答案或人工评分参考；
- `source` 和 `license`；
- `split`：必须标明为未参与训练的测试集；
- `tags`：用于统计细分类别。

生成正式基线后，计算并保存清单文件和全部媒体文件的 SHA-256。训练后必须使用相同哈希的副本。

## 5. 运行前记录

### 5.1 GPU 使用约束

当前服务器 GPU 均被占用，本节命令只能在确认有空闲卡后执行。

```bash
nvidia-smi
```

执行要求：

- 只使用已经明确分配给自己的 GPU；
- 不停止、不迁移其他用户进程；
- 不为了评估重启此前手动停止的 ASR 稳定性测试；
- 显存不足时停止本次模型服务，调整本模型参数后重试；
- 不连接或调用任何外部模型服务。

### 5.2 记录代码与权重版本

从仓库根目录执行：

```bash
git status --short
git rev-parse HEAD
git branch --show-current
git log -1 --format=fuller
```

记录模型目录：

```bash
MODEL_PATH=/tmp/models/jdopensource/JoyAI-VL-Interaction
find "$MODEL_PATH" -maxdepth 1 -type f -printf '%f\t%s\n' | sort
sha256sum \
  "$MODEL_PATH/config.json" \
  "$MODEL_PATH/generation_config.json" \
  "$MODEL_PATH/tokenizer_config.json" \
  "$MODEL_PATH/model.safetensors.index.json"
```

若某个元数据文件不存在，在记录中写明“文件不存在”，不要临时替换成其他文件。正式基线至少应保存权重文件名和大小；条件允许时对全部权重分片计算一次 SHA-256。

### 5.3 记录软件和硬件

```bash
nvidia-smi
python --version
python -m pip show torch transformers vllm
ffmpeg -version | head -n 1
```

把输出保存到本次运行目录的 `environment.txt`。不得只记录“使用 4090”或“使用最新版环境”。

## 6. 启动原生 VLM

基线能力评估只启动主模型，不启动 summary、adapter、WebUI、ASR 或 TTS。

从仓库根目录执行：

```bash
cd services/webinfer

MAIN_GPU=<空闲GPU编号> \
MODEL_PATH=/tmp/models/jdopensource/JoyAI-VL-Interaction \
SERVED_MODEL_NAME=jdopensource/JoyAI-VL-Interaction \
MAIN_MODEL_PORT=7060 \
MAX_MODEL_LEN=8192 \
MAIN_GPU_MEMORY_UTILIZATION=0.85 \
MAIN_SMOKE_ENABLE=1 \
bash scripts/run.sh models
```

说明：

- 首次使用 `MAX_MODEL_LEN=8192`，先验证能否稳定加载；长视频基准需要更长上下文时，单独建立另一套配置，不得与 8192 配置混合计分。
- 多卡张量并行时，`MAIN_GPU` 使用逗号分隔，例如 `0,1`，并显式设置 `TENSOR_PARALLEL_SIZE=2`。
- 正式基线和训练后复测必须使用一致的上下文长度和并行方式。
- 服务运行在前台，结束时使用 `Ctrl+C`，不要使用模糊匹配批量杀进程。

在另一个终端检查：

```bash
curl http://127.0.0.1:7060/v1/models
```

看到 `jdopensource/JoyAI-VL-Interaction` 后，再执行一次仓库自带冒烟：

```bash
cd services/webinfer
python smoke.py main-vlm \
  --api-base http://127.0.0.1:7060/v1 \
  --model jdopensource/JoyAI-VL-Interaction \
  --attempts 3 \
  --interval 2 \
  --timeout 60
```

只有 `/v1/models` 和图像冒烟都成功，才进入正式评估。

## 7. 请求方式与生成参数

直接请求原生 vLLM：

```text
http://127.0.0.1:7060/v1/chat/completions
```

固定参数：

```json
{
  "model": "jdopensource/JoyAI-VL-Interaction",
  "max_tokens": 256,
  "temperature": 0.0,
  "top_p": 1.0,
  "stream": false
}
```

约束：

- 客观题使用确定性解码，不对答错题反复抽样挑选最好回答。
- 多选题提示词必须要求只输出选项字母，答案抽取失败按错误计。
- 开放题保存完整原始回答，评分前不得人工改写。
- 图片以原始字节编码为 data URL 发送，避免服务器路径白名单差异。
- 视频必须在评估前按固定策略抽帧；不要根据问题人工选择“最佳帧”。
- 每张帧图片之前传入固定格式时间戳，例如 `<12.0 seconds>`。
- 超过上下文上限的样本计为失败，并单独记录 `context_overflow`，不得静默减少帧数。

建议的多帧消息结构：

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "<0.0 seconds>"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "<1.0 seconds>"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "问题：人物先做了什么，随后发生了什么？"}
  ]
}
```

## 8. 视频抽帧

项目训练数据转换默认按 1 FPS 组织时间对齐帧。项目固定回归集建议同样使用 1 FPS，便于复现：

```bash
ffmpeg \
  -hide_banner \
  -loglevel error \
  -i input.mp4 \
  -vf fps=1 \
  -q:v 5 \
  -start_number 0 \
  frames/frame_%06d.jpg
```

同时用 `ffprobe` 保存：

- 原视频时长；
- 帧率和分辨率；
- 是否存在旋转信息；
- 实际抽取帧数。

公开基准必须遵守该基准的官方抽帧协议；不能统一改成项目的 1 FPS。若公开基准允许不同帧数，需要提前固定配置，并在训练前后保持相同。

## 9. 执行顺序

一次正式基线运行遵循以下顺序：

1. 确认 GPU 获得授权且空闲；
2. 创建唯一运行编号，例如 `native_vlm_20260730_001`；
3. 记录 Git、权重、环境、GPU 和数据集哈希；
4. 启动原生 VLM；
5. 完成 10 条冒烟样本；
6. 清空冒烟结果，重启服务或确保正式运行不受会话状态影响；
7. 按清单固定顺序执行项目回归集；
8. 执行公开基准；
9. 保存每条原始请求、原始响应、错误和耗时；
10. 自动评分客观题；
11. 对开放题进行盲评；
12. 生成分子集报告；
13. 停止本次 VLM 服务并确认显存释放；
14. 将结果目录设为只读备份。

正式运行期间不要修改提示词、代码、环境变量或数据文件。出现故障时终止该次运行，修复后使用新的运行编号完整重跑。

## 10. 输出记录格式

每题至少保存以下字段：

```json
{
  "run_id": "native_vlm_20260730_001",
  "sample_id": "video_0001",
  "model": "jdopensource/JoyAI-VL-Interaction",
  "model_revision": "weights fingerprint",
  "request": {},
  "raw_response": {},
  "answer_text": "模型抽取后的文本",
  "reference": "标准答案",
  "score": 1.0,
  "error_type": null,
  "latency_ms": 1832.4,
  "input_frame_count": 12,
  "started_at": "ISO-8601 timestamp"
}
```

推荐目录：

```text
evaluation_results/
└── native_vlm_baseline/
    └── native_vlm_20260730_001/
        ├── run_config.json
        ├── environment.txt
        ├── dataset.sha256
        ├── requests.jsonl
        ├── responses.jsonl
        ├── scores.jsonl
        ├── failures.jsonl
        └── summary.json
```

评估原始媒体和结果默认不提交 Git，尤其不要提交受许可证限制的视频。是否归档到其他存储位置由用户另行决定。

## 11. 评分方法

### 11.1 客观题

- 单选/多选：官方答案准确率；
- 是非题：准确率；
- 短答案：规范化后 exact match，同时报告 token F1；
- 数值题：提前固定允许误差；
- 时间定位：使用官方 IoU、mAP 或 Recall 指标；
- 动作选择：标签准确率和混淆矩阵；
- 沉默判断：重点报告 `silence` precision、recall 和误报率。

规范化只能做大小写、首尾空白和明确允许的标点处理，不能用大模型把错误答案“解释成正确答案”。

### 11.2 开放题

开放回答采用不知道模型身份的盲评。建议两名评审分别打分：

| 维度 | 分值 | 说明 |
| --- | ---: | --- |
| 视觉事实正确性 | 0–2 | 是否与画面直接证据一致 |
| 答案完整性 | 0–2 | 是否覆盖问题要求的关键信息 |
| 时间关系 | 0–1 | 是否正确表达顺序、持续或时间位置 |
| 指令遵循 | 0–1 | 是否按要求回答且不过度展开 |
| 幻觉 | 0 或 -2 | 捏造画面中不存在的重要事实时扣分 |

两名评审分差大于 2 分时，由第三名评审仲裁。训练前和训练后回答要打乱顺序并隐藏模型名称，避免主观偏向。

### 11.3 性能指标

记录但不要混入能力总分：

- 请求成功率；
- 端到端响应时间；
- 首 token 延迟（评估工具支持流式记录时）；
- 输入帧数和视觉 token 数；
- 输出 token 数；
- 峰值显存；
- 吞吐量。

性能数据只有在 GPU、并行方式、上下文长度、vLLM 版本和并发数相同时才可直接比较。

## 12. 训练前基线报告

基线报告至少包括：

1. 运行编号和日期；
2. 代码 commit；
3. 权重指纹；
4. GPU、CUDA、PyTorch、Transformers、vLLM 版本；
5. 数据集版本和 SHA-256；
6. 每个子集的题数、得分和失败数；
7. 幻觉率、动作混淆矩阵；
8. 长视频上下文溢出数量；
9. 延迟和显存；
10. 典型正确、错误和拒答案例；
11. 所有偏离本说明的配置。

不要只报告总平均分。至少分别报告：

- 单图视觉；
- 短视频；
- 长视频；
- 时间理解；
- 视觉推理；
- 交互动作选择。

## 13. 训练后对比与建议验收线

验收线必须在查看训练后结果之前冻结。建议初始标准如下：

- 项目固定回归集视觉语言宏平均下降不超过 2 个百分点；
- 任一核心子集下降超过 3 个百分点必须分析；
- 视觉事实幻觉率不得明显上升；
- 客观题失败或格式错误率不得上升超过 1 个百分点；
- `silence/response/delegate` 使用相同视觉输入时，不得出现不可解释的大规模行为漂移；
- 如果 Omni 在音频任务上提升但视觉能力超过阈值退化，不判定为里程碑 B 完成。

报告使用配对差值：

```text
delta = Omni 得分 - 原生 VLM 得分
```

除总体差值外，应列出逐题变化：

- 原生正确、Omni 错误：视觉能力回退；
- 原生错误、Omni 正确：训练带来的正迁移；
- 两者都错：模型共同短板；
- 两者都对但 Omni 更慢：性能回退。

## 14. 常见错误

- 通过 `8070` adapter 测原生模型，导致摘要和历史状态影响结果；
- 原生模型使用 1 FPS，Omni 使用人工挑选关键帧；
- 训练后向模型额外提供 ASR 或事件标签，再称为视觉能力提升；
- 使用 README 发布分数代替本地实际运行；
- 只保存清洗后的答案，没有保存原始响应；
- 在正式评估中途修改 prompt 后继续沿用同一个运行编号；
- 看到模型答错后修改参考答案；
- 只计算总平均，掩盖某个视觉能力子集明显退化；
- GPU 配置不同却直接比较延迟和显存；
- 将评估集或人工评分参考泄漏进训练数据。

## 15. 本次待办

在 GPU 可用前可以完成：

- [ ] 确定项目固定回归集的样本来源和许可；
- [ ] 冻结 JSONL schema；
- [ ] 划分训练集、开发集和隐藏测试集；
- [ ] 准备 10 条冒烟样本；
- [ ] 准备首版不少于 200 题的固定回归集；
- [ ] 固定问题模板、生成参数和评分规则；
- [ ] 编写批量评估与报告脚本；
- [ ] 对清单和媒体生成 SHA-256；
- [ ] 确定优先复现的公开基准及其官方评测工具。

GPU 可用后第一时间执行：

- [ ] 记录原生模型权重与运行环境；
- [ ] 启动 `7060` 原生 VLM；
- [ ] 完成冒烟；
- [ ] 完成项目固定回归集；
- [ ] 完成选定公开基准；
- [ ] 生成并冻结训练前基线报告；
- [ ] 确认基线结果已安全备份后，再开始 Omni 模型训练。
