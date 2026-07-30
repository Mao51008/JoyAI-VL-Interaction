# 里程碑 B：B1 音频 Encoder 接入 JoyAI 的可行性调查

> 调查日期：2026-07-30
> 调查范围：本地代码、官方模型配置、官方源码、官方论文和官方模型/数据集卡。
> 本阶段未连接服务器、未下载权重、未启动 GPU 或模型。

## 1. 结论

**B1 结论：有条件可行（Conditional Go）。**

> B2 更新（2026-07-30）：无 GPU 接口原型已经实现。`training/omni` 包含
> `omni-training-v1` 数据协议、纯音频无文本样本、假 Encoder、时间和 mask 批处理、
> 小维度 bridge/action head、CPU loss 以及 checkpoint 保存恢复。该结果不代表真实
> JoyAI/Qwen 模型已经加载或训练。

建议路线：

```text
16 kHz PCM
  -> Qwen3-ASR 特征提取器
  -> 冻结的 Qwen3-ASR Encoder
  -> 冻结的 Qwen3-ASR 原生 projector（1024 -> 2048）
  -> 新增可训练线性桥（2048 -> 4096）
  -> 音频时间/模态嵌入
  -> 作为 inputs_embeds 插入 JoyAI/Qwen3-VL
```

理由：

- JoyAI 基于标准 `Qwen3VLForConditionalGeneration`，语言隐藏维度为 4096，可通过 `inputs_embeds` 接入新模态表示。
- Qwen3-ASR 原生 Encoder 已提供独立接口，Encoder 输出为 1024 维，其已训练 projector 输出为 2048 维。
- 第一阶段只新增约 839 万参数的 `2048 -> 4096` 线性桥，尽量保留 Qwen3-ASR 已学到的语音语义对齐能力。
- 音频约为 13 token/秒，短时流式窗口的 token 开销可接受。
- 2 张 24 GB RTX 4090 足以进行 projector-only 和 LoRA/QLoRA 试验，但不适合 8.77B 模型的全参数训练。

GPU 可用前可以继续：

1. 定义音频 Encoder、projector 和模态拼接接口。
2. 用假 Encoder 验证形状、掩码、时间戳、流式窗口和无音频兼容路径。
3. 建立训练配置、断点续训、checkpoint 轮转和最小数据管线。
4. 建立视觉语言回归评测入口，但不运行真实模型。

必须等 GPU 和权重可用后确认：

1. JoyAI 与新版 Transformers/Qwen3-ASR 原生实现能否在同一训练环境正确加载。
2. 音频特征插入后是否保持视觉语言基线能力。
3. Qwen3-ASR Encoder 对环境声音的语义表示是否足够。
4. 2×4090 下的真实显存、吞吐和最长上下文。
5. projector-only 短训是否能让 JoyAI 稳定使用音频信息。

因此当前允许进入 **B2 无 GPU 接口原型**；真实训练保持“有条件放行”，长时间训练暂不启动。

## 2. JoyAI 基座盘点

### 2.1 模型结构

官方配置显示：

| 项目 | 值 |
|---|---:|
| 模型类 | `Qwen3VLForConditionalGeneration` |
| 总参数量 | 8,767,123,696 |
| BF16 权重大小 | 约 17.53 GB |
| 语言隐藏维度 | 4096 |
| 语言层数 | 36 |
| 注意力头 / KV 头 | 32 / 8 |
| 最大位置长度 | 262,144 |
| 视觉 Encoder 隐藏维度 | 1152 |
| 视觉输出维度 | 4096 |
| 位置编码 | Qwen3-VL MRoPE |

JoyAI 的 tokenizer 已添加 `</silence>` 和 `</response>`，但模型卡中的模板只定义了图像、视频和文本输入，没有可确认的音频占位符。不能直接照搬 Qwen3-ASR 的 `audio_token_id=151676`，因为两个 tokenizer 的 token 映射不保证相同。

### 2.2 训练资产

论文说明 JoyAI 使用了：

- 时间对齐的交互数据与普通回合数据混合 SFT；
- `</silence>`、`</response>` 和 delegation 动作；
- 对重复 silence 和 response 使用不同损失权重；
- 后续使用 EasyVideoR1/GRPO。

但当前官方仓库和本地仓库没有可直接执行的 SFT/LoRA/DeepSpeed 训练配置或训练入口。README 提到 LLaMA-Factory 和 EasyVideoR1，不等于已经提供可复现的训练工程。

**工程结论：** B 阶段需要建立独立训练目录和最小训练入口。projector-only 阶段优先采用 Transformers + Accelerate/DeepSpeed + PEFT 的受控实现；确认模态接口稳定后，再决定是否适配 LLaMA-Factory。

## 3. Qwen3-ASR Encoder 盘点

建议使用官方原生 Transformers 权重：

```text
Qwen/Qwen3-ASR-1.7B-hf
```

第一版不依赖 Qwen 仓库的自定义完整推理封装，因为我们只需要 Encoder 和已训练 projector，原生 Transformers 类更便于拆分、冻结和测试。

| 项目 | 值 |
|---|---:|
| 输入采样率 | 16 kHz |
| Mel bins | 128 |
| Encoder 隐藏维度 | 1024 |
| Encoder 层数 | 24 |
| 注意力头 | 16 |
| FFN 维度 | 4096 |
| 卷积时间下采样 | 8× |
| 原生 projector 输出 | 2048 |
| 每秒音频 token | 约 13 |
| CNN 基本分块 | 100 Mel 帧，约 1 秒 |
| 推理注意力窗口 | 800 Mel 帧，约 8 秒 |

官方源码中的三层 stride-2 Conv2d 将时间轴下采样约 8 倍；长度计算表明每 100 个 Mel 帧得到 13 个 Encoder token。

原生 Encoder 可独立实例化为 `Qwen3ASREncoder`。完整 ASR 模型的 `multi_modal_projector` 为：

```text
Linear(1024, 1024)
-> activation
-> Linear(1024, 2048)
```

## 4. 流式能力判断

Qwen 官方 streaming wrapper 的基本行为是：

1. 接收新音频块；
2. 把音频追加到累计缓冲区；
3. 将从开头到当前时刻的全部音频重新送入模型；
4. 对文本结果做前缀回滚。

这能提供流式用户体验，但不是有界计算量的增量 Encoder，也没有可直接复用的 Encoder KV cache。

第一版建议：

- 以 1 秒为 CNN 提交单位；
- 以 8 秒为 Encoder 注意力窗口；
- 已完成的 8 秒窗口输出进入只读缓存；
- 只重算当前未完成窗口；
- 不足 1 秒的尾部可产生临时结果，但只有跨过 1 秒边界才提交稳定 token；
- 上层 Streaming Thinker 只保留最近原始音频窗口，旧音频转为已提交音频 token、摘要或事件。

这个方案能把重算范围限制在当前约 8 秒窗口内，但是否与离线完整编码数值等价，必须在 GPU 阶段用真实权重验证。第一版不承诺 Encoder KV cache。

## 5. 模态兼容矩阵

| 接口项 | Qwen3-ASR | JoyAI/Qwen3-VL | 处理方式 |
|---|---|---|---|
| 音频输入 | 16 kHz、128-bin Mel | 无原生音频 | 沿用 Qwen3-ASR processor |
| Encoder 输出 | 1024 | 语言隐藏 4096 | 先过原生 2048 projector，再桥接到 4096 |
| 时间压缩 | 约 13 token/秒 | 支持长上下文 | 最近窗口保留细粒度，旧窗口摘要化 |
| 位置编码 | Encoder 正弦位置 + 局部窗口 | 三轴 MRoPE | 音频 token 先按 text-like 位置处理 |
| 时间语义 | 音频帧顺序 | 视觉/文本时间线 | 额外加入连续时间嵌入和时间戳元数据 |
| 占位 token | Qwen3-ASR 有专用 ID | JoyAI 未定义 | 按字符串新增 token，不硬编码其他模型的 ID |
| 流式缓存 | 官方封装重算累计音频 | LLM 可缓存历史 token | 缓存完成音频窗口，重算活动窗口 |
| 无音频兼容 | 不适用 | 现有视觉文本链路 | 无音频时严格走原 JoyAI 输入路径 |

### 5.1 音频 special token

建议新增：

```text
<|audio_start|>
<|audio_pad|>
<|audio_end|>
```

实现要求：

- 通过 token 字符串动态查询新 ID，不使用硬编码 ID；
- 扩展 input embedding 和 LM head；
- 只训练新增 token 行，原 token 行保持冻结；
- `<|audio_pad|>` 的 embedding 在前向时由投影后的音频特征替换；
- 模型保存时同时保存 tokenizer 和新增配置；
- 无音频请求不得插入这些 token。

如果后续发现当前 tokenizer 中存在官方保留 token，可在真实 tokenizer 加载测试后再决定是否复用；B2 假模型阶段不做此假设。

### 5.2 位置编码

第一版把音频 token 视为 text-like 序列：三个 MRoPE 轴使用相同递增位置。音频的真实时间信息由独立的时间嵌入和时间戳元数据承载，不伪装成视觉网格。

如果后续评测证明长音频时间定位不足，再研究专用 audio MRoPE；这不作为第一版前置条件。

## 6. 建议接口契约

接口只约束语义和形状，不要求 B2 引入 PyTorch 或真实模型依赖。

```text
AudioBatch
  samples:            [B, T]
  sample_rate:        16000
  sample_lengths:     [B]
  chunk_start_ms:     [B]
  is_final:           [B]

AudioEncoderOutput
  hidden_states:      [B, Ta, 1024]
  attention_mask:     [B, Ta]
  token_times_ms:     [B, Ta]
  committed_lengths:  [B]

ASRProjectedOutput
  hidden_states:      [B, Ta, 2048]
  attention_mask:     [B, Ta]
  token_times_ms:     [B, Ta]

JoyAudioEmbeddings
  inputs_embeds:      [B, Ta, 4096]
  attention_mask:     [B, Ta]
  position_times_ms:  [B, Ta]
```

所有实现必须支持空音频、批内不同长度、非整秒尾部和显式 reset。

## 7. Token 预算

按 13 token/秒估算：

| 音频长度 | 音频 token |
|---:|---:|
| 1 秒 | 13 |
| 2 秒 | 26 |
| 6 秒 | 78 |
| 8 秒 | 104 |
| 10 秒 | 130 |
| 30 秒 | 390 |
| 60 秒 | 780 |
| 120 秒 | 1,560 |

第一版无需额外 resampler。建议保留最近 8～10 秒细粒度音频 token；更早内容根据任务转成 ASR 文本、环境事件或短摘要。若真实训练发现长音频成本或冗余明显，再增加 learned resampler。

## 8. 双 4090 资源判断

以下为配置和参数量推导出的工程估算，不是真实跑测结果。

| 部分 | 估算 |
|---|---:|
| JoyAI BF16 权重 | 约 17.53 GB |
| Qwen3-ASR Encoder | 约 3.15 亿参数，BF16 约 0.63 GB |
| Qwen3-ASR 原生 projector | 约 315 万参数 |
| 新增 `2048 -> 4096` bridge | 约 839 万参数 |

| 训练方式 | 2×24 GB 4090 | 结论 |
|---|---|---|
| 仅训练 bridge/projector | FSDP/ZeRO-3、激活检查点、batch 1 | 可行，需实测 |
| JoyAI LoRA | QLoRA 或分片 LoRA | 可行，需限制初始序列长度 |
| 少量解冻音频 Encoder | 分片 + 小 batch | 有条件可行 |
| JoyAI 全参数 BF16 + Adam | 参数、梯度、优化器状态远超 48 GB | 不可行 |

4090 没有 NVLink，跨卡分片会受 PCIe 通信影响，但不妨碍小规模可行性训练。第一轮应从 2k～4k 总序列、batch 1、梯度累积开始；8k 及以上必须在显存 profiling 后决定。

不推荐简单 DDP：每张卡复制约 17.53 GB JoyAI 权重后，留给音频 Encoder、激活、视觉 token 和 CUDA 工作区的空间过小，视频训练容易溢出。

优先顺序：

1. projector-only：FSDP/ZeRO-3 分片 JoyAI；
2. LoRA：QLoRA 或 FSDP + LoRA；
3. 只有显存实测有余量时，才逐层解冻音频 Encoder；
4. 不在 2×4090 上尝试 JoyAI 全参数训练。

## 9. 训练环境建议

建立与在线推理隔离的训练环境，避免破坏当前 WebUI、ASR 和 vLLM 环境。

- 使用包含原生 Qwen3-ASR 支持的 Transformers 版本；
- 同版本完成 JoyAI/Qwen3-VL 加载兼容测试；
- projector-only 使用 Accelerate/DeepSpeed 或 FSDP；
- LoRA 使用 PEFT；
- 所有配置支持断点续训；
- checkpoint 数量做成配置项，默认保留最近若干个，最终值等服务器磁盘容量确认后再定。

## 10. 分阶段验证门

### B2：无 GPU 接口原型

- 假 Encoder 固定产生约 13 token/秒；
- 验证 shape、mask、时间戳和空输入；
- 验证 1 秒提交、8 秒活动窗口及 reset；
- 验证音频插入不改变无音频输入；
- 验证最近窗口与历史摘要的上下文组织。

通过条件：全部 CPU 测试通过，且不依赖模型权重。

### B3：单卡加载冒烟

- 加载 JoyAI tokenizer/config；
- 加载 Qwen3-ASR Encoder 和原生 projector；
- 检查实际参数名、dtype、shape 和 token ID；
- 离线音频与窗口化音频做数值对比；
- 记录单模型峰值显存。

通过条件：接口与 B2 契约一致，窗口边界误差有可解释结论。

### B4：双卡 projector 短训

- 20～50 steps；
- 验证 loss 下降、梯度只进入允许模块；
- 验证 checkpoint 保存、恢复和跨夜运行机制；
- 同时跑无音频视觉语言回归样本。

通过条件：无 OOM/NaN，断点可恢复，视觉语言能力未出现明显结构性退化。

### B5：语义覆盖试验

分别验证中英文语音、情绪和说话方式、用户重叠说话，以及火警、爆炸、玻璃破碎等环境声音。若 Qwen Encoder 对环境声音不足，再增加 CLAP 或事件 detector；第一版不预先引入双音频 Encoder。

## 11. 风险与待验证项

| 风险 | 当前处理 |
|---|---|
| 官方仓库无可执行训练入口 | 自建最小训练工程 |
| Qwen 官方 streaming 会重算累计音频 | 缓存完成 8 秒窗口，只重算活动窗口 |
| JoyAI 无音频 token | 新增字符串 token，禁止硬编码 ID |
| Transformers 版本兼容未知 | 建立隔离训练环境并先做配置/加载冒烟 |
| 环境声音语义能力未知 | 先测 Qwen Encoder，不足再引入 CLAP |
| 视觉语言能力可能退化 | 训练前固定原生 VLM 基线，训练中持续回归 |
| 2×4090 显存只是估算 | 首轮 GPU 测试记录峰值显存和吞吐 |
| 验收阈值尚未最终确定 | 在正式长训前冻结，不阻塞 B2/B3 |

## 12. Go / No-Go 判定

### 立即 Go

- B2 无 GPU 音频接口和假模型测试；
- 训练配置骨架；
- 数据格式、断点和 checkpoint 机制；
- 原生 VLM 回归评测入口。

### 条件 Go

- Qwen3-ASR Encoder 真实加载；
- 2×4090 projector-only 短训；
- JoyAI LoRA；
- 有限解冻音频 Encoder。

条件是 GPU 可用，并通过 B3/B4 的加载、显存、数值和回归测试。

### 当前 No-Go

- JoyAI 全参数训练；
- 未做原生 VLM 基线就开始正式长训；
- 未验证 Qwen Encoder 就直接叠加 CLAP；
- 把官方 wrapper 的累计重算宣称为真正的增量 Encoder cache；
- 在现有在线推理环境中直接升级核心训练依赖。

## 13. 官方资料

- JoyAI 官方仓库：<https://github.com/jd-opensource/JoyAI-VL-Interaction>
- JoyAI 官方模型：<https://huggingface.co/jdopensource/JoyAI-VL-Interaction>
- JoyAI 官方数据集：<https://huggingface.co/datasets/jdopensource/JoyAI-VL-Interaction>
- JoyAI 技术报告：<https://echovideo.jd.cn/JoyAI-VL-Interaction/JoyAI-VL-Interaction-Reportv1.pdf>
- Qwen3-ASR 官方仓库：<https://github.com/QwenLM/Qwen3-ASR>
- Qwen3-ASR 原生 Transformers 权重：<https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf>
- Transformers Qwen3-ASR 文档：<https://huggingface.co/docs/transformers/main/model_doc/qwen3_asr>
- Transformers Qwen3-ASR 实现：<https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_asr/modeling_qwen3_asr.py>
- Qwen3-ASR 技术报告：<https://arxiv.org/abs/2601.21337>
