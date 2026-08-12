# VLM-Omni 阶段二技术交接与运行手册

最后核对：2026-08-12。主力服务器 maoyy@10.11.12.30，唯一部署分支 feature/continuous-omni。

## 目标与架构

线上系统：视频/RTSP/WebRTC 到 JoyAI-VL 主视觉模型，模型决定 speak/silence/delegate；浏览器音频到 Qwen3-ASR；文本输出到 Qwen3-TTS；复杂任务到 background-agent。

本轮是 VLM-Omni 阶段二训练准备，而不是上线部署。目标是把冻结 Qwen3-ASR 特征经 audio_projector 映射到 Qwen3-VL token embedding，并以 decoder LoRA 学习音频理解和回答；混入少量离线图文教师 SFT，避免音频训练造成视觉/文本能力退化。

- Stage1：冻结 ASR、Qwen3-VL、原生 projector，只训练 audio_projector。
- Stage2：冻结 ASR encoder 和原始 Qwen3-VL，只训练 audio_projector 加指定 LLM LoRA。
- 完整 Omni：只在音频、视觉、混合 dev 都稳定后开始。

不得用 torch.no_grad 包住冻结 LLM forward；assistant loss 到 audio_projector 的梯度图必须保留。

## 当前结论

32 条混合任务过拟合实验不能证明 Stage2 成功。SpokenWOZ 过窄（酒店/餐厅）、回复模板化且同问多答；数字阿拉伯/英文读法差异会放大 WER/CER；ASR 与对话必须使用匹配的训练提示评测。因此旧 Stage2 best/last 仅作对照，不能当默认初始化。

当前计划：扩展音频源为 VoiceAssistant-400K 单轮子集、Clotho-AQA 共识子集、LibriSpeech ASR replay；SpokenWOZ 不能是唯一来源。先完成数据适配、CPU preflight、小样本 overfit 和独立 dev，得到用户授权后才运行 GPU pilot。

## 必须核对的服务器状态

每次连接、同步、训练前：

~~~bash
REPO=/data/maoyy/JoyAI-VL-Interaction
GIT=/data/maoyy/miniforge3/bin/git
$GIT -C "$REPO" branch --show-current
$GIT -C "$REPO" status --short
~~~

当前分支必须为 feature/continuous-omni。发现工作树改动时不得 reset、checkout、覆盖或切换分支。

唯一训练环境：

~~~bash
TRAIN_PY=/data/maoyy/joyai-runtime/omni-training/.venv/bin/python
UV=/data/maoyy/.local/bin/uv
export PYTHONPATH=/data/maoyy/JoyAI-VL-Interaction:$PYTHONPATH
~~~

已验证版本：Python 3.12.13，torch 2.11.0+cu130，torchvision 0.26.0+cu130，transformers 小于 5，qwen-asr 0.0.6。

禁止修改 services/.venv、vLLM/vLLM-Omni、系统 CUDA、驱动或 /data/maoyy 外内容。GPU 前先执行：

~~~bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
~~~

没有用户明确 GPU 授权时，只能做 CPU preflight、文件检查和测试。

## 本地权重、维度、初始化

~~~text
Qwen3-ASR-1.7B:
/data/maoyy/models/Qwen3-ASR-1.7B

Qwen3-VL-4B-Instruct:
/data/maoyy/models/Qwen3-VL-4B-Instruct

Qwen3-TTS:
/data/maoyy/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
~~~

Stage2 当前加载 Qwen3-VL-4B-Instruct。Stage2 runtime 从冻结 feature cache 读取 ASR 特征，--asr-model 不能取代必需的 --feature-dir。

唯一通过 SHA256 校验的候选 Stage1 初始化：

~~~text
/data/maoyy/datasets/projector_stage1/librispeech_train100_20260804/full100h_projector_only_init95_lr3e-5_3ep_20260806_run2/best.pt
SHA256: 4e1573a3091d7ed438af16cee130d28e11b9701f5c22eb830e179e2ef315a22c
format: projector-stage1-v4
input_size=2048, output_size=4096, hidden_size=4096, dropout=0.0
~~~

旧 Stage2 checkpoints 位于 /data/maoyy/datasets/projector_stage2，例如 overfit32_mix16_projector_only_200steps_gckpt_20260810/best.pt；只用于独立 dev 对照。

## LoRA 的精确配置

Qwen3-VL-4B 本机 safetensors index 已确认：文本 decoder 共 36 层，层名为 model.language_model.layers.0 到 35。

第一条可复现的 LoRA 实验固定为全部 36 层的 self_attn.q_proj 和 self_attn.v_proj，共 72 个 target。超参：

~~~text
rank=8, alpha=16, scale=2
projector LR=3e-6
LoRA LR=1e-5
weight decay=0.01
gradient accumulation=8
max grad norm=1.0
~~~

空的 --lora-target 实际等于 projector-only。不要默认加 k/o/MLP，它们属于下一轮容量消融。

~~~bash
LORA_TARGETS=()
for LAYER in $(seq 0 35); do
  LORA_TARGETS+=(--lora-target "model.language_model.layers.$LAYER.self_attn.q_proj")
  LORA_TARGETS+=(--lora-target "model.language_model.layers.$LAYER.self_attn.v_proj")
done
~~~

启动后必须记录训练参数名。允许训练的只能是 audio_projector 和 lora_A/lora_B；ASR encoder、原始 LLM 参数必须无梯度且不更新。

## 数据状态

| 来源 | 绝对路径 | 当前状态 |
| --- | --- | --- |
| SpokenWOZ 历史基线 | /data/maoyy/datasets/projector_stage2/spokenwoz_turns_20260807/train.jsonl 和 dev.jsonl | 74,524 train / 9,169 dev；只作历史基线/schema/cache 验证。 |
| SpokenWOZ feature cache | /data/maoyy/datasets/projector_stage2/spokenwoz_feature_cache_4gpu_20260807/merged | 仅匹配 SpokenWOZ manifests。 |
| Clotho-AQA 共识 | /data/maoyy/datasets/audio_understanding_pilot/manifests/clotho_aqa_consensus.jsonl | 至少两人答案完全一致；待转换和建 cache。 |
| VoiceAssistant-400K | /data/maoyy/datasets/audio_understanding_pilot/manifests/voiceassistant_single_turn.jsonl | 已选 20,000 条；待固定 dev、转换、建 cache。 |
| LibriSpeech | /data/maoyy/datasets/projector_stage1/librispeech_train100_20260804 | ASR replay；待纳入正式 Stage2 manifest/cache。 |

Stage2 audio loader 只接受以下正式字段：

~~~text
sample_id, dialogue_id, turn_id, split, audio_path, clip_duration_ms,
clip_sha256, source_audio_sha256, user_text, assistant_response,
dialogue_history, provenance
~~~

因此 Clotho/VoiceAssistant 的 generic JSONL 不能直接传给 --train-manifest，必须先适配并为每条记录建立匹配的冻结 ASR feature cache。

视觉 retention 数据已经完成：

~~~text
图像目录：
/data/maoyy/datasets/vlm/coco_selected

训练教师 manifest，2702 条：
/data/maoyy/datasets/audio_understanding_pilot/llava/manifests_direct/teacher_train_complete.jsonl

验证教师 manifest，298 条：
/data/maoyy/datasets/audio_understanding_pilot/llava/manifests_direct/teacher_dev.jsonl
~~~

教师是本机 Qwen3-VL-4B-Instruct，确定性生成 do_sample=False、max_new_tokens=128。训练时不重新加载教师。旧 llava/images 已删除；HF coco2014_hf 的 image_id 是 Karpathy 重排索引，不能按 LLaVA 原 filename 直接匹配。

## 已知工作树改动与错误

服务器当前存在用户未提交改动：

~~~text
M training/omni/projector_stage2/train.py
~~~

该改动把视觉 teacher response 的 assistant content 改为 Qwen3-VL processor 所需的文本 content list。不得覆盖、reset、checkout 或丢失；先 CPU 验证 vision collate，再由用户决定是否提交。

已解决环境报错：

~~~text
ImportError: AutoVideoProcessor requires the Torchvision library
~~~

解决方式已执行：

~~~bash
$UV pip install --python "$TRAIN_PY" torchvision==0.26.0
~~~

验证：

~~~bash
$TRAIN_PY - <<'PY'
import torch, torchvision
from transformers import AutoProcessor
print(torch.__version__, torchvision.__version__)
print(type(AutoProcessor.from_pretrained('/data/maoyy/models/Qwen3-VL-4B-Instruct')).__name__)
PY
~~~

## 正确执行顺序

### 无 GPU 阶段

1. 核对 branch/status、模型路径、磁盘。
2. 校验视觉 teacher manifests：2702/298 行、JSONL、sample_id 唯一、回复非空、图片存在。
3. 将 Clotho、VoiceAssistant、LibriSpeech replay 转成正式 Stage2 audio manifest，固定独立 dev 和 provenance/hash。
4. 用冻结 Qwen3-ASR 对新 audio manifest 建 feature cache。
5. 跑 Stage2 preflight 和 CPU batching。旧 SpokenWOZ-only preflight 不能代表新混合数据已可训练。
6. 在小集合检查视觉 chat template、labels mask、监督 token 数、projector/LoRA 梯度。

CPU preflight 模板：替换所有尖括号路径；不加载 GPU 模型。

~~~bash
OUT=/data/maoyy/datasets/projector_stage2/<new_preflight_name>
$TRAIN_PY -m training.omni.projector_stage2.train   --train-manifest /data/maoyy/datasets/projector_stage2/<new_audio_train>.jsonl   --dev-manifest /data/maoyy/datasets/projector_stage2/<new_audio_dev>.jsonl   --vision-train-manifest /data/maoyy/datasets/audio_understanding_pilot/llava/manifests_direct/teacher_train_complete.jsonl   --vision-dev-manifest /data/maoyy/datasets/audio_understanding_pilot/llava/manifests_direct/teacher_dev.jsonl   --output-dir "$OUT"   --lora-rank 8 --lora-alpha 16   --projector-learning-rate 3e-6 --lora-learning-rate 1e-5   --gradient-accumulation-steps 8 --asr-replay-ratio 0.3   --steps 1000 --validation-every 100 --warmup-steps 100 --preflight
~~~

### GPU pilot，必须另获用户授权

新输出目录必须不存在。替换尖括号路径后：

~~~bash
export PYTHONPATH=/data/maoyy/JoyAI-VL-Interaction:$PYTHONPATH
TRAIN_PY=/data/maoyy/joyai-runtime/omni-training/.venv/bin/python
OUT=/data/maoyy/datasets/projector_stage2/<new_pilot_output>
CACHE=/data/maoyy/datasets/projector_stage2/<new_feature_cache>

LORA_TARGETS=()
for LAYER in $(seq 0 35); do
  LORA_TARGETS+=(--lora-target "model.language_model.layers.$LAYER.self_attn.q_proj")
  LORA_TARGETS+=(--lora-target "model.language_model.layers.$LAYER.self_attn.v_proj")
done

torchrun --standalone --nproc_per_node=4   -m training.omni.projector_stage2.train   --distributed --run --dtype bfloat16 --gradient-checkpointing   --llm-model /data/maoyy/models/Qwen3-VL-4B-Instruct   --feature-dir "$CACHE"   --projector-in-features 2048 --projector-out-features 4096   --init-projector-checkpoint /data/maoyy/datasets/projector_stage1/librispeech_train100_20260804/full100h_projector_only_init95_lr3e-5_3ep_20260806_run2/best.pt   --init-projector-sha256 4e1573a3091d7ed438af16cee130d28e11b9701f5c22eb830e179e2ef315a22c   --train-manifest /data/maoyy/datasets/projector_stage2/<new_audio_train>.jsonl   --dev-manifest /data/maoyy/datasets/projector_stage2/<new_audio_dev>.jsonl   --vision-train-manifest /data/maoyy/datasets/audio_understanding_pilot/llava/manifests_direct/teacher_train_complete.jsonl   --vision-dev-manifest /data/maoyy/datasets/audio_understanding_pilot/llava/manifests_direct/teacher_dev.jsonl   --output-dir "$OUT"   --batch-size 1 --gradient-accumulation-steps 8 --max-cached-feature-shards 8   --lora-rank 8 --lora-alpha 16   --projector-learning-rate 3e-6 --lora-learning-rate 1e-5   --weight-decay 0.01 --max-grad-norm 1.0   --asr-replay-ratio 0.3 --steps <pilot_steps>   --validation-every <interval> --warmup-steps <warmup_steps>   "${LORA_TARGETS[@]}"
~~~

## 评测与交接要求

独立 dev 必须分开报告：

| 能力 | 正确验证 |
| --- | --- |
| ASR | 只转写提示；规范化 WER/CER，处理数字阿拉伯/英文读法等价。 |
| 音频对话 | 生成回复提示；任务成功、slot/entity、一致性、人工或独立 LLM judge。 |
| 环境音 QA | Clotho-AQA 共识 dev；报告 exact/语义接受度及粒度、时态、单复数偏差。 |
| 视觉保留 | 298 条视觉 teacher dev；检查视觉 loss/生成未明显退化。 |

每次接手必须报告：branch/status；当前阶段；audio manifest、feature cache、vision manifest、base model、projector checkpoint 的绝对路径；LoRA target/rank/alpha；GPU PID/GPU/日志/输出/进度；错误与不破坏文件的恢复方案。

不得删除 /data/maoyy 文件；不得切换主力服务器分支；不得提交权重、原始媒体或日志。代码修复只提交相关源码，先推送 GitHub，再同步服务器。
