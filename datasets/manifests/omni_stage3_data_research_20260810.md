# Omni 下一阶段数据调研与 pilot 规划

更新日期：2026-08-10

本文件是只读仓库审计和官方来源调研结果。当前没有下载新数据、没有启动 GPU、没有加载模型，也没有修改 Stage1/Stage2 产物。

## 1. 结论摘要

- 当前没有任何新增来源可以在“媒体真实可得、许可证已核验、可转换、可隔离、可生成 provenance”全部完成前直接标记为 pilot-ready。下面的 pilot 清单均为“用户授权下载后执行”的条件清单。
- 500 GB 条件下，首批主线以 `datasets/manifests/source_audio_status_checklist.md` 的“明确有音频”栏目为唯一来源级优先清单：ActivityNet、YouCook2、Kinetics-400/600/700、Ego4D、HoloAssist、EPIC-KITCHENS、Ego-Exo4D、VideoInstruct-100K、VideoInternSeg、Vript、TextVR、Molmo2 两个子集、VideoChat2 的 YouTube 子集、Live-WhisperX 与 omnistar。按 JoyAI 的具体样本映射、媒体可得性和固定容量配额回收，不因来源体量大而整体排除。Common Voice、AMI、Nonspeech7k、Clotho/FSD50K 等外部来源仅作为后续辅助能力数据，不复用 JoyAI 的 Interaction 标签。
- 现阶段最明显的缺口是带真实人声的多轮交互、明确的 agent barge-in/interrupt 标签，以及经许可的 silence/no-speech 负样本。AMI 和 AISHELL-4 可以覆盖重叠语音、轮次和 VAD 代理任务，但不能被表述为已经覆盖真实全双工 agent 打断。
- AudioCaps、AVQA、AVSpeech、AudioSet 原始媒体等 URL/YouTube 依赖来源不进入正式训练 pilot；它们最多作为标注层或评测参考，不能把可访问 URL 当作可审计原始媒体。

## 2. 现有资产与不可改变的事实

### 仓库与本地审计

- 当前分支为 `feature/continuous-omni`。
- 工作树已有他人修改和未跟踪文件；本次没有覆盖、回退或删除它们。
- `datasets/audit_output/report.md`：本地 `chat_shard_06.json` 有 213,709 条标注、69,472 个唯一 `(source, video_name)`，问题/回答时间戳之和为 305.031 小时的媒体时长下界，不是实际媒体总时长。
- 本地 JoyAI 标注主要是视觉交互问答：203,641 条（95.29%）问题和回答时间相同。它们不能被当作原生语音全双工记录。
- `datasets/audit_output/multimodal_audio_allowlist.summary.json`：已有逐文件抽样 38 条，37 条有可探测音轨、1 条静音。已有样本的音轨语义是 `original_scene_audio_not_user_speech`，不能当作用户语音转写。
- `datasets/manifests/projector_stage1_librispeech_samples.summary.json`：当前本地 Stage1 manifest 已转换 95 条 LibriSpeech 样本。LibriSpeech 已承担 Stage1 音频—文本对齐，不在下一阶段重复计数。

### 已继承的边界

- SpokenWOZ train/dev 已用于 Stage2；后续训练不能重新使用其 dev，test 不能用于调参。
- Ego4D、HoloAssist、EPIC-KITCHENS/EPIC-Sounds、Ego-Exo4D 是值得优先核验的音视频来源，但必须按许可和真实媒体逐文件验收。
- ActivityNet、YouCook2、Kinetics 等只有本地抽样音频证据，不代表所有媒体均有可用音轨。
- CharadesEgo 的现有抽样 20/20 无音轨；不能重新标为可用。
- JoyAI 问题是 `user_text` 或视觉交互标注，不等于视频中有人说出了问题。
- schema `omni-training-v1` 要求每条样本保留 dataset/version/source/license 和媒体路径、时间范围、采样率、样本数；`targets` 只允许 `silence/response/delegate/interrupt` 四类动作。音频类型应继续区分 `user_speech`、`other_speech`、`environment_audio`、`music`、`mixed_audio` 和 `silence_or_no_speech`。

## 3. 去重与转换规则

1. LibriSpeech 不再作为新增 Stage1 数据计数；新增 ASR 仅用于口音、录音条件和语言多样性补强。
2. SpokenWOZ 的官方 split 固定保留，后续只使用已批准的训练范围；已有 dev 不得进入训练。
3. Clotho-AQA 使用 Clotho 音频，若同时纳入 Clotho caption，按同一音频一个 `sample_id`、多个任务 target 处理，不能按两份音频重复计数。
4. FSD50K、Clotho 和其他 Freesound 来源必须按 Freesound 原始文件 ID、URL 和 SHA256 去重；逐文件许可证决定是否可进入目标许可层。
5. JoyAI、ActivityNet、YouCook2、Kinetics、AudioCaps、AVQA、AVSpeech 等 YouTube/公开视频来源必须以 `source + video_name + upstream_path` 为媒体主键，并在下载后使用 ffprobe；不能按 dataset 名称或 URL 命中推断音频存在。
6. 训练/开发/测试按 speaker、dialogue、source media、规范化路径和媒体 SHA256 隔离。同一视频切片、同一对话 turn、同一 Freesound uploader 或同一音频 hash 不得跨 split。
7. ASR transcript 是音频监督目标，不是 user 输入；转换时沿用现有 Stage1 的 `metadata.assistant_target_text` 语义，必要时把 transcript 作为 `text` 的 `auxiliary=true`，但不能用文字替代音频。

## 4. 候选来源表

优先级含义：P0 表示调研优先且适合授权后做小 pilot；P1 表示价值明确但仍有许可、split、媒体或体量门槛；P2 表示后续补充；拒绝表示当前不进入正式训练。

| 来源/版本 | 模态与能力桶 | 官方证据、规模和监督 | 许可/获取 | 去重与主要风险 | 优先级 |
|---|---|---|---|---|---|
| Common Voice Scripted Speech 25.0；Spontaneous Speech 3.0 | audio-text；ASR 锚定、口音/设备多样性 | Mozilla Data Collective 官方下载表；按 locale 提供 MP3 和文本，当前页面列出中文（中国）整包 21.38 GB、英语 spontaneous 459.05 MB 等 | 页面列 CC0-1.0；需通过官方 Data Collective 获取当前版本 | 单说话人短句为主，缺少多轮回复；按 clip hash、speaker metadata 和 locale split 审计 | P2（外部 ASR 辅助，不复用 JoyAI Interaction 标签） |
| AMI Meeting Corpus | audio-video-text；多说话人、重叠、dialogue act、时间线 | 官方说明 100 小时会议，多麦克风、视频、正字转写、dialogue acts、head movement等；官方提供 scenario-only/full-corpus 的 train/dev/test 建议划分 | CC BY 4.0；下载页要求注册，部分高分辨率视频需另行取得 | 会议语域和任务型助手不同；不能套用 JoyAI `silence/response`，须使用其自身时序标签 | P2（外部时序交互辅助） |
| HoloAssist | audio-video；实时语音指导、错误纠正、干预和动作预测 | 官方项目页：169 小时、350 对 instructor-performer、七路同步数据流，提供动作/对话标注和 train/val/test；压缩视频约 144.62 GB | 官方页面称 CDLAv2 permissive；商业使用和下游再分发仍需按 CDLAv2 原文逐条核对 | JoyAI 已含 `holoassist` source；先以 `source + video_name` 映射，保留 JoyAI 的 Interaction 时间线 | P0（JoyAI 母媒体主池） |
| Nonspeech7k v1 | audio；非语音人声、环境/非语音负样本 | Zenodo 官方页：7,014 个 32 kHz mono WAV，train 6,289、test 725，单文件 0.5–4 秒；train.zip 2.3 GB、test.zip 221.8 MB | CC BY-NC-SA 4.0，仅学术/非商业；来源含 Freesound、YouTube、Aigei，需保留来源字段 | 覆盖人类非语音声，不是静音本身；不能把 label 直接当 `silence_or_no_speech` | P1（非商业实验） |
| FSD50K v1.0 | audio；环境声/声音事件 | Zenodo 官方页：51,197 clips、108.3 小时、200 类、0.3–30 秒、16-bit 44.1 kHz mono；dev 40,966/80.4h，eval 10,231/27.9h；dev 音频 24.7 GB、metadata 6.7 MB | 每条 clip 独立 CC0/CC-BY/CC-BY-NC/CC Sampling+；官方明确商业使用需联系作者 | 不能把整库标成单一商业许可；先只取 CC0/CC-BY，逐文件保存 licence URL 和 attribution | P1 |
| Clotho v2.1 | audio-caption；音频描述、环境声问答前置 | Zenodo 官方页：6,974 音频、34,870 captions、15–30 秒；dev 3,840、validation 1,046、evaluation 其余；完整文件 7.1 GB | 音频和 metadata 继承 Freesound 每文件许可证；caption 主要为 Tampere University 非商业署名许可 | 不是 speech transcript；caption 不能作为用户语音；不得按 caption 数量重复音频 | P1 |
| Clotho-AQA | audio-text QA；audio QA | 论文与 Zenodo：1,991 个 Clotho 音频，每个 6 个问题，yes/no 与单词答案；复用 Clotho 音频 | 继承 Clotho 音频和 caption 的逐文件许可，须另核对 AQA 标注许可 | 与 Clotho 音频高度重叠；只作为新增 QA target，不新增音频计数 | P1 |
| AISHELL-4 SLR111 | audio；中文会议 ASR、VAD、speaker diarization、overlap | OpenSLR 官方页：211 场、4–8 speaker、120 小时、8-channel array；有 transcript 和 speaker voice activity；train_L/M/S 共约 46 GB，test 5.2 GB | CC BY-SA 4.0；官方页提供 train/test，没有明确 dev | 只有 train/test，需在 train 内按 session/room/speaker 做内部 dev；不能与 SpokenWOZ 的任务型语音混为同一能力 | P1 |
| MInDS-14 | audio-text；14 intent、14 language varieties | 官方 Hugging Face card：16,336 rows、14 intents、14 language varieties；每 config 约 600 train examples，8 kHz，约 1.13 GB 总文件 | CC BY；只有 train split，需自建 speaker/recording-level dev | e-banking 单域、短句、无多轮回复；与 SLURP/SpokenWOZ 的 intent overlap 要做文本/音频 hash 和 intent 分布审计 | P1 |
| SLURP | audio-text；18 domains、intent/action/entity、spoken SLU | 作者官方仓库和论文：约 72k recordings、约 58 小时 acoustic data，提供 intent、action、entity 和 close/far microphone；音频下载约 6 GB | 文本 CC BY 4.0；Zenodo 音频 CC BY-NC 4.0，可联系作者申请更宽许可；含 `slurp_real` 与 `slurp_synth` | 只使用真实录音子集；合成语音不进入本阶段正式训练；需确认官方 split 和实际 real/synth 时长 | P1 |
| Ego4D v2 | audio-video-text；第一人称 AV、AV diarization、social interaction | 官方文档：3,600 小时级第一人称视频，部分视频含音频；AV diarization 需要说话人活动定位和转写；全量 primary 约 7.1 TB | 必须签署 license agreement，审批约 48 小时并取得 AWS credentials；访问和再分发受协议控制 | JoyAI 已含 `ego4d/ego4d_vqa/egoqa/EgoQA`；仅下载能映射、有音频且在配额内的母视频/clip | P0（JoyAI 母媒体主池） |
| Ego-Exo4D v2 | audio-video-text；同步多视角、专家语音评论、技能动作理解 | 官方文档：1,286.30 video hours、5,035 takes，视频伴随 7-channel audio；GoPro 音频 48 kHz stereo AAC；takes 约 10.55 TB，annotations 约 10.5 GB | 需单独 license agreement 和 AWS 配置，官方说明审批约 2 天 | JoyAI 已含 `egoexo4d/egoexolearn`；按 take ID、participant 和媒体 hash 隔离，仅取配额内命中媒体 | P0（JoyAI 母媒体主池） |
| EPIC-SOUNDS / EPIC-KITCHENS-100 | audio-video；动作发声、环境音、视听同步 | 官方页：75.9k audible-event segments、44 classes，来自 EPIC-KITCHENS-100 audio stream；底层 EPIC-KITCHENS-100 约 100 小时、45 kitchens | CC BY-NC 4.0；商业使用需联系作者 | JoyAI 已含 `epickitchens`；主要是厨房环境和动作声，按命中片段保留 JoyAI 时间线 | P1（JoyAI 母媒体补充） |

## 5. 暂缓或拒绝来源

- **AudioSet**：官方只发布 YouTube ID/时间段/标签和预计算特征，不发布可直接用于本地训练的原始 waveform；不能把标签表当原始音频。
- **AudioCaps**：官方说明约 46k audio-caption pairs，音频来自 AudioSet；依赖 YouTube/AudioSet 原始媒体，当前未核验逐文件媒体权利和可再分发性，暂不进入训练。
- **AVQA**：官方标注来自 VGG-Sound，原始视频仅提供 YouTube URL/时间戳；官网许可限定个人/课堂使用并限制再发布，拒绝进入正式训练。
- **AVSpeech**：官方说明约 4,700 小时、29 万 YouTube 视频，3–10 秒片段；页面未给出可满足项目要求的原始媒体再分发许可，拒绝进入正式训练。
- **InteractSpeech**：论文报告 150 小时、打断和 backchannel 标签，但数据同时包含由文本经 TTS 合成的交互对话和筛选出的真实片段；当前未核验数据集许可证及 real/synthetic 完整清单，不进入正式训练。可作为后续实时交互评测/研究候选。
- **Fluent Speech Commands**：官方页面注明 strictly academic research only，不进入需要更宽许可的正式训练。
- **GigaSpeech**：官方仓库要求填写表单后获取 raw release；代码是 Apache-2.0 不代表原始音频同样是 Apache-2.0，当前不把它标为可商用或 pilot-ready。
- **JoyAI 混合来源中的 WebVid、DiDeMo、NExT-QA、LLaVA-Video-178K、VideoChat2、Vript、TimeLens 等**：已有本地抽样不能扩展到整个混合来源；必须先按母数据、官方媒体索引、许可证和逐文件 ffprobe 拆分。

## 6. 能力桶统计与缺口

| 能力桶 | 已有可计数资产 | 新候选可提供的有效监督 | 当前缺口 |
|---|---|---|---|
| ASR 锚定 | LibriSpeech 已用于 Stage1；本地 Stage1 manifest 95 条 | Common Voice、AISHELL-4、SLURP real subset、MInDS-14 | 口音/远场/多人中文覆盖需要补强，但不能让 ASR 占据联合训练主体 |
| 语音指令与多轮对话 | SpokenWOZ train/dev 已用于 Stage2；dev 禁止再训练 | SLURP、MInDS-14、AMI dialogue acts | 新增来源多数是单轮或会议，不足以替代真实 task-oriented 多轮语音 |
| 非语音音频 | JoyAI 已确认音轨样本仅为场景音；Kinetics 抽样有 1 条静音 | FSD50K、Clotho、Nonspeech7k、EPIC-SOUNDS | 缺少可靠的 `silence_or_no_speech` 和区分 user/other speech 的统一标注 |
| 音视频联合 | JoyAI 38 条逐文件音频 allowlist；Ego4D/HoloAssist/EPIC/Ego-Exo 候选 | HoloAssist、Ego-Exo4D、AMI、EPIC-SOUNDS | 需要真实媒体、音视频同步、说话人/音源角色及法律许可 |
| 实时交互与时间线 | JoyAI 有 response/silence 形式的视觉时间标注，但不是原生语音 | AMI/AISHELL-4 可提供 overlap、VAD、turn boundary 代理；InteractSpeech 有研究价值 | 没有已许可、真实录制、明确 agent response/interrupt/delegate 的统一训练集 |
| 能力回放 | 现有 JoyAI VLM/文本资产可保留独立 replay | 只增加小比例 text-only/vision-language，不引入大规模新视觉数据 | 需要独立原始 VLM/text 基准，防止联合训练退化 |

## 7. 外部辅助数据的条件 pilot 清单

以下来源不属于 JoyAI 母媒体回收主线；仅在主线稳定、且其自身监督可转换为独立辅助任务后考虑。它们不得复用 JoyAI 的 `silence/response` 时间线。下载统一写入用户指定的 `/data/maoyy/datasets/omni_stage3/raw/<dataset>/<version>/`，转换、审计、manifest 和 provenance 使用独立目录。

| Pilot | 建议范围 | 官方下载体量/规划预算 | 授权与验收 |
|---|---|---:|---|
| CV-25-locale | 选择一个 locale 的 5k–20k clips，优先包含 speaker diversity；不下载整包 | 官方整包示例：中文（中国）21.38 GB；pilot 预算先按 0.5–2 GB 预留，实际以官方 split manifest 为准 | CC0-1.0；固定官方 train/dev/test 或公开 split，按 speaker、clip hash、文本 hash 审计，ffprobe/解码/采样率核验 |
| AMI-small | 2–4 个完整 meeting/session，下载 audio、transcript、dialogue-act 和可用低分辨率 video | 官方页面未提供统一全库单文件大小；先按 chooser 选择，预算按 2–10 GB 预留，实际下载前确认 | CC BY 4.0，需注册；按 meeting/session 隔离，核验多通道、重叠、speaker turn、视频音轨和时间戳 |
| HoloAssist-small | 先取 labels 及 train/val/test split，再按 participant pair 选择少量 compressed videos | labels 111 MB；全 compressed video 144.62 GB，pilot 只预算 5–15 GB，实际以选中清单为准 | CDLAv2 条款逐项确认；按 instructor-performer pair 隔离，ffprobe 音频/视频，核验语音指导、干预和 action 时间标注 |
| Nonspeech7k-train-small | 从官方 train 中抽取少量非语音类别，保留 test 仅作评测 | 官方 train.zip 2.3 GB；test.zip 221.8 MB 仅用于评测 | CC BY-NC-SA 4.0；按文件 label 归入 `other_speech`/`environment_audio`/`music` 等，不能直接标成 silence |
| Clotho-dev-filtered | 先下载 metadata/许可证表；若没有逐文件选择下载入口，再决定是否接受 dev archive | 官方 dev archive 4.5 GB、全库 7.1 GB；只过滤后再预算 | 音频逐文件 Freesound 许可，caption 为非商业署名许可；按 uploader/source ID 去重，人工确认 caption 与音频一致 |
| FSD50K-safe-subset | 先下载 metadata 6.7 MB 和 ground truth 334.7 kB；只在作者许可和下载方式允许时取 CC0/CC-BY 音频 | dev 音频官方分片合计 24.7 GB，不能为了小样本默认下载全库 | 逐文件保留许可证、attribution 和 SHA256；不混入 CC-BY-NC、CC Sampling+，不把全库 CC-BY 声明为商业可用 |

Ego4D 与 Ego-Exo4D 已移入第 11 节的 JoyAI 母媒体主线，但仍受访问许可、逐条映射和空间配额约束；AISHELL-4、SLURP 等保留为外部辅助数据，待主线 pilot 稳定后再评估。

## 8. 建议的 Omni 混合比例

比例按训练 batch/有效音频时长计算，不按原始文件数计算；每个来源还要设置单源上限，防止大量短片段制造虚假的多样性。

| 能力桶 | 建议比例 | 首批 100 有效小时的目标量 | 理由 |
|---|---:|---:|---|
| ASR 锚定 | 30% | 30 h | 保住音频到文本的基础对齐，但低于一半，避免模型只学转写 |
| 语音指令/多轮对话 | 25% | 25 h | 直接支持 assistant reply、intent、slot/entity 和历史上下文 |
| 非语音音频 | 15% | 15 h | 引入环境声、非语音和人声状态，防止所有音轨被解释成 user speech |
| 音视频联合 | 20% | 20 h | 保持 VLM 任务和音频事件的联合推理能力 |
| replay（vision/text） | 10% | 10 h | 防止原有视觉/语言能力遗忘；单独监控原始基准退化 |

这只是 pilot 的起始配比。每个能力桶都要有独立 dev/test 与指标：ASR WER/CER；语音对话 Inform/Success/slot/entity F1；audio QA accuracy/semantic score；音视频事件准确率和时间定位；实时任务的静默误报、漏报、响应延迟与动作准确率；replay 的原始 VLM/text 基准不退化。若某一桶在独立验证集退化，应按桶重采样或降权，不能以总 loss 下降作为唯一依据。

## 9. 转换与验收门槛

在任何来源进入训练前，必须生成不改写原始媒体的内部 JSONL，并通过以下检查：

- `sample_id` 唯一；`dataset/version/split/source_uri/license` 完整；
- train/dev/test 在 speaker、dialogue、source media、规范化路径和 SHA256 层面零交集；
- 每个音视频文件 ffprobe 记录 codec、采样率、声道、时长、时间起点/终点和解码状态；
- 静音、损坏、缺失、private、deleted、替换媒体分别记录，不用 URL 成功代替媒体成功；
- user/system/history/audio placeholder/padding 不进入 assistant loss；
- 原始音频和文字转写、assistant reply、环境音 caption、动作 target 的角色明确分离；
- `research_only` 与 `redistributable` 许可层不混为一个可商用 checkpoint；
- 每个来源保存原始 manifest、转换 manifest、provenance、license snapshot、内容 hash 和审计摘要。

## 10. 需要用户确认的事项

1. 是否允许在后续步骤下载首批 pilot；允许时请明确目标服务器和目录，仍只写入 `/data/maoyy/datasets/omni_stage3/raw/`。
2. 目标 checkpoint 是否接受研究/非商业许可数据（SpokenWOZ、SLURP 音频、Nonspeech7k、EPIC-SOUNDS、Clotho caption 等）；如果不接受，应只保留 CC0/CC-BY/已确认更宽许可的样本。
3. 是否把 Common Voice 的哪个 locale 作为 ASR 多样性 pilot；默认建议先选一个中文或英语 locale 的小 split，不下载整包。
4. 是否优先 AMI 的会议重叠/轮次代理任务，还是优先 HoloAssist 的语音指导/干预任务；两者都不是现成的 agent `interrupt/delegate` 监督。
5. 是否允许为 AISHELL-4 从官方 train 按 session 建立内部 dev；其官方发布页面没有明确 dev split。
6. 是否接受仅用于研究的 Nonspeech7k/Clotho/FSD50K 子集进入非商业实验；商业或再分发目标必须另行审核。

在这些选择得到确认前，本阶段停止于调研、规划和 CPU-only 审计，不开始下载、GPU 验证或训练。

## 11. 500 GB 服务器：JoyAI 母媒体优先收集方案

### 目标与转换原则

本计划的准入条件不是“必须属于 JoyAI 的原始来源”，而是能够被可靠转换为 JoyAI 的实时 Interaction 时间线：连续音视频输入、少量明确的 `response`（以及未来可能的 `interrupt`/`delegate`）时刻，其他秒均为 `silence`。JoyAI 自身来源的媒体优先，是因为已有 `source + video_name + question/response time` 映射，可直接复用其秒级 `silence/response` 转换规则；视频原始音轨可以是场景声、他人说话、动作声或混合声，不要求是用户语音。

外部音视频数据后续可以作为复杂声学理解补充，但不得只因“没有答案”就整段填为 `silence`；它们需要自己的事件、问答或时序行为监督，并与 JoyAI Interaction 主数据分桶统计。

### 空间上限与存储纪律

服务器可用空间固定为 **500 GB**。本节的来源选择以 `source_audio_status_checklist.md` 中已经“明确有音频”的来源为准；它们均进入首批队列，而非只优先其中的 HoloAssist/Ego 系数据。原始媒体、审核产物和转换结果都写入 `/data/maoyy/datasets/omni_stage3/`；不保存逐秒抽帧，也不保留可由原媒体重建的中间视频。所有来源先完成 `source + video_name -> official media ID/path` 映射，再按批次下载、`ffprobe`、生成 manifest；任一来源达到配额即停止扩容。

| 空间桶 | 上限 | 用途 |
|---|---:|---|
| HoloAssist 压缩视频与标签 | 145 GB | 完整下载；官方压缩视频规模约 144.62 GB，优先保留其 JoyAI 命中会话与 labels |
| ActivityNet 与 VideoInstruct-100K 已映射媒体 | 55 GB | VideoInstruct-100K 文件名可指向 ActivityNet；两者合并去重后按 JoyAI 具体文件回收 |
| YouCook2 已映射媒体 | 30 GB | 已有 10/10 音轨抽样证据；优先料理过程中的持续场景音与事件时序 |
| Kinetics-400/600/700 已映射媒体 | 70 GB | 三个来源均有已解码音轨抽样；跨版本按原媒体 ID/hash 去重 |
| Ego4D 已映射 JoyAI 片段 | 55 GB | 仅下载 JoyAI `ego4d/ego4d_vqa/egoqa/EgoQA` 实际命中的有音频母视频/官方 clip，不下载约 7.1 TB 全库 |
| Ego-Exo4D 已映射 JoyAI takes | 35 GB | 仅取 JoyAI `egoexo4d/egoexolearn` 命中的带音频 take/导出片段，不下载约 10 TB 全库 |
| EPIC-KITCHENS/EPIC-Sounds 已映射片段 | 20 GB | 厨房动作、场景声和视听同步；只取 JoyAI 命中片段 |
| 已验证的 YouTube 子集 | 35 GB | VideoInternSeg、Vript、TextVR、Molmo2-AskModelAnything、Molmo2-VideoCapQA、VideoChat2-YouTube、Live-WhisperX、omnistar；只收录清单中已能恢复且 ffprobe 通过的样本 |
| manifest、provenance、审计帧与转换特征 | 30 GB | JSONL、许可快照、哈希、少量人工审计帧和可复用音频特征；不保存全量帧 |
| 受控下载与应急余量 | 25 GB | 单批下载、失败重试和容量估计误差；不得被常规媒体长期占用 |
| **合计** | **500 GB** | |

### 获取顺序

1. **先完成全栏目的映射清单**：对“明确有音频”的每一个 source 生成 JoyAI `source + video_name`、母媒体 ID、许可、预计大小和已验收状态。没有映射的来源不下载，但不从优先清单删除。
2. **HoloAssist（145 GB）**：先 labels/split/小样本验收，再完整下载压缩媒体；它占用最大固定配额，但不是唯一首批来源。
3. **ActivityNet/VideoInstruct、YouCook2、Kinetics（155 GB）**：利用既有抽样音频证据，按来源各自 5 GB 小批量轮转下载和验收，避免一个来源先耗尽容量。
4. **Ego4D、Ego-Exo4D、EPIC（110 GB）**：完成授权和官方索引映射后，按配额下载命中媒体；许可或凭据未就绪的额度临时让给本节其他已验证来源，但不得改作外部非 JoyAI 数据。
5. **已验证 YouTube 子集（35 GB）**：覆盖 VideoInternSeg、Vript、TextVR、Molmo2、VideoChat2-YouTube、Live-WhisperX、omnistar；优先已 `ffprobe` 成功的媒体，失效 ID 单独记录。
6. **每一轮完成后按 hash 去重和容量复盘**：从未使用的“明确有音频”来源补足配额，不下载“无法读取，后续再处理”栏目中的来源。

### 每条样本的必备记录

- JoyAI 对齐样本必须保存 `joyai_source`、`joyai_video_name`、原始问题/响应时间、官方母媒体 ID/路径、版本、许可证、哈希和实际时长。
- Omni 转换保留原始音频通道（如 `environment_audio`、`other_speech` 或 `mixed_audio`）和视频；在 response/event 秒写显式 target，所有未写 target 的秒按协议自动成为 `silence`。
- 音频存在不等于语义已验证：每批必须记录音画同步、可听内容、无声比例和与 JoyAI 时间线的对应状态。
- 外部复杂音频理解数据单列为 `external_audio_understanding`，不复用 JoyAI 的 `silence/response` 标签；只有具备自身可靠时序行为标签时才进入联合动作训练。

### 停止条件

- 任意媒体下载前，计算该来源已用空间与本表配额；超过配额不下载。
- 一批在映射、许可、音频流、时长或解码任一项失败时，停止该来源扩容，保留审计记录而不以 URL 或 source 名称推定成功。
- 当总占用达到 475 GB 时，停止新增常规媒体；仅允许清理可再生产物、写入 manifest/审计，或使用预留的 25 GB 完成已在途批次。任何扩容必须先释放同等空间。

## 12. 官方来源

- Common Voice 下载与版本列表：https://commonvoice.mozilla.org/en/datasets
- AMI Corpus：https://groups.inf.ed.ac.uk/ami/corpus/；下载与 split：https://groups.inf.ed.ac.uk/ami/download/、https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml
- HoloAssist：https://holoassist.github.io/
- Nonspeech7k：https://zenodo.org/records/6967442
- FSD50K：https://zenodo.org/records/4060432
- Clotho v2.1：https://zenodo.org/records/4783391
- Clotho-AQA：https://arxiv.org/abs/2204.09634；Zenodo：https://zenodo.org/records/6473207
- AISHELL-4：https://www.openslr.org/111/
- MInDS-14：https://huggingface.co/datasets/PolyAI/minds14
- SLURP 作者仓库：https://github.com/pswietojanski/slurp；论文：https://arxiv.org/abs/2011.13205
- Ego4D：https://ego4d-data.org/docs/start-here/；AV diarization：https://ego4d-data.org/docs/benchmarks/overview/
- Ego-Exo4D：https://docs.ego-exo4d-data.org/；下载：https://docs.ego-exo4d-data.org/download/
- EPIC-SOUNDS：https://epic-kitchens.github.io/epic-sounds/site
- AudioSet：https://research.google.com/audioset/download.html
- AVQA：https://mn.cs.tsinghua.edu.cn/avqa/
- AVSpeech：https://looking-to-listen.github.io/avspeech/
- InteractSpeech：https://interactspeech.github.io/；论文：https://aclanthology.org/2025.findings-emnlp.424/
