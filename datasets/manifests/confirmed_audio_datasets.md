# 已核验可用音频来源

本清单只收录官方资料明确支持音频、且原始数据中存在可获取媒体的来源。它们仍需在下载后逐文件运行 ffprobe，因为不是每个视频都必然有音轨。

| 来源 | 样本数 | 核验结论 | 可用于 Omni 的部分 | 获取限制 |
|---|---:|---|---|---|
| ego4d（含 ego4d_vqa、EgoQA 等派生标注的母视频） | 39,466 | Ego4D 官方页面明确写明每个视频提供 Audio availability 元数据，并设有 AV Diarization/Social 基准 | 原始视频音频、视频-音频同步、说话人/社会交互辅助监督 | 需签署许可并获得 AWS 凭据；须按元数据筛选有音频的视频 |
| holoassist | 13,838 | 官方说明称每个录制会话采集七路同步数据流；指导者实时通过语音指导执行者；官方发布视频压缩包和标注 | 视听交互、语音指导、干预/抢话相关场景、视频音频对齐 | CDLAv2；视频发布包约 145-184 GB，需下载后抽样核验编码 |
| epickitchens | 7,974 | EPIC-Sounds 官方说明明确写明数据来自 EPIC-KITCHENS-100 的 audio stream | 厨房场景中的人声、环境声、可听动作和视听对齐 | 需遵守原始数据访问条款；不是助手对话数据 |
| egoexo4d（含 egoexolearn） | 3,708 | Ego-Exo4D 官方文档明确写出每段视频伴随 7-channel audio，并提供时间索引的第一人称叙述和第三人称语音评论 | 多视角视听同步、语音叙述、专家评论、交互场景 | 需申请数据访问；按版本下载 |

## 其余来源的核验结果

这里的“媒体依赖”不是说 JoyAI 样本一定没有音频，而是说不能仅凭来源名称判断。JoyAI 的定案单位必须是具体的 source + video_name：

1. 从 JoyAI 行中提取实际出现的 video_name。
2. 按上游数据集的官方元数据，将该名称连接到原始媒体文件。
3. 对连接成功的文件运行 ffprobe，确认是否有 Audio 流及其时长。
4. 只把连接成功且有可用音频流的 JoyAI 样本纳入 Omni 音频训练。

因此，VideoChat2、LLaVA-Video-178K、Molmo2 等混合来源不能整体判为“无音频”。它们在 JoyAI 中的每个视频必须先完成母数据集归属和媒体连接；只有实际连接到的那些视频才进入音频核验。

| 来源 | 核验状态 | 原因 |
|---|---|---|
| Kinetics-400/600/700 | 媒体依赖，未确认 | 对 JoyAI 实际 video_name 逐条回溯 YouTube 媒体；下载器可抽取 sound track，但没有统一音频发布保证 |
| ActivityNet | 媒体依赖，未确认 | 对 JoyAI 实际 video_name 逐条回溯公开视频；音频随失效/替换视频变化 |
| YouCook2 | 媒体依赖，未确认 | YouTube 视频来源，官方标注不承诺音频流 |
| LSMDC | 可能含音频，未确认 | 电影片段通常有音轨，但数据访问受控，当前没有可核验媒体样本 |
| WebVid2M/10M | 未确认 | 只核验 JoyAI 中实际出现的视频 URL/媒体，不能由标注推断音频 |
| DiDeMo、NExT-QA | 未确认 | 视频问答标注不包含统一音频字段，母视频来源不一致 |
| NTU RGB+D、Assembly101、Something-Something V2 | 未确认 | 核心任务是动作/RGB-D，官方资料未把音频作为稳定发布模态 |
| UCF-Crime、UVO、OOPS、FAVD | 未确认 | 视频可能带原始音轨，但没有统一、可复核的音频发布说明 |
| EgoIT、EgoIT-99K、EgoLife、EgoProceL、Vript、EgoBlind、Perception Test | 未确认 | 多为派生或多母集视频标注，必须追溯具体媒体文件 |
| Live-WhisperX | 不作为独立来源 | 更像转写/处理管线名称，不能据此确认原始音频归属 |
| CharadesEgo | 确定不可用（已抽样） | 仓库已有媒体审计未发现可用音轨 |
| Molmo2-*、Open-o3-Video、VideoChat2、LLaVA-Video-178K、TimeLens-100K、VideoInternSeg、VideoGPT-plus、EgoSchema、EgoTaskQA、omnistar | 混合来源/派生标注 | JoyAI 本身没有原始媒体；必须按每条 source + video_name 回溯母数据集，不能整体判定音频状态 |
| TGIF、CLEVRER | 无音频主模态 | GIF 或合成视频任务，音频不是原始数据模态 |

## 官方证据

- Ego4D: https://ego4d-data.org/  
  页面包含 “Information about the availability of IMU, Audio ...” 以及 AV Diarization 基准说明。
- HoloAssist: https://holoassist.github.io/  
  页面说明七路同步数据流、语音指导、视频下载链接和 CDLAv2 许可。
- EPIC-Sounds: https://epic-kitchens.github.io/epic-sounds/  
  官方说明其数据来自 EPIC-KITCHENS-100 的 audio stream。
- Ego-Exo4D: https://docs.ego-exo4d-data.org/  
  官方文档明确写出 “The video is accompanied by 7-channel audio”。

## 使用结论

当前可以直接进入 Omni 音频数据准备阶段的来源级候选为 ego4d、holoassist、epickitchens 和 egoexo4d。对于 VideoChat2 等混合来源，必须先建立 JoyAI 实际视频的逐条媒体映射。最终纳入仍以具体文件为单位。每个文件必须记录：

- 是否存在音频流；
- 编码、采样率、声道数；
- 音频与视频时长及起始时间偏差；
- 是否包含人声、多人重叠或仅环境声；
- 原始文件名、许可和校验哈希。

## 新增考量范围

FAVDBench 已加入候选数据集考量范围：

- 页面：https://opendatalab.com/OpenNLPLab/FAVDBench
- 当前状态：待核验其原始媒体是否包含音频、音频字段/标注结构和许可。
- 处理原则：如果 FAVDBench 的 JoyAI 关联样本能映射到实际媒体，则按具体 video_name 做 ffprobe；不能因为数据集名称或说明推断音频存在。

## 项目默认规则

对于 YouTube 等公开视频来源，项目允许先按“默认有音频流”纳入候选范围；删除、private、unavailable、静音或实际无音频的文件作为样本级缺失处理，不再因此否定整个来源。最终训练仍以成功映射且有可听内容的 JoyAI video_name allowlist 为准。静音文件只在需要 silence/no-speech 负样本时保留。
