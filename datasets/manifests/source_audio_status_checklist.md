# JoyAI source 音频判别清单

本清单按本地 6 个 chat shard 扫描得到的 56 个命名 source 建立；另有 19,680 个具体 YouTube/Bilibili URL 值（69,714 条 URL 来源记录）。URL 必须按实际媒体逐条核验，不能只按域名判定。每次完成新的媒体映射或 `ffprobe` 抽样后，更新本文件和对应逐文件 manifest。

“明确有音频”表示已有官方音频说明或本地抽样 `ffprobe` 证据，不代表该 source 的所有 JoyAI 样本都已下载。视频问答文本也不能当成音频转写。

## 1. 尚未读取、判别

- egoblind
- Perception Test
- EgoProceL
- shot2story
- Open-o3-Video
- Molmo2-AskModelAnything
- Molmo2-VideoCapQA
- VideoInstruct-100K
- VideoChat2
- LLaVA-Video-178K
- VideoInternSeg
- VideoGPT-plus
- TimeLens-100K
- Vript
- TextVR
- MovieChat
- EgoTaskQA
- EgoSchema
- omnistar
- TransRAC
- STAR / Charades
- GUI-World
- gui_world
- NTU RGB+D
- assembly101
- Something-Something V2
- ucfcrime
- UVO
- OOPS
- FAVD
- Live-WhisperX

## 2. 明确有音频

- ActivityNet（抽样 8/8；后续仍需扩大抽样）
- youcook2 / YouCook2（抽样 10/10；后续仍需扩大抽样）
- Kinetics-400（抽样 9/10；allowlist 已排除 1 条静音）
- Kinetics-600（抽样 5 条成功；均有可解码音频流；仍是抽样结论）
- Kinetics-700（抽样 5 条成功；均有可解码音频流；仍是抽样结论）
- ego4d / ego4d_vqa / egoqa / EgoQA（Ego4D 官方提供 audio availability 元数据）
- holoassist（官方说明含同步音频流）
- epickitchens（EPIC-Sounds 明确来自 audio stream）
- egoexo4d / egoexolearn（官方说明每段含 7-channel audio）

## 3. 明确没有

- TGIF（GIF 为主模态，原始音频不是稳定模态）
- CLEVRER（合成视频，音频不是原始模态）

## 4. 无法读取，后续再处理

- WebVid2M（官方已停止公开 URL/caption 分发，JoyAI 数字 video_name 无法恢复合法母媒体索引）
- WebVid10M（官方已停止公开 URL/caption 分发，JoyAI 数字 video_name 无法恢复合法母媒体索引）
- DiDeMo（母媒体在 Flickr；当前 Flickr API 返回 502，无法取得对应视频）
- NExT-QA（已找到 VidOR 映射文件，但原始 Google Drive 视频包尚未下载/核验）
- LSMDC（电影片段媒体访问受控，当前没有可核验的本地母视频）
- CharadesEgo（当前抽样媒体未发现可用音轨；需先确认 prepared-clip 与 JoyAI 命名映射后再复核）
- URL 来源（YouTube/Bilibili，19,680 个具体 URL；尚未建立逐 URL 媒体索引，因此不能整体判定）
- EgoIT（片段名可解析，但母视频媒体当前不可访问）
- EgoIT-99K（片段名可解析，但母视频媒体当前不可访问）
- EgoLife（当前没有可直接映射的本地媒体样本）

## 已生成的逐文件证据

- `datasets/audit_output/multimodal_audio_allowlist.jsonl`：ActivityNet、YouCook2、Kinetics-400/600/700 共 38 条抽样，37 条确认有音频流，1 条确认静音。
- `datasets/audit_output/activitynet10_media/ffprobe_report.csv`
- `datasets/audit_output/youcook2_10_media/ffprobe_report.csv`
- `datasets/audit_output/kinetics10_media/ffprobe_report.csv`
- `datasets/audit_output/kinetics60010_media/ffprobe_report.csv`
- `datasets/audit_output/kinetics70010_media/ffprobe_report.csv`

## 抽样最低标准

- 可访问 source：每个 source 至少抽样 5 条，优先 10 条；记录成功、失败、音频流和可听内容状态。
- 只有 1～2 条成功样本时，只能写“初步发现有音频”，不能写成 source 级确认。
- 媒体访问失败时移动到“无法读取，后续再处理”，并在同一行写明失败原因、访问入口和下一步恢复条件。

## 更新规则

1. 先从 JoyAI 的具体 `source + video_name` 建立母数据映射。
2. 只下载小样本并检查单文件大小，不执行超过 1GB 的下载。
3. 对实际文件运行 `ffprobe`；必要时再做人耳静音抽查。
4. 把 source 从“尚未读取”移动到“明确有音频”“明确没有”或“无法读取”，并保留理由、URL、许可证和 SHA-256。
5. 混合来源（VideoChat2、LLaVA-Video-178K、Open-o3-Video、Molmo2 等）必须按母数据集和具体文件判定，不能整体移动栏目。
