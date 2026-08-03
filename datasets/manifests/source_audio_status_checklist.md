# JoyAI source 音频判别清单

本清单按本地 6 个 chat shard 扫描得到的 **56 个命名 source** 建立；另有 19,680 个具体 YouTube/Bilibili URL 值（共 69,714 条 URL 来源记录），URL 必须按实际媒体逐条核验，不能只按域名判定。每次完成新的媒体映射或 `ffprobe` 抽样后，更新本文件和对应逐文件 manifest。

> “明确有音频”表示已有官方音频说明或本地抽样 `ffprobe` 证据；不代表该 source 的所有 JoyAI 样本都已下载。视频问答文本也不能当成音频转写。

| 尚未读取、判别 | 明确有音频 | 明确没有 | 无法读取，后续再处理 |
|---|---|---|---|
| EgoIT<br>EgoIT-99K<br>EgoLife<br>egoblind<br>Perception Test<br>EgoProceL<br>shot2story<br>Open-o3-Video<br>Molmo2-AskModelAnything<br>Molmo2-VideoCapQA<br>VideoInstruct-100K<br>VideoChat2<br>LLaVA-Video-178K<br>VideoInternSeg<br>VideoGPT-plus<br>TimeLens-100K<br>Vript<br>TextVR<br>MovieChat<br>EgoTaskQA<br>EgoSchema<br>omnistar<br>TransRAC<br>STAR / Charades<br>GUI-World<br>gui_world<br>NTU RGB+D<br>assembly101<br>Something-Something V2<br>ucfcrime<br>UVO<br>OOPS<br>FAVD<br>Kinetics-600<br>Kinetics-700<br>Live-WhisperX | ActivityNet（抽样 8/8）<br>youcook2 / YouCook2（抽样 10/10）<br>Kinetics-400（抽样 9/10；逐文件 allowlist 已排除 1 条静音）<br>ego4d / ego4d_vqa / egoqa / EgoQA（Ego4D 官方提供 audio availability 元数据）<br>holoassist（官方说明含同步音频流）<br>epickitchens（EPIC-Sounds 明确来自 audio stream）<br>egoexo4d / egoexolearn（官方说明每段含 7-channel audio） | TGIF（GIF 主模态，原始音频不是稳定模态）<br>CLEVRER（合成视频，音频不是原始模态） | WebVid2M<br>WebVid10M<br>DiDeMo<br>NExT-QA<br>LSMDC<br>CharadesEgo（已有抽样未发现可用音轨）<br>URL 来源（YouTube/Bilibili，19,680 个具体 URL，需逐条映射） |

## 已生成的逐文件证据

- `datasets/audit_output/multimodal_audio_allowlist.jsonl`：ActivityNet、YouCook2、Kinetics-400 共 28 条抽样，27 条确认有音频流，1 条确认静音。
- `datasets/manifests/multimodal_audio_allowlist.md`：上述 allowlist 的汇总和语义限制。
- `datasets/audit_output/activitynet10_media/ffprobe_report.csv`
- `datasets/audit_output/youcook2_10_media/ffprobe_report.csv`
- `datasets/audit_output/kinetics10_media/ffprobe_report.csv`

## 更新规则

1. 先从 JoyAI 的具体 `source + video_name` 建立母数据映射。
2. 只下载小样本并检查单文件大小，不执行超过 1GB 的下载。
3. 对实际文件运行 `ffprobe`；必要时再做人耳静音抽查。
4. 把 source 从“尚未读取”移动到“明确有音频”“明确没有”或“无法读取”，并保留理由、URL、许可证和 SHA-256。
5. 混合来源（VideoChat2、LLaVA-Video-178K、Open-o3-Video、Molmo2 等）必须按母数据集和具体文件判定，不能整体移动栏目。
