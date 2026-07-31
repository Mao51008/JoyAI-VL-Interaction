# CharadesEgo 切片调查结论

## 结论

JoyAI 标注中的 `XXXXEGO_action_N` 应当作为已经预处理好的独立片段名称使用，不应在
没有生成脚本或媒体对照的情况下，把 `N` 当作 CharadesEgo 官方 CSV 中 action 列表的
序号并自行裁剪父视频。

前一版审计把 JoyAI 的时间戳与官方 action 时长进行了比较。这个比较口径不成立，已
撤销“5,999 条时长不一致”的质量结论。

## 证据

1. JoyAI 官方数据说明只要求根据 `source` 找到与 `video_name` 对应的视频并填写
   `video_path`，没有发布 CharadesEgo action 裁剪规则。
2. JoyAI 技术报告说明，`question.time` 和 `response.time` 是每秒交互监督：表示何时
   提问、保持沉默或响应。闲聊样本会随机选择时间点，并使用附近三帧生成和核验对话。
   因而这些时间戳不是动作持续时间。
3. 社区复现数据集 `momo321654/Interaction-videos` 明确说明，归档中的视频名与训练
   数据的 `video_name` 对应，并提供两个 CharadesEgo 预切片分片：

   - `data/CharadesEgo__000.tar`：10,752,307,200 字节；
   - `data/CharadesEgo__001.tar`：5,738,188,800 字节；
   - 合计 16,490,496,000 字节，约 16.49 GB / 15.36 GiB。

4. 官方 CharadesEgo CSV 可以确认父视频 ID 存在，但不能单独证明 `_action_N` 的生成
   方式。按 action 顺序连接得到的边界只能算假设性诊断。

## “777 个边界异常”的正确解释

在假设 `_action_N` 对应官方 CSV 第 N 个 action 时，有 777 个 JoyAI 切片名称连接到
结束时间早于开始时间的官方 action。这只说明：

- 官方 CSV 的部分 action 边界本身异常，或者该顺序映射假设不成立；
- 不能据此断言对应的 JoyAI 预切片损坏；
- 也不能据此生成裁剪命令。

这些条目必须等取得预切片后检查实际时长和画面。

## `CharadesEgo__001.tar` 实际验收

2026-07-31 已在服务器对社区归档的第二分片完成无 GPU 抽样验收：

- 文件大小：5,738,188,800 字节；
- SHA-256：
  `7ef33d5393b094fdea6dbcc83ae3784a406294a4a706fd119526031293f36f7c`；
- tar 完整遍历通过；
- 归档包含 10,609 个无扩展名媒体成员，路径形如
  `videos_pool/CharadesEgo/N87MOEGO_action_11`；
- 其中 10,590 个名称可与当前 `chat_shard_06.json` 的 CharadesEgo 标注连接；
- 稳定抽样 20 条均可被 `ffprobe` 识别，20 条完整解码通过，20 条 PNG 抽帧通过；
- 16 条的最大标注时间位于媒体时长内；
- 4 条存在标注时间超出媒体时长，其中
  `O0628EGO_action_1` 的媒体仅 0.467 秒、标注到 24 秒，属于明显异常；
- 抽样 20 条均没有音轨，不能作为环境音或用户语音训练输入。

这次验收证明该分片的命名连接和大部分抽样视频编码可用，但还没有完成人工观看、
问题回答语义核对和许可审查，也不能把 20 条无音轨样本外推为整个分片全部无音轨。

归档成员没有 `.mp4` 扩展名。审计器已兼容
`videos_pool/<source>/<video_name>` 这种无扩展名结构，并使用安全生成的本地文件名抽取；
审计帧使用 PNG，且对不足一个采样周期的极短视频回退抽取首帧。

## 后续验收

当前不需要下载 CharadesEgo 官方 11 GB 父视频再自行裁剪。更直接的核验路径是：

1. 实看并人工核对本次抽样中的 10～20 条问题、回答、画面和时间点；
2. 单独复核 4 条时间戳越界样本，决定过滤还是修正；
3. 扩大音轨统计，确认“无音轨”是分片级特征还是抽样现象；
4. 检查社区归档的再分发许可是否满足我们的研究和训练用途；
5. 通过后再决定是否下载 `CharadesEgo__000.tar` 和扩大训练清单。

社区归档不是 JoyAI 官方发布，且 CharadesEgo 官方许可对再分发和用途有限制。因此，
媒体内容匹配与许可审查必须同时通过，才能进入正式训练清单。

## 参考

- JoyAI 数据说明：https://github.com/jd-opensource/JoyAI-VL-Interaction/tree/main/datasets
- JoyAI 技术报告：https://arxiv.org/abs/2606.14777
- 社区预切片归档：https://huggingface.co/datasets/momo321654/Interaction-videos
- CharadesEgo 官方页面：https://prior.allenai.org/projects/charades-ego
