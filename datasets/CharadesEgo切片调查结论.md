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

## 下载前的下一步

当前不需要下载 CharadesEgo 官方 11 GB 父视频再自行裁剪。更直接的核验路径是：

1. 从社区归档中取得少量与抽样 `video_name` 完全一致的预切片；
2. 对每条执行 `ffprobe`，记录时长、视频流、音频流、采样率和声道；
3. 实看 10～20 条，核对问题、回答、画面以及时间点；
4. 同时检查社区归档的再分发许可是否满足我们的研究和训练用途；
5. 通过后再决定是否下载完整的两个 CharadesEgo 分片。

社区归档不是 JoyAI 官方发布，且 CharadesEgo 官方许可对再分发和用途有限制。因此，
媒体内容匹配与许可审查必须同时通过，才能进入正式训练清单。

## 参考

- JoyAI 数据说明：https://github.com/jd-opensource/JoyAI-VL-Interaction/tree/main/datasets
- JoyAI 技术报告：https://arxiv.org/abs/2606.14777
- 社区预切片归档：https://huggingface.co/datasets/momo321654/Interaction-videos
- CharadesEgo 官方页面：https://prior.allenai.org/projects/charades-ego
