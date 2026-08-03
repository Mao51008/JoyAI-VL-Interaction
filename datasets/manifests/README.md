# 首轮数据来源 manifest

`sources.jsonl` 是里程碑 B 首轮数据集的来源登记表。当前 checkpoint 仅用于研究和内部
实验，不要求商用；这不免除原始数据的署名、相同方式共享、受控访问或禁止再分发条款。

## 使用规则

- 先审计 pilot scope，再决定是否扩容；不得按此文件直接全量下载。
- 每个转换样本还必须记录 `source_dataset`、`source_version`、原始文件路径或 ID、
  时间窗口、角色映射、转换器 commit 和人工/教师标注信息。
- `research_only` 数据不得与可发布数据无记录混合；训练运行必须保存使用过的 manifest
  指纹。
- Microsoft AEC Challenge 必须按原始组成来源逐文件登记，仓库代码的 MIT 许可证不代表
  所有音频数据均为 MIT。
- 环境声音和危险声音不在当前数据集范围；本清单不收录环境声音分类数据。
