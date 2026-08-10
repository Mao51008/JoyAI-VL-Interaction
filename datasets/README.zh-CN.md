# JoyAI-VL-Interaction 使用说明

> 原文档: [README.md](./README.md)

数据集：https://huggingface.co/datasets/jdopensource/JoyAI-VL-Interaction

### 1. 下载标注

```bash
hf download hf://datasets/jdopensource/JoyAI-VL-Interaction \
  --local-dir ./JoyAI-VL-Interaction
```

### 2. 数据格式

使用 `source` 字段定位原始视频，然后在运行转换前，将 `video_path` 填写为视频在本地的绝对路径。

```json
[
  {
    "video_name": "name.mp4",
    "video_path": "/path/to/videos/name.mp4",
    "task_type": "task_category",
    "source": "source_dataset",
    "question": [
      {"content": "user question or instruction", "time": "4"}
    ],
    "response": [
      {"content": "expected answer or event response", "time": "7"}
    ]
  }
]
```

- `question`：用户提示列表。`content` 是提示文本，`time` 是发出提示时的时间戳，单位为秒。
- `response`：真实响应列表。`content` 是预期答案，`time` 是答案或事件应当出现的时间戳，单位为秒。

### 3. 运行转换

```bash
python ./convert_data.py \
  --input /path/to/JoyAI-VL-Interaction/annotations_with_video_path \
  --output /path/to/JoyAI-VL-Interaction/processed_data \
  --frame_dir /path/to/JoyAI-VL-Interaction/images \
  --log_dir /path/to/JoyAI-VL-Interaction/logs \
  --max_duration MAX_DURATION_SECONDS \
  --workers NUM_WORKERS
```

### 4. 转换后的输出格式

脚本会为每一秒创建一组 user/assistant 消息。如果问题从第 4 秒开始，预期响应出现在第 7 秒，转换后的消息如下：

```json
[
  {
    "messages": [
      {"role": "user", "content": "<0.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</silence>"},
      {"role": "user", "content": "<1.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</silence>"},
      {"role": "user", "content": "<2.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</silence>"},
      {"role": "user", "content": "<3.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</silence>"},
      {"role": "user", "content": "user question or instruction\n<4.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</silence>"},
      {"role": "user", "content": "<5.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</silence>"},
      {"role": "user", "content": "<6.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</silence>"},
      {"role": "user", "content": "<7.0 seconds>\n<image>"},
      {"role": "assistant", "content": "</response> expected answer or event response"}
    ],
    "images": [
      "/path/to/images/task_category/source_dataset/name/frame_000000.jpg",
      "/path/to/images/task_category/source_dataset/name/frame_000001.jpg",
      "/path/to/images/task_category/source_dataset/name/frame_000002.jpg",
      "/path/to/images/task_category/source_dataset/name/frame_000003.jpg"
    ],
    "video_name": "name.mp4",
    "video_path": "/path/to/videos/name.mp4",
    "task_type": "task_category",
    "source": "source_dataset"
  }
]
```

### 5. Omni Stage3：下载计划与多来源转换

`download_omni_stage3_media.py` 只接受已完成母媒体映射的 JSONL 清单。默认仅输出
下载计划；只有显式加 `--execute` 才会访问网络。每条清单必须包含 `source`、
`video_name`、`download_url`、`relative_path`、`size_bytes`，并建议提供 `sha256`。
下载器按 source 配额和总容量上限拒绝超额条目。

`mapped_media.jsonl` 的一行示例：

```json
{"source":"holoassist","video_name":"R007.mp4","download_url":"https://approved.example/R007.mp4","relative_path":"holoassist/R007.mp4","size_bytes":123456789,"sha256":"可选的64位sha256"}
```

`source_quotas.json` 是字节单位的 source 到容量上限映射；只为已经通过来源映射和许可核验的 source 配置额度：

```json
{"holoassist":155692564480,"ActivityNet":59055800320,"YouCook2":32212254720}
```

```bash
python datasets/download_omni_stage3_media.py mapped_media.jsonl \
  --quota-config source_quotas.json \
  --output-root /data/maoyy/datasets/omni_stage3/raw \
  --receipt /data/maoyy/datasets/omni_stage3/manifests/download_plan.jsonl
```

常规媒体验收完成后，`convert_omni_stage3_sources.py` 会按 source 分别调用已有的
Omni 转换器。混合 audit JSONL 的每一行应含 `source`，从而避免同名视频跨来源错误连接。
未显式写入的时间步依照 Omni 协议自动监督为 `silence`。

`source_provenance.json` 采用 `source -> provenance JSON 文件路径` 的映射；每个 provenance 文件仍须完整满足 `omni-training-v1` 的许可证字段约束。

```bash
python datasets/convert_omni_stage3_sources.py annotations.json media_audit.jsonl \
  --provenance-registry source_provenance.json \
  --output-dir /data/maoyy/datasets/omni_stage3/converted
```
