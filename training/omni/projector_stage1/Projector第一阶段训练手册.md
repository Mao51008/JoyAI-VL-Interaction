# Projector 第一阶段训练手册

本手册用于主力服务器 `maoyy@10.11.12.30` 的 LibriSpeech `train-clean-100`
音频-文本 projector-only 训练。训练仅更新 `audio_projector`；Qwen3-ASR
缓存特征和 JoyAI-VL 均冻结。

## 训练前确认

连接服务器后，先确认分支、工作树和 GPU：

```bash
cd /data/maoyy/JoyAI-VL-Interaction
/data/maoyy/miniforge3/bin/git branch --show-current
/data/maoyy/miniforge3/bin/git status --short
nvidia-smi
```

必须显示 `feature/continuous-omni`。只在四张 GPU 空闲、且确认允许训练时启动。

## 正式续训命令

这是当前正式任务的完整续训命令。它优先读取 `$OUT/last.pt`；若尚未生成，
会兼容读取此前中断训练留下的 `$OUT/latest.pt`。

```bash
cd /data/maoyy/JoyAI-VL-Interaction

PY=/data/maoyy/joyai-runtime/omni-training/.venv/bin/python
ROOT=/data/maoyy/datasets/projector_stage1/librispeech_train100_20260804
OUT=$ROOT/checkpoints-ddp-4gpu-3epochs-20260804

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PY" -m torch.distributed.run --nproc_per_node=4 \
  -m training.omni.projector_stage1.train \
  --ddp \
  --manifest "$ROOT/train_samples_max17_5s_train-clean-100.jsonl" \
  --feature-dir "$ROOT/features-train-100-sharded" \
  --validation-manifest "$ROOT/validation_samples_train-clean-100.jsonl" \
  --validation-feature-dir "$ROOT/features-validation-20260804" \
  --joyai-model /data/maoyy/models/jdopensource/JoyAI-VL-Interaction \
  --config training/omni/projector_stage1/projector_stage1.json \
  --output-dir "$OUT" \
  --steps 10731 \
  --steps-per-epoch 3577 \
  --batch-size 2 \
  --max-attention-cost 90000 \
  --checkpoint-every-steps 1000 \
  --validate-every-steps 250 \
  --early-stopping-patience 4 \
  --warmup-steps 200 \
  --max-grad-norm 1.0 \
  --seed 3407 \
  --resume
```

不要添加 `--no-progress`。rank 0 会显示进度条、训练 loss、学习率；每轮结束会额外输出
`validation_loss`。

## 参数说明

- `--nproc_per_node=4` 和 `--ddp`：一张 4090 对应一个 rank，做 DDP 梯度同步。
- `--manifest`：清洗后的训练清单，27,838 条样本；只移除了 5 条超过 17.5 秒的长尾音频。
- `--feature-dir`：冻结 Qwen3-ASR 特征的分片缓存，不会在训练中重复运行 ASR encoder。
- `--validation-*`：696 条 speaker-disjoint 验证集及其缓存，用于选择 `best.pt`。
- `--steps 10731`：3 个 epoch 的总训练 step。
- `--steps-per-epoch 3577`：清洗后、长度感知 DDP batch 下每个 rank 一轮的 step 数。
- `--batch-size 2`：每卡 batch 上限为 2；常规全局 batch 为 8。
- `--max-attention-cost 90000`：长样本 pair 自动拆成单条，以避免 OOM；大多数 batch 保持 2 条。
- `--checkpoint-every-steps 1000`：每 1000 step 写一次 `last.pt`，而不是每 step 写盘。
- `--validate-every-steps 250`：每 250 step 在隔离验证集计算 loss，并据此更新 `best.pt`。
- `--early-stopping-patience 4`：连续 4 次验证未改善时安全停止并保存 `last.pt`。
- `--warmup-steps 200`、`--max-grad-norm 1.0`：避免一开始的过大更新并限制异常梯度。
- `--seed 3407`：固定 batch 打乱和 projector 初始化；首次运行会保存公共 `step-0.pt`。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：减少长时间训练的 CUDA 显存碎片。
- `--resume`：从最近保存的训练状态继续；第一次从零开始的训练移除此参数，并使用新的输出目录。

## Checkpoint 规则

所有文件都在 `$OUT`：

- `last.pt`：每 1000 step、每轮结束、训练结束时覆盖写；用于恢复 optimizer、scheduler 和 projector。
- `best.pt`：仅当整轮验证 loss 优于历史最佳时覆盖写；用于最终评测和部署候选。
- `latest.pt`：旧训练器留下的兼容 checkpoint。新训练不再持续写它，但 `--resume` 会在不存在 `last.pt` 时读取它。
- `step-0.pt`：固定种子的公共 projector 初始状态，可通过 `--init-projector-checkpoint` 复用于 95 条、1 小时和全量受控对照。
- `metrics.jsonl`：每 step 的训练 loss、监督 token 数、学习率及每次验证 loss，可直接用于绘制曲线。

基础模型权重不会复制到 checkpoint 中。

## 监控与停止

另开一个 SSH 终端监控：

```bash
nvidia-smi -l 2
```

训练进度由启动训练的终端直接显示。若在后台运行，可查看对应日志文件末尾：

```bash
tail -f "$OUT/train.log"
```

需要停止时，先精确定位本次训练进程，再向 torchrun launcher 发送 `SIGTERM`：

```bash
ps -eo pid,ppid,args | grep '[t]orch.distributed.run'
kill -TERM <torchrun-pid>
```

不要使用宽泛的 `pkill python` 或 `pkill torch`。停止后重复“正式续训命令”即可恢复。

## 每轮后的检查

每轮结束后确认输出目录中同时存在：

```bash
ls -lh "$OUT/last.pt" "$OUT/best.pt"
```

`last.pt` 代表可恢复的最新状态；`best.pt` 应以验证 loss 最低的轮次为准。三轮完成后，使用
`best.pt` 运行独立评测脚本，记录验证 loss/perplexity，并在后续阶段增加 WER/CER 解码评测。
