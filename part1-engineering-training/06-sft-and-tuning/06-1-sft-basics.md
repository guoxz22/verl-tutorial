# 06-1 - SFT 基础训练

<!-- NAV_START -->
> 阅读： [← 05-5 - 其他算法](../05-algorithms/05-5-other-algos.md) · [目录](../../README.md#catalog) · [06-2 - 模型规模与调参 →](06-2-model-tuning.md)
<!-- NAV_END -->

SFT 在 v0.8.0 中使用统一入口：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=<gpu数> \
  -m verl.trainer.sft_trainer \
  engine=fsdp \
  model.path=<模型路径>
```

对应默认配置是 `verl/trainer/config/sft_trainer_engine.yaml`。最容易写错的键是 batch：v0.8.0 使用 `data.micro_batch_size_per_gpu`，不是旧教程里的 `data.micro_batch_size`。

## 最小可运行脚本

```bash
#!/usr/bin/env bash
set -xeuo pipefail

SAVE_PATH=${SAVE_PATH:-$PWD/checkpoints/sft-qwen2_5-0_5b}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

python examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  -m verl.trainer.sft_trainer \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=4 \
  data.max_length=2048 \
  optim.lr=1e-4 \
  engine=fsdp \
  model.path=${MODEL_PATH} \
  model.use_remove_padding=true \
  trainer.default_local_dir=${SAVE_PATH} \
  trainer.project_name=gsm8k-sft \
  trainer.experiment_name=qwen2_5_0_5b \
  trainer.logger='["console"]' \
  trainer.total_epochs=1
```

## 常用开关

| 目的 | 配置 |
| --- | --- |
| 调小显存 | `data.micro_batch_size_per_gpu=1` |
| 序列并行 | `engine.ulysses_sequence_parallel_size=2` |
| Liger kernel | `model.use_liger=True` |
| LoRA/PEFT | `model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear` |
| FSDP2 | `engine.distributed_strategy=fsdp2` 或参考官方 FSDP2 示例 |
| 多轮 SFT | `data.messages_key=messages`，数据里每条样本是 messages 列表 |
| 大模型 Megatron | `engine=megatron` 并配置 TP/PP/CP |

## 数据格式

SFT 数据通常是 parquet，每行包含 `messages`：

```python
{
  "messages": [
    {"role": "user", "content": "1+1 等于几？"},
    {"role": "assistant", "content": "2"}
  ]
}
```

若列名不是 `messages`，就改：

```bash
data.messages_key=my_messages
```

## SFT 与 RL 的衔接

SFT 产物可以作为 RL 的 actor 初始模型：

```bash
actor_rollout_ref.model.path=/path/to/sft/checkpoint/huggingface
critic.model.path=/path/to/sft/checkpoint/huggingface
```

如果 checkpoint 是 FSDP 分片，先看 [10-2-checkpoint.md](../10-operations/10-2-checkpoint.md) 的 merge 说明。

## 排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Key 'micro_batch_size' is not in struct` | 使用了旧键 | 改为 `data.micro_batch_size_per_gpu` |
| loss mask 全 0 | messages 格式或 chat template 不匹配 | 检查 tokenizer chat template 和数据列 |
| OOM | 单卡 micro batch 或最大长度太大 | 降低 `data.micro_batch_size_per_gpu` / `data.max_length` |
| 多轮样本被截断 | 上下文超过 `data.max_length` | 增大 max_length 或清洗超长样本 |

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 05-5 - 其他算法](../05-algorithms/05-5-other-algos.md) · [目录](../../README.md#catalog) · [06-2 - 模型规模与调参 →](06-2-model-tuning.md)
<!-- NAV_BOTTOM_END -->
