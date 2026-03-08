# 06-1 - SFT 基础训练

SFT（Supervised Fine-Tuning）是 RL 训练前的重要准备步骤。

## 概述

verl 提供独立的 SFT 训练器，支持：

- 标准监督学习
- 序列并行（Ulysses SP）
- LoRA/PEFT
- 多模态模型

## 训练入口

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.sft_trainer \
    data.train_files=./data/train.parquet \
    data.val_files=./data/val.parquet \
    model.path=Qwen/Qwen2.5-7B-Instruct \
    optim.lr=1e-5 \
    trainer.total_training_steps=1000
```

## 完整训练脚本

### 标准 SFT

```bash
#!/bin/bash
# run_sft.sh

nproc_per_node=8
save_path=./checkpoints/sft_model

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size=8 \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.05 \
    optim.grad_clip=1.0 \
    engine=fsdp \
    model.path=Qwen/Qwen2.5-7B-Instruct \
    model.enable_gradient_checkpointing=true \
    model.use_remove_padding=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=qwen2_5_7b_sft \
    trainer.logger='["console","wandb"]' \
    trainer.total_training_steps=2000 \
    trainer.save_freq=500 \
    trainer.test_freq=100
```

### SFT + 序列并行

```bash
#!/bin/bash
# run_sft_sp2.sh - SFT with Sequence Parallel

nproc_per_node=8
save_path=./checkpoints/sft_model_sp2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size=4 \
    optim.lr=1e-4 \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=2 \
    model.path=Qwen/Qwen2.5-7B-Instruct \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft-sp \
    trainer.experiment_name=qwen2_5_7b_sp2 \
    trainer.logger=console \
    trainer.total_training_steps=2000
```

### SFT + LoRA

```bash
#!/bin/bash
# run_sft_lora.sh - SFT with LoRA

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size=16 \
    optim.lr=5e-5 \
    engine=fsdp \
    model.path=Qwen/Qwen2.5-7B-Instruct \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
    trainer.default_local_dir=./checkpoints/sft_lora \
    trainer.project_name=gsm8k-sft-lora \
    trainer.total_training_steps=1000
```

### 多模态 SFT

```bash
#!/bin/bash
# run_sft_vlm.sh - SFT for Vision-Language Model

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.sft_trainer \
    data.train_files=$HOME/data/vlm_train.parquet \
    data.val_files=$HOME/data/vlm_val.parquet \
    data.messages_key=messages \
    data.image_key=images \
    data.micro_batch_size=4 \
    optim.lr=1e-5 \
    engine=fsdp \
    model.path=Qwen/Qwen2-VL-7B-Instruct \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=./checkpoints/vlm_sft \
    trainer.project_name=vlm-sft \
    trainer.total_training_steps=1000
```

## 数据格式

### 标准对话格式

```python
# train.parquet 结构
import pandas as pd
import pyarrow.parquet as pq

data = {
    'messages': [
        [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'What is 2+2?'},
            {'role': 'assistant', 'content': '2+2 equals 4.'}
        ],
        # ... more examples
    ]
}

df = pd.DataFrame(data)
df.to_parquet('train.parquet')
```

### 多模态格式

```python
data = {
    'messages': [
        [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'Describe this image.'}
            ]},
            {'role': 'assistant', 'content': 'This is a...'}
        ]
    ],
    'images': [
        ['base64_encoded_image_or_path']
    ]
}
```

## 关键配置

```bash
# 数据
data.train_files=./data/train.parquet
data.val_files=./data/val.parquet
data.messages_key=messages      # 消息字段名
data.micro_batch_size=8         # 批次大小

# 优化器
optim.lr=1e-5                   # 学习率
optim.lr_warmup_steps_ratio=0.05  # 预热比例
optim.grad_clip=1.0             # 梯度裁剪
optim.lr_scheduler_type=cosine  # cosine | constant

# 引擎
engine=fsdp                     # fsdp | fsdp2
engine.ulysses_sequence_parallel_size=2  # 序列并行大小

# 模型
model.path=Qwen/Qwen2.5-7B-Instruct
model.enable_gradient_checkpointing=true
model.use_remove_padding=true

# 训练器
trainer.total_training_steps=2000
trainer.save_freq=500
trainer.test_freq=100
trainer.logger='["console","wandb"]'
```

## 下一步

- [06-2-model-tuning.md](06-2-model-tuning.md) - 不同规模模型调参
