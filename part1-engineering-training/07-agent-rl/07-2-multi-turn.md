# 07-2 - 多轮对话 RL 训练

多轮对话 RL 让模型学会在复杂交互场景中优化长期目标。

## 概述

verl 通过 SGLang 支持多轮 RL 训练：

- 多轮对话历史管理
- 支持 Tool Calling
- 支持环境交互
- Partial Rollout（部分采样）

## 训练脚本

### GSM8K 多轮训练

```bash
#!/bin/bash
# run_multiturn_gsm8k.sh

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.enable_overlong_filter=true \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='multiturn_gsm8k' \
    trainer.experiment_name='qwen2_5_3b'
```

### Geo3K 多轮训练

```bash
#!/bin/bash
# run_multiturn_geo3k.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/geo3k/train.parquet \
    data.val_files=$HOME/data/geo3k/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10
```

## 关键配置

```bash
# 启用多轮
actor_rollout_ref.rollout.multi_turn.enable=true

# 轮数限制
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
actor_rollout_ref.rollout.multi_turn.max_user_turns=10

# 长度过滤
actor_rollout_ref.rollout.multi_turn.enable_overlong_filter=true

# SGLang Server 模式（推荐用于多轮）
actor_rollout_ref.rollout.multi_turn.server_mode=true
```

## 数据格式

多轮对话数据需要包含对话历史：

```python
# multiturn/train.parquet
data = {
    'data_source': [
        [
            {'role': 'user', 'content': 'What is 2+2?'},
            {'role': 'assistant', 'content': 'Let me calculate. 2+2=4.'},
            {'role': 'user', 'content': 'And 3+3?'},
        ]
    ],
    'tools': [
        [{'type': 'python_executor'}]
    ]
}
```

## 下一步

- [07-3-agent-loop.md](07-3-agent-loop.md) - Agent Loop 训练
