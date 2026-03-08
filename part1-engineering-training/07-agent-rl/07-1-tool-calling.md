# 07-1 - Tool Calling 训练

Tool Calling 是 AgentRL 的基础能力，让模型学会调用外部工具。

## 概述

verl 通过 SGLang 支持多轮 Tool Calling 训练：

- 支持多种工具类型（Python 执行器、搜索、代码沙箱等）
- 多轮对话交互
- 可自定义工具

## 训练脚本

### 基础 Tool Calling

```bash
#!/bin/bash
# run_tool_calling.sh

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/gsm8k_tool/train.parquet \
    data.val_files=$HOME/data/gsm8k_tool/test.parquet \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.train_batch_size=256 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='tool_calling' \
    trainer.experiment_name='qwen2_5_3b_gsm8k'
```

### GSM8K + Code Tool

```bash
#!/bin/bash
# run_gsm8k_tool.sh - GSM8K with Python Code Tool

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/gsm8k_tool/train.parquet \
    data.val_files=$HOME/data/gsm8k_tool/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.tool.dict.python_executor=true \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15
```

## 关键配置

```bash
# 启用多轮
actor_rollout_ref.rollout.multi_turn.enable=true

# 最大助手轮数
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5

# 工具配置
actor_rollout_ref.rollout.multi_turn.tool.dict.python_executor=true
```

## 数据格式

Tool Calling 数据需要包含工具调用和返回：

```python
# gsm8k_tool/train.parquet
data = {
    'data_source': [
        'Janet has 8 apples. She buys 2 more. How many does she have?',
    ],
    'tools': [
        [{'type': 'python_executor'}],
    ]
}
```

## 下一步

- [07-2-multi-turn.md](07-2-multi-turn.md) - 多轮对话 RL
- [07-3-agent-loop.md](07-3-agent-loop.md) - Agent Loop 训练
