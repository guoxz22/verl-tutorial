# 07-3 - Agent Loop 训练

Agent Loop 是完整的 Agent 训练框架，支持复杂的多步任务。

## 概述

verl 的 Agent Loop 支持：

- 环境交互（Sandbox、代码执行器等）
- 长轨迹训练
- 多种工具集成
- Partial Rollout 优化

## 训练脚本

### 基础 Agent Loop

```bash
#!/bin/bash
# run_agent_loop.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/agent_train.parquet \
    data.val_files=$HOME/data/agent_val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    \
    reward.num_workers=4 \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='agent_loop' \
    trainer.experiment_name='qwen2_5_7b'
```

### GSM8K Tool Agent Loop

```bash
#!/bin/bash
# run_gsm8k_tool_agent.sh - from examples/sglang_multiturn

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
    data.tool_provider=gsm8k_tool_agent \
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
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=15
```

## 工具配置

### Python 执行器

```bash
actor_rollout_ref.rollout.multi_turn.tool.dict.python_executor=true
```

### 代码沙箱

```bash
actor_rollout_ref.rollout.multi_turn.tool.sandbox_fusion.enable=true
actor_rollout_ref.rollout.multi_turn.tool.sandbox_fusion.server_url=http://localhost:8080
```

### 搜索工具

```bash
actor_rollout_ref.rollout.multi_turn.tool.search.enable=true
actor_rollout_ref.rollout.multi_turn.tool.search.engine=google
```

## 环境配置

### 启动 Sandbox Server

```bash
# 启动代码沙箱
docker run -d -p 8080:8080 --name sandbox \
    --privileged \
    -v /var/run/docker.sock:/var/run/docker.sock \
    sandbox-fusion:latest
```

### 配置 SGLang Server

```bash
# 使用 Server 模式（推荐）
actor_rollout_ref.rollout.multi_turn.server_mode=true
actor_rollout_ref.rollout.multi_turn.server.port=30000
```

## 示例项目

| 项目 | 描述 | 链接 |
|------|------|------|
| Search-R1 | 搜索+推理 | github.com/PeterGriffinJin/Search-R1 |
| RAGEN | 通用 Agent | github.com/ZihanWang314/ragen |
| OpenManus-RL | Agent 训练 | github.com/OpenManus/OpenManus-RL |
| verl-agent | 长程 Agent | github.com/langfengQ/verl-agent |

## 下一步

- [08-distributed-training](../08-distributed-training/) - 分布式训练
