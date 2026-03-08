# 05-3 - REINFORCE++ / RLOO / ReMax 训练

本章介绍 verl 支持的其他 Policy Gradient 算法。

## REINFORCE++

REINFORCE++ 是优化的 REINFORCE 算法，加入了方差减少技术。

### 特点

- 不需要 Critic（类似 GRPO）
- 使用 baseline 减少方差
- 支持 KL 惩罚

### 训练脚本

```bash
#!/bin/bash
# run_reinforce_pp.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.gamma=1.0 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.01 \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=1 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=20 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_reinforce_pp' \
    trainer.experiment_name='qwen2_7b'
```

## RLOO (Reinforce Leave-One-Out)

RLOO 使用 Leave-One-Out baseline，比标准 REINFORCE 方差更低。

### 特点

- 每个 Prompt 生成 n 个 Response
- 使用其他 Response 的平均 Reward 作为 baseline
- 比 GRPO 更简单，但需要更多样本

### 训练脚本

```bash
#!/bin/bash
# run_rloo.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    algorithm.gamma=1.0 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.test_freq=5 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_rloo_example' \
    trainer.experiment_name='qwen2_7b_gsm8k'
```

## ReMax

ReMax 使用 Reward Maximization 策略，适合有明确奖励信号的场景。

### 特点

- 最大化期望奖励
- 支持多种 baseline
- 适合可验证奖励

### 训练脚本

```bash
#!/bin/bash
# run_remax.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=remax \
    algorithm.gamma=1.0 \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_remax_example'
```

## 算法对比

| 算法 | Critic | 采样数 (n) | KL 位置 | 适用场景 |
|------|--------|-----------|---------|----------|
| PPO | 需要 | 1 | Reward 或 Loss | Reward Model |
| GRPO | 不需要 | 5-8 | Loss | 可验证奖励 |
| REINFORCE++ | 不需要 | 1 | Reward | 简单任务 |
| RLOO | 不需要 | 5+ | Reward | 可验证奖励 |
| ReMax | 不需要 | 4+ | Loss | 可验证奖励 |

## 配置速查

### 算法选择

```bash
# PPO (GAE)
algorithm.adv_estimator=gae

# GRPO
algorithm.adv_estimator=grpo

# REINFORCE++
algorithm.adv_estimator=reinforce_plus_plus

# RLOO
algorithm.adv_estimator=rloo

# ReMax
algorithm.adv_estimator=remax
```

### Critic 配置

```bash
# 需要 Critic (PPO)
trainer.critic_warmup=10
critic.model.path=Qwen/Qwen2-7B-Instruct
critic.optim.lr=1e-5

# 不需要 Critic (GRPO, REINFORCE++, RLOO, ReMax)
trainer.critic_warmup=0
```

### 采样配置

```bash
# 单样本 (PPO, REINFORCE++)
actor_rollout_ref.rollout.n=1

# 多样本 (GRPO, RLOO, ReMax)
actor_rollout_ref.rollout.n=5
```

### KL 配置

```bash
# KL in Reward (PPO, REINFORCE++, RLOO)
algorithm.use_kl_in_reward=True
algorithm.kl_ctrl.kl_coef=0.01

# KL in Loss (GRPO, ReMax)
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
```

## 下一步

- [05-4-rloo-remax.md](05-4-rloo-remax.md) - RLOO/ReMax 详解
- [05-5-other-algos.md](05-5-other-algos.md) - 其他算法
