# 05-4 - RLOO / ReMax 详解

本章详细介绍 RLOO 和 ReMax 算法的使用。

## RLOO (Reinforce Leave-One-Out)

### 算法原理

RLOO 对每个 Prompt 生成 n 个 Response，计算 Advantage 时使用 Leave-One-Out baseline：

```
对于 Response i:
  baseline_i = mean(R_j for j != i)  # 其他 Response 的平均奖励
  Advantage_i = R_i - baseline_i
```

### 与 GRPO 的区别

| 特性 | GRPO | RLOO |
|------|------|------|
| Baseline | Group 均值 | Leave-One-Out 均值 |
| 归一化 | 按 std 归一化 | 无归一化 |
| KL 位置 | Loss | Reward |
| 方差 | 较低 | 中等 |

### 完整训练脚本

```bash
#!/bin/bash
# run_rloo_gsm8k.sh

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_rloo_gsm8k' \
    trainer.experiment_name='qwen2_7b'
```

### 关键参数

```bash
# 采样数 n：影响 baseline 质量
actor_rollout_ref.rollout.n=5  # 推荐 4-8

# KL 惩罚：RLOO 使用 KL in Reward
algorithm.use_kl_in_reward=True
algorithm.kl_ctrl.kl_coef=0.001  # 比 GRPO 通常更小
```

## ReMax

### 算法原理

ReMax（Reward Maximization）直接最大化期望奖励：

```
Loss = -log_prob * (R - baseline)

其中 baseline 可以是：
- 平均奖励
- 最小奖励
- 零
```

### 完整训练脚本

```bash
#!/bin/bash
# run_remax_gsm8k.sh

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=remax \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=20 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_remax_gsm8k' \
    trainer.experiment_name='qwen2_7b'
```

## 性能对比

### GSM8K 上的表现（示例）

| 算法 | 模型 | n | Acc@1 | 训练时间 |
|------|------|---|--------|----------|
| PPO | 7B | 1 | 75% | 1x |
| GRPO | 7B | 5 | 78% | 0.9x |
| RLOO | 7B | 5 | 77% | 0.85x |
| ReMax | 7B | 4 | 76% | 0.8x |

### 计算效率

| 算法 | 额外计算 | 内存占用 |
|------|----------|----------|
| PPO | Critic 前向 | 高 |
| GRPO | n 倍采样 | 中 |
| RLOO | n 倍采样 | 中 |
| ReMax | n 倍采样 | 低 |

## 选择建议

### 使用 GRPO 的场景

- 数学推理、代码生成等可验证奖励
- 需要稳定的 KL 控制
- 大规模训练

### 使用 RLOO 的场景

- 需要更灵活的 baseline
- 实验不同 KL 策略
- 与 GRPO 对比实验

### 使用 ReMax 的场景

- 奖励信号明确
- 追求简单实现
- 计算资源受限

## 下一步

- [05-5-other-algos.md](05-5-other-algos.md) - 其他算法（CISPO, DPPO, FAPO 等）
