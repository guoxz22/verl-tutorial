# 05-2 - GRPO 训练

GRPO（Group Relative Policy Optimization）是 DeepSeek-R1 使用的算法，适合可验证奖励场景（如数学、代码）。

## 算法特点

- **不需要 Critic**：使用 Group 内的平均 Reward 作为 baseline
- **KL Loss**：KL 惩罚在 Loss 中，而非 Reward 中
- **多样本采样**：每个 Prompt 生成 n 个 Response（通常 n=5-8）
- **高效**：比 PPO 节省 Critic 计算开销

## 算法原理简述

```
对于每个 Prompt，生成 n 个 Response：
  r_1, r_2, ..., r_n

计算每个 Response 的 Reward：
  R_1, R_2, ..., R_n

计算 Advantage（Group 内归一化）：
  A_i = (R_i - mean(R)) / std(R)

更新 Policy，最小化：
  Loss = -log_prob * A + kl_coef * KL(π || π_ref)
```

## 完整训练脚本

### GRPO 基础配置

```bash
#!/bin/bash
# run_grpo.sh - GRPO on GSM8K

set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
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
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example' \
    trainer.experiment_name='qwen2_7b_gsm8k'
```

### GRPO + LoRA（节省内存）

```bash
#!/bin/bash
# run_grpo_lora.sh - GRPO with LoRA

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj"]' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=20 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_lora' \
    trainer.experiment_name='qwen2_5_3b_lora'
```

### GRPO + Megatron（大模型）

```bash
#!/bin/bash
# run_grpo_megatron.sh - GRPO with Megatron-LM for large models

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.train_batch_size=2048 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    \
    actor_rollout_ref.model.path=/path/to/deepseek-671b \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.n=8 \
    \
    critic.strategy=megatron \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=16 \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]'
```

## 关键配置说明

### GRPO 必需配置

```bash
# 算法类型
algorithm.adv_estimator=grpo

# GRPO 特有配置
algorithm.norm_adv_by_std_in_grpo=True  # 按 std 归一化 Advantage

# 不使用 Critic
trainer.critic_warmup=0

# KL Loss（GRPO 使用 KL in Loss，而非 Reward）
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl

# 多样本采样（每个 Prompt 生成 n 个 Response）
actor_rollout_ref.rollout.n=5
```

### 采样数 n 的选择

| n 值 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| 2-4 | 计算快 | baseline 方差大 | 快速实验 |
| 5-8 | 平衡 | - | 推荐默认 |
| 16+ | baseline 稳定 | 计算慢 | 大规模训练 |

### KL Loss 配置

```bash
# KL 损失系数
actor_rollout_ref.actor.kl_loss_coef=0.001  # 通常 0.001-0.01

# KL 损失类型
actor_rollout_ref.actor.kl_loss_type=low_var_kl  # 推荐使用低方差 KL
```

## 性能优化

### 内存优化

```bash
# 启用 Gradient Checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing=True

# Reference 模型 Offload
actor_rollout_ref.ref.fsdp_config.param_offload=True

# 使用 LoRA
actor_rollout_ref.model.lora_rank=64
```

### 推理优化

```bash
# vLLM V1（推荐）
export VLLM_USE_V1=1

# 调整 GPU 内存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# 调整 Tensor Parallel
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

### 训练优化

```bash
# 使用 FSDP2
actor_rollout_ref.actor.strategy=fsdp2

# 启用 torch.compile
actor_rollout_ref.actor.use_torch_compile=True

# 调整 micro batch size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64
```

## 超参数调优建议

### 学习率

| 模型大小 | 推荐学习率 |
|----------|-----------|
| 0.5B - 3B | 1e-5 |
| 7B - 14B | 1e-6 |
| 32B+ | 5e-7 |

### KL 系数

| 任务类型 | 推荐 kl_loss_coef |
|----------|------------------|
| 数学推理 | 0.001 |
| 代码生成 | 0.001 - 0.005 |
| 通用对话 | 0.005 - 0.01 |

### 采样温度

```bash
# 训练时
actor_rollout_ref.rollout.temperature=1.0  # 默认

# 验证时（贪婪解码）
actor_rollout_ref.rollout.val_kwargs.temperature=0
actor_rollout_ref.rollout.val_kwargs.do_sample=False
```

## 监控指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `train/reward` | 平均奖励 | 应持续上升 |
| `train/actor_loss` | Policy Loss | 应趋于稳定 |
| `train/kl` | KL 散度 | 不应持续增大 |
| `train/entropy` | 策略熵 | 不应过快下降 |
| `val/accuracy` | 验证准确率 | 数学/代码任务 |

## 下一步

- [05-3-reinforce-pp.md](05-3-reinforce-pp.md) - REINFORCE++ 训练
- [09-1-data-preprocess.md](../09-data-and-reward/09-1-data-preprocess.md) - 数据预处理
