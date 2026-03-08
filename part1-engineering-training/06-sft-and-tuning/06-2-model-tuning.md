# 06-2 - 模型规模与调参

不同规模的模型需要不同的配置策略。

## 模型规模分类

| 规模 | 参数量 | 典型模型 | GPU 需求 |
|------|--------|----------|----------|
| 小型 | 0.5B-3B | Qwen2.5-0.5B/1.5B/3B | 1-4 GPU |
| 中型 | 7B-14B | Qwen2-7B, Qwen2.5-7B, Gemma-7B | 4-8 GPU |
| 大型 | 32B-70B | Qwen2.5-32B/72B, Llama-70B | 8-32 GPU |
| 超大 | 100B+ | DeepSeek-671B, Qwen3-235B | 64+ GPU |

## 小型模型 (0.5B-3B)

### 配置特点

- 可以使用较大 batch size
- 不需要太多优化
- 适合快速实验

### 示例配置

```bash
# Qwen2.5-0.5B GRPO
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2048 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=30
```

## 中型模型 (7B-14B)

### 配置特点

- 需要梯度检查点
- 合理的 batch size
- 可能需要 offload

### 示例配置

```bash
# Qwen2.5-7B GRPO
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15
```

## 大型模型 (32B-70B)

### 配置特点

- 必须使用梯度检查点
- 需要 offload 优化
- 考虑 Megatron-LM

### 示例配置 (FSDP)

```bash
# Qwen2.5-32B GRPO with FSDP
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-32B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.total_epochs=10
```

### 示例配置 (Megatron-LM)

```bash
# Qwen2.5-72B GRPO with Megatron
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    \
    actor_rollout_ref.model.path=/path/to/qwen2_5_72b_megatron \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=4 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.total_epochs=5
```

## 超大模型 (100B+)

### 配置特点

- 必须使用 Megatron-LM
- 需要 Expert Parallelism (MoE)
- 多机训练

### DeepSeek-671B 示例

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    \
    actor_rollout_ref.model.path=/path/to/deepseek_v3_671b \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.expert_model_parallel_size=8 \
    actor_rollout_ref.rollout.n=4 \
    \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=64 \
    trainer.total_epochs=3
```

## 调参指南

### 学习率

| 模型规模 | 推荐 LR | 备注 |
|----------|---------|------|
| 0.5B-3B | 1e-5 - 2e-5 | 可以较大 |
| 7B-14B | 5e-7 - 1e-6 | 标准范围 |
| 32B-70B | 1e-7 - 5e-7 | 需要较小 |
| 100B+ | 1e-8 - 1e-7 | 非常小 |

### Batch Size

| 模型规模 | train_batch_size | ppo_mini_batch_size |
|----------|------------------|---------------------|
| 0.5B-3B | 1024-2048 | 256-512 |
| 7B-14B | 512-1024 | 128-256 |
| 32B-70B | 256-512 | 64-128 |
| 100B+ | 64-256 | 16-64 |

### 内存优化策略

| 模型规模 | 策略 |
|----------|------|
| 0.5B-3B | 基本不需要 |
| 7B-14B | Gradient Checkpointing + Ref Offload |
| 32B-70B | FSDP2 + CPU Offload + LoRA |
| 100B+ | Megatron + EP + PP + LoRA |

## 下一步

- [08-distributed-training](../08-distributed-training/) - 分布式训练配置
