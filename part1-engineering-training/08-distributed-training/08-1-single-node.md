# 08-1 - 单机多卡训练

本章介绍单机多 GPU 的分布式训练配置。

## 基本配置

### 单机 8 卡

```bash
python -m verl.trainer.main_ppo \
    ... \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1
```

### 单机 4 卡

```bash
python -m verl.trainer.main_ppo \
    ... \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1
```

## Tensor Parallel 配置

Rollout 阶段使用 vLLM/SGLang 的 Tensor Parallel：

```bash
# 8 卡，TP=2（4 个 Rollout 组）
actor_rollout_ref.rollout.tensor_model_parallel_size=2

# 8 卡，TP=4（2 个 Rollout 组）
actor_rollout_ref.rollout.tensor_model_parallel_size=4

# 8 卡，TP=8（1 个 Rollout 组）
actor_rollout_ref.rollout.tensor_model_parallel_size=8
```

**选择建议**：
- 模型小（<7B）：TP=1-2
- 模型中（7B-32B）：TP=2-4
- 模型大（>32B）：TP=4-8

## GPU 内存配置

```bash
# vLLM GPU 内存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.5  # 推荐 0.4-0.6

# SGLang GPU 内存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

## 完整示例

### Qwen2-7B 单机 8 卡

```bash
#!/bin/bash
# run_single_node_8gpu.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='single_node' \
    trainer.experiment_name='qwen2_7b_8gpu'
```

## 常见问题

### OOM

**症状**：训练时 GPU 内存不足

**解决**：
1. 降低 `gpu_memory_utilization`
2. 减小 batch size
3. 启用 offload
4. 增加 TP

### 速度慢

**解决**：
1. 增加 `gpu_memory_utilization`（在 OOM 限制内）
2. 使用 VLLM_USE_V1=1
3. 调整 micro batch size

## 下一步

- [08-2-multi-node.md](08-2-multi-node.md) - 多机训练
