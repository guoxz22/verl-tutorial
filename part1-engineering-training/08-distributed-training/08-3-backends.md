# 08-3 - 训练后端选择

verl 支持多种训练后端，本章介绍如何选择和配置。

## 后端对比

| 后端 | 特点 | 适用场景 |
|------|------|----------|
| FSDP | PyTorch 原生，易用 | 研究、原型、中小模型 |
| FSDP2 | 更高性能，CPU Offload | 内存受限、追求性能 |
| Megatron-LM | 大规模优化 | 超大模型、多机训练 |

## FSDP

### 基本配置

```bash
actor_rollout_ref.actor.strategy=fsdp
critic.strategy=fsdp
```

### 完整示例

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    critic.strategy=fsdp \
    ...
```

## FSDP2

### 特点

- 更好的内存管理
- 支持 CPU Offload
- 更高的训练吞吐

### 启用 FSDP2

```bash
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2
actor_rollout_ref.ref.strategy=fsdp2
```

### CPU Offload

```bash
# 参数 Offload
actor_rollout_ref.actor.fsdp_config.offload_policy=True

# 优化器 Offload
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

### 完整示例

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.tiled_mlp.enabled=True \
    actor_rollout_ref.model.tiled_mlp.num_shards=4 \
    critic.strategy=fsdp2 \
    ...
```

## Megatron-LM

### 特点

- Tensor Parallel
- Pipeline Parallel
- Expert Parallel（MoE）
- 适合超大模型

### 基本配置

```bash
actor_rollout_ref.actor.strategy=megatron
critic.strategy=megatron
```

### 完整示例

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.model.path=/path/to/megatron/checkpoint \
    actor_rollout_ref.model.override_config.moe_config.freeze_moe_router=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=2 \
    actor_rollout_ref.rollout.expert_model_parallel_size=8 \
    critic.strategy=megatron \
    ...
```

## 选择指南

```
            ┌─────────────────────┐
            │ 模型大小？          │
            └──────────┬──────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
        < 7B        7B-70B       > 70B
          │            │            │
          ▼            ▼            ▼
       ┌──────┐    ┌──────┐    ┌──────────┐
       │ FSDP │    │FSDP2 │    │ Megatron │
       └──────┘    └──────┘    └──────────┘
          │            │            │
          │            │            │
          ▼            ▼            ▼
       快速实验    内存优化    大规模训练
```

## 下一步

- [08-4-inference-engine.md](08-4-inference-engine.md) - 推理引擎配置
