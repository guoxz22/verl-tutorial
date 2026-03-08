# 10-3 - 性能分析与调优

本章介绍如何分析和优化 verl 训练性能。

## 性能分析工具

### PyTorch Profiler

```bash
global_profiler.tool=torch \
global_profiler.steps="[10, 20]" \
global_profiler.save_path=./profiles
```

### Nsight Systems

```bash
global_profiler.tool=nsys \
global_profiler.steps="[10]" \
global_profiler.global_tool_config.nsys.controller_nsight_options.trace="cuda,nvtx,cublas,ucx"
```

### 内存分析

```bash
global_profiler.tool=torch_memory \
global_profiler.steps="[10]" \
global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=100000
```

## 性能调优指南

### 1. Rollout 优化

```bash
# 使用 vLLM V1
export VLLM_USE_V1=1

# 调整 GPU 内存
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# 调整批处理
actor_rollout_ref.rollout.max_num_batched_tokens=16384
actor_rollout_ref.rollout.max_num_seqs=2048
```

### 2. 训练优化

```bash
# 使用 FSDP2
actor_rollout_ref.actor.strategy=fsdp2

# 启用 torch.compile
actor_rollout_ref.actor.use_torch_compile=True

# 优化 micro batch
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64
```

### 3. 内存优化

```bash
# Gradient Checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing=True

# CPU Offload (FSDP2)
actor_rollout_ref.actor.fsdp_config.offload_policy=True

# Reference Offload
actor_rollout_ref.ref.fsdp_config.param_offload=True

# Remove Padding
actor_rollout_ref.model.use_remove_padding=True
```

### 4. 通信优化

```bash
# 序列并行
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2

# Tensor Parallel
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

## 常见瓶颈

### Rollout 瓶颈

**症状**：Rollout 时间占比高

**解决**：
- 增加 `gpu_memory_utilization`
- 调整 `max_num_batched_tokens`
- 考虑使用 SGLang

### 训练瓶颈

**症状**：训练时间占比高

**解决**：
- 增加 `ppo_micro_batch_size_per_gpu`
- 使用 FSDP2 + torch.compile
- 减少梯度检查点

### 通信瓶颈

**症状**：多机训练慢

**解决**：
- 检查网络带宽
- 优化 NCCL 配置
- 考虑 Pipeline Parallel

## 性能指标

```bash
# 关键指标
train/step_time          # 每步时间
train/rollout_time       # Rollout 时间
train/update_time        # 更新时间
train/samples_per_second # 吞吐量
```

## 下一步

- [10-4-cluster.md](10-4-cluster.md) - 集群调度
