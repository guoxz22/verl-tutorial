# 08-4 - 推理引擎配置

verl 支持多种推理引擎用于 Rollout 生成。

## 引擎对比

| 引擎 | 特点 | 适用场景 |
|------|------|----------|
| vLLM | 高吞吐，稳定 | 标准训练任务 |
| SGLang | 多轮、Agent、灵活 | AgentRL、多轮对话 |
| HF | 简单，调试方便 | 小规模实验 |
| TRTLLM | NVIDIA 优化 | 生产部署 |

## vLLM

### 基本配置

```bash
actor_rollout_ref.rollout.name=vllm
```

### 关键参数

```bash
# Tensor Parallel
actor_rollout_ref.rollout.tensor_model_parallel_size=2

# GPU 内存
actor_rollout_ref.rollout.gpu_memory_utilization=0.6

# 采样参数
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
actor_rollout_ref.rollout.top_k=-1

# 批处理
actor_rollout_ref.rollout.max_num_batched_tokens=8192
actor_rollout_ref.rollout.max_num_seqs=1024

# 数据类型
actor_rollout_ref.rollout.dtype=bfloat16
```

### vLLM V1（推荐）

```bash
export VLLM_USE_V1=1
```

### 完整示例

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    ...
```

## SGLang

### 基本配置

```bash
actor_rollout_ref.rollout.name=sglang
```

### 多轮配置

```bash
# 启用多轮
actor_rollout_ref.rollout.multi_turn.enable=true
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
```

### Server 模式

```bash
# 使用 Server 模式（推荐用于多轮）
actor_rollout_ref.rollout.multi_turn.server_mode=true
```

### 完整示例

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    ...
```

## HuggingFace

### 基本配置

```bash
actor_rollout_ref.rollout.name=hf
```

### 适用场景

- 小规模实验
- 调试
- 兼容性测试

## TRTLLM

### 基本配置

```bash
actor_rollout_ref.rollout.name=trtllm
```

### 预备工作

需要先构建 TRTLLM 引擎：

```bash
# 转换模型
trtllm-build --model-path /path/to/model --output-dir /path/to/engine
```

## 选择指南

| 场景 | 推荐引擎 |
|------|----------|
| 标准 RL 训练 | vLLM |
| AgentRL / 多轮 | SGLang |
| 快速实验 / 调试 | HF |
| 生产部署 | TRTLLM |

## 性能优化

### vLLM 调优

```bash
# 增大 GPU 内存利用率（在 OOM 限制内）
actor_rollout_ref.rollout.gpu_memory_utilization=0.7

# 调整批处理大小
actor_rollout_ref.rollout.max_num_batched_tokens=16384
actor_rollout_ref.rollout.max_num_seqs=2048
```

### SGLang 调优

```bash
# Server 模式减少启动开销
actor_rollout_ref.rollout.multi_turn.server_mode=true

# 调整内存
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

## 下一步

- [09-data-and-reward](../09-data-and-reward/) - 数据与奖励
