# 03 - 核心概念

<!-- NAV_START -->
> 阅读： [← 02 - 快速上手](02-quick-start.md) · [目录](../README.md#catalog) · [04 - 配置系统 →](04-configuration.md)
<!-- NAV_END -->

理解 verl 的核心抽象是使用和扩展框架的基础。

## 整体架构

verl 采用 **混合控制器编程模型**（Hybrid Controller Programming Model），将训练流程分解为多个可独立配置和扩展的组件。

```
┌────────────────────────────────────────────────────────────────┐
│                      RayPPOTrainer                             │
│                    (训练协调器)                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌──────────────────┐   ┌──────────────────┐                 │
│   │ ActorRollout     │   │     Critic       │                 │
│   │ Worker           │   │     Worker       │                 │
│   │                  │   │                  │                 │
│   │ - Policy Network │   │ - Value Network  │                 │
│   │ - Rollout Engine │   │                  │                 │
│   │ - Reference      │   │                  │                 │
│   └────────┬─────────┘   └────────┬─────────┘                 │
│            │                      │                            │
│            └──────────┬───────────┘                            │
│                       │                                        │
│                       ▼                                        │
│            ┌─────────────────────┐                             │
│            │   Reward Manager    │                             │
│            │   (可选)            │                             │
│            └─────────────────────┘                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    DataProto        │
              │  (数据传输协议)      │
              └─────────────────────┘
```

## 核心组件

### 1. Actor（策略网络）

**作用**：决定在给定状态下采取什么行动（生成什么 token）。

**职责**：
- 训练时：计算 log probability，更新策略参数
- 推理时：生成 response（通过 Rollout）

**配置示例**：

```bash
actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=256 \
actor_rollout_ref.actor.use_kl_loss=True
```

### 2. Critic（价值网络）

**作用**：估计当前状态的价值，用于计算 Advantage。

**职责**：
- 估计 value function V(s)
- 为 Advantage 计算提供 baseline

**注意**：GRPO 等某些算法不需要 Critic（`trainer.critic_warmup=0`）。

**配置示例**：

```bash
critic.strategy=fsdp \
critic.optim.lr=1e-5
```

### 3. Rollout（推理引擎）

**作用**：使用 Actor 模型生成 response。

**支持的引擎**：

| 引擎 | 特点 | 配置 |
|------|------|------|
| vLLM | 高吞吐 | `actor_rollout_ref.rollout.name=vllm` |
| SGLang | 多轮、Agent | `actor_rollout_ref.rollout.name=sglang` |
| TRTLLM | NVIDIA 优化 | `actor_rollout_ref.rollout.name=trtllm` |

**配置示例**：

```bash
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.n=5 \  # 每个 prompt 生成 5 个 response
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

### 4. Reference Model（参考模型）

**作用**：提供 KL 散度计算的参考，防止策略偏离太远。

**启用条件**：
- `actor_rollout_ref.actor.use_kl_loss=True`（KL Loss）
- `algorithm.use_kl_in_reward=True`（KL Reward）

**配置示例**：

```bash
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.ref.fsdp_config.param_offload=True  # 节省内存
```

### 5. Teacher Model（OPD 教师模型）

**作用**：在 OPD 中提供冻结教师策略的 token 级监督。

**职责**：
- 接收学生 rollout 形成的 prompt + response 前缀
- 返回教师 logprob 或 top-k token 分布
- 在多教师 OPD 中按 `distillation.teacher_key` 路由不同样本

**配置示例**：

```bash
distillation.enabled=True \
distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-32B \
distillation.teacher_models.teacher_model.inference.name=vllm \
distillation.distillation_loss.loss_mode=k1
```

### 6. Reward Manager（奖励管理器）

**作用**：计算每个 response 的奖励。

**类型**：

| 类型 | 说明 | 配置 |
|------|------|------|
| 函数奖励 | 基于规则的奖励（如数学答案匹配） | 默认，需自定义 reward function |
| 模型奖励 | 使用 Reward Model | `reward.reward_model.enable=True` |

### 7. Advantage Estimator（优势估计器）

**作用**：计算 Advantage = Q(s,a) - V(s)，指导策略更新。

**内置算法**：

| 算法 | 说明 | 配置值 |
|------|------|--------|
| GAE | Generalized Advantage Estimation（PPO 默认） | `gae` |
| GRPO | Group-wise PPO（DeepSeek-R1 使用） | `grpo` |
| REINFORCE++ | 方差优化的 Policy Gradient | `reinforce_plus_plus` |
| RLOO | Rollout-based 优化 | `rloo` |

**配置示例**：

```bash
algorithm.adv_estimator=grpo \
algorithm.gamma=1.0 \
algorithm.lam=1.0
```

## 数据协议：DataProto

**DataProto** 是 verl 中数据在各组件间传输的标准格式。

### 结构

```python
from verl import DataProto

# DataProto 包含：
# - batch: TensorDict，存储张量数据（input_ids, attention_mask, rewards, etc.）
# - non_tensor_batch: dict，存储非张量数据（原始文本、元数据等）
# - meta_info: dict，存储元信息

data = DataProto(
    batch=TensorDict({
        'input_ids': torch.tensor([...]),      # 输入 token IDs
        'attention_mask': torch.tensor([...]), # 注意力掩码
        'position_ids': torch.tensor([...]),   # 位置 IDs
        'rewards': torch.tensor([...]),        # 奖励值
        'old_log_probs': torch.tensor([...]),  # 旧策略 log prob
        'advantages': torch.tensor([...]),     # Advantage
        'values': torch.tensor([...]),         # Critic 估计值
    }),
    non_tensor_batch={
        'data_source': [...],  # 原始 prompt 文本
    },
    meta_info={}
)
```

### 常用操作

```python
# 切片
data_slice = data[0:10]

# 拼接
data_combined = DataProto.concat([data1, data2])

# 填充到指定大小的倍数
data_padded, pad_size = pad_dataproto_to_divisor(data, size_divisor=8)

# 去除填充
data_unpadded = unpad_dataproto(data_padded, pad_size)
```

## 训练流程

一个完整的 PPO 训练迭代包含以下步骤：

```
┌─────────────────────────────────────────────────────────────┐
│                    训练迭代流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 数据加载                                                │
│     └─► 从 DataLoader 获取 Prompt batch                    │
│                                                             │
│  2. Rollout 生成                                            │
│     └─► Actor 模型生成 N 个 Response                        │
│                                                             │
│  3. Reward 计算                                             │
│     └─► 计算 Reward（函数或模型）                           │
│                                                             │
│  4. Reference Log Prob（如果启用 KL）                       │
│     └─► Reference Model 计算参考 log prob                   │
│                                                             │
│  5. Advantage 计算                                          │
│     └─► 使用 GAE/GRPO 等计算 Advantage                      │
│                                                             │
│  6. Policy Update                                           │
│     └─► 更新 Actor 参数                                     │
│                                                             │
│  7. Value Update（如果使用 Critic）                         │
│     └─► 更新 Critic 参数                                    │
│                                                             │
│  8. 日志 & 检查点                                           │
│     └─► 记录指标，保存模型                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Worker 和 WorkerGroup

### Worker

Worker 是执行具体计算任务的单元，运行在 Ray Actor 中。

**主要 Worker 类型**：

| Worker | 职责 |
|--------|------|
| `ActorRolloutRefWorker` | Actor 训练 + Rollout 生成 + Reference 计算 |
| `CriticWorker` | Critic 训练 |
| `RewardModelWorker` | Reward Model 推理（可选） |

### WorkerGroup

WorkerGroup 管理一组 Worker，协调分布式计算。

```python
from verl.single_controller.ray import RayWorkerGroup

# WorkerGroup 负责：
# - Worker 创建和初始化
# - 数据分发（Dispatch）
# - 结果收集（Collect）
# - 分布式同步
```

## 后端系统

### FSDP（Fully Sharded Data Parallel）

```bash
# 启用 FSDP
actor_rollout_ref.actor.strategy=fsdp

# FSDP2（推荐，更高性能）
actor_rollout_ref.actor.strategy=fsdp2
actor_rollout_ref.actor.fsdp_config.offload_policy=True  # CPU Offload
```

### Megatron-LM

适用于超大模型（70B+）：

```bash
actor_rollout_ref.actor.strategy=megatron \
actor_rollout_ref.model.path=/path/to/megatron/checkpoint
```

## 下一步

- 阅读 [04-configuration.md](04-configuration.md) 深入了解配置系统
- 查看 `examples/` 目录中的实际示例
- 阅读 Part1 学习如何使用内置功能
- 阅读 Part2 学习如何扩展框架

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 02 - 快速上手](02-quick-start.md) · [目录](../README.md#catalog) · [04 - 配置系统 →](04-configuration.md)
<!-- NAV_BOTTOM_END -->
