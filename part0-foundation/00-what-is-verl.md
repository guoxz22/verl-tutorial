# 00 - 什么是 verl

## 概述

**verl**（Volcano Engine Reinforcement Learning）是字节跳动开源的大语言模型强化学习训练框架。它是一个灵活、高效、生产就绪的 RLHF（Reinforcement Learning from Human Feedback）训练库。

verl 是论文 **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)** 的开源实现。

## 核心特性

### 1. 多种 RL 算法支持

verl 内置支持多种主流 RL 算法，无需修改代码即可切换：

| 算法 | 用途 | 配置值 |
|------|------|--------|
| PPO | 经典 Policy Gradient | `algorithm.adv_estimator=gae` |
| GRPO | Group-wise PPO，DeepSeek-R1 使用 | `algorithm.adv_estimator=grpo` |
| REINFORCE++ | 方差优化的 Policy Gradient | `algorithm.adv_estimator=reinforce_plus_plus` |
| RLOO | Rollout-based 优化 | `algorithm.adv_estimator=rloo` |
| ReMax | 奖励最大化 | 需要单独配置 |

### 2. 多种训练后端

| 后端 | 特点 | 适用场景 |
|------|------|----------|
| FSDP | PyTorch 原生，易用 | 中小模型（<70B） |
| FSDP2 | 更高性能，支持 CPU Offload | 内存受限场景 |
| Megatron-LM | 大规模分布式 | 超大模型（671B+） |

### 3. 多种推理引擎

| 引擎 | 特点 | 配置 |
|------|------|------|
| vLLM | 高吞吐推理 | `actor_rollout_ref.rollout.name=vllm` |
| SGLang | 支持多轮、Agent | `actor_rollout_ref.rollout.name=sglang` |
| TRTLLM | NVIDIA 优化 | `actor_rollout_ref.rollout.name=trtllm` |

### 4. 灵活的设备映射

verl 支持将不同模型（Actor、Critic、Reference、Reward Model）放置到不同的 GPU 组上，实现资源的高效利用。

### 5. AgentRL 支持

- 多轮对话 RL 训练
- Tool Calling 训练
- Agent Loop 支持

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        verl 架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   训练入口   │  │  配置系统   │  │     监控系统        │ │
│  │  main_ppo   │  │   Hydra     │  │  WandB/TensorBoard  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    RayPPOTrainer                       │ │
│  │  (协调各 Worker 的训练流程)                            │ │
│  └───────────────────────┬───────────────────────────────┘ │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │
│  │ActorRollout│  │   Critic   │  │   Reward Manager   │   │
│  │   Worker   │  │   Worker   │  │   (可选)           │   │
│  └─────┬──────┘  └─────┬──────┘  └────────────────────┘   │
│        │               │                                   │
│        ▼               ▼                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   训练后端                           │  │
│  │     FSDP / FSDP2 / Megatron-LM                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   推理引擎                           │  │
│  │     vLLM / SGLang / TRTLLM                          │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## 训练流程

典型的 verl RL 训练流程：

```
1. 加载 Prompt 数据
        │
        ▼
2. Rollout 生成 (Actor 模型生成 Response)
        │
        ▼
3. 计算 Reward (函数奖励或模型奖励)
        │
        ▼
4. 计算 Advantage (GAE/GRPO 等)
        │
        ▼
5. 更新 Actor (Policy Loss)
        │
        ▼
6. 更新 Critic (Value Loss，可选)
        │
        ▼
7. 保存 Checkpoint / 验证
        │
        └──────► 重复 1-7 直到训练结束
```

## 与其他框架对比

| 特性 | verl | DeepSpeed-Chat | OpenRLHF | NeMo-Aligner |
|------|------|----------------|----------|--------------|
| 灵活算法扩展 | ✅ 强 | ⚠️ 一般 | ⚠️ 一般 | ⚠️ 一般 |
| 推理引擎集成 | ✅ vLLM/SGLang | ❌ | ⚠️ 部分 | ⚠️ 部分 |
| 超大模型支持 | ✅ 671B | ⚠️ | ⚠️ | ✅ |
| AgentRL | ✅ | ❌ | ⚠️ | ❌ |
| 混合引擎 | ✅ 3D-HybridEngine | ❌ | ❌ | ❌ |

## 适用场景

### 适合使用 verl 的场景

- 大模型的 RLHF/RLAIF 训练
- 数学推理、代码生成等可验证奖励任务
- 多轮对话、Agent 训练
- 自定义 RL 算法研究
- 大规模分布式训练

### 可能不适合的场景

- 纯监督学习（建议使用 verl 的 SFT 功能或直接用 transformers）
- 非 LLM 的传统 RL 任务（建议使用 RLlib、Stable-Baselines3 等）
- 极小规模实验（单卡以下）

## 相关资源

- [GitHub 仓库](https://github.com/volcengine/verl)
- [官方文档](https://verl.readthedocs.io/)
- [HybridFlow 论文](https://arxiv.org/abs/2409.19256)
