# 00 - 什么是 verl

<!-- NAV_START -->
> 阅读： [目录](../README.md#catalog) · [01 - 安装与配置 →](01-installation.md)
<!-- NAV_END -->

## 概述

verl 是一个面向大模型后训练的强化学习框架，也是 HybridFlow 论文思想的开源实现。它把 RLHF / RLAIF / 可验证奖励 / Agentic RL 中常见的模块抽象成统一训练流：数据、rollout、reward、advantage、actor update、checkpoint 和监控。

本教程基于 `verl-project/verl` v0.8.0。学习时需要注意：verl 更新很快，命令能否运行主要取决于当前 tag 与配置键是否匹配。

## 一句话理解

```text
verl = Ray 分布式编排 + Hydra 配置 + PyTorch 训练后端 + 高吞吐 rollout 后端 + 可扩展 RL 算法
```

## 核心组件

| 组件 | 作用 |
| --- | --- |
| Actor | 被训练的策略模型，生成 response 并接受 policy loss 更新 |
| Rollout | 用 vLLM/SGLang/HF/TRTLLM 等后端批量生成样本 |
| Reference | 参考策略，用于 KL 或 logprob 对比 |
| Critic | 价值模型，PPO/GAE 常用；GRPO/RLOO 等可不用 |
| Teacher | OPD 场景下的冻结强模型，为学生轨迹提供 logprob / top-k 分布 |
| Reward | 函数奖励、Reward Manager、Reward Model、Sandbox |
| Trainer | RayPPOTrainer 负责调度数据流和 worker |

## 支持的算法入口

| 类型 | 代表算法 | 配置 |
| --- | --- | --- |
| actor-critic | PPO | `algorithm.adv_estimator=gae` |
| critic-free group RL | GRPO | `algorithm.adv_estimator=grpo` |
| leave-one-out baseline | RLOO | `algorithm.adv_estimator=rloo` |
| greedy baseline | ReMax | `algorithm.adv_estimator=remax` |
| policy-gradient baseline | REINFORCE++ / GPG | `reinforce_plus_plus` / `gpg` |
| policy loss 变体 | GSPO / SAPO / CISPO / GMPO / DPPO | `actor_rollout_ref.actor.policy_loss.loss_mode=*` |
| 特殊训练 | MTP / OPD | `examples/mtp_trainer` / `distillation.enabled=True` |

## 训练后端

| 后端 | 适合场景 |
| --- | --- |
| FSDP/FSDP2 | 入门、单机多卡、中小模型研究 |
| Megatron-LM | 多机大模型、TP/PP/CP/EP/MoE |
| VeOmni | MoE、多模态、NPU/GPU 进阶场景 |
| TorchTitan | TorchTitan 生态实验 |
| Automodel | 官方新统一 engine 示例路径 |

## Rollout 后端

| 后端 | 配置 | 适合场景 |
| --- | --- | --- |
| HF | `actor_rollout_ref.rollout.name=hf` | 调试、小模型 |
| vLLM | `actor_rollout_ref.rollout.name=vllm` | 通用高吞吐 rollout |
| SGLang | `actor_rollout_ref.rollout.name=sglang` | 多轮、工具、AgentRL |
| TRTLLM | `actor_rollout_ref.rollout.name=trtllm` | NVIDIA TensorRT-LLM 高性能场景 |

## 典型训练流程

```text
1. 读取 parquet 数据
2. Actor 通过 rollout 后端生成 response
3. Reward Manager / Reward Model 计算 token-level reward
4. core_algos 计算 advantage / returns
5. Actor 根据 policy loss 更新
6. Critic 更新（如果算法需要）
7. 验证、记录日志、保存 checkpoint
```

## 为什么适合研究和工程

- **研究友好**：advantage、policy loss、reward manager 都有注册器。
- **工程友好**：官方 examples 覆盖 PPO、GRPO、SFT、LoRA、多轮、MTP、大模型后端。
- **可扩展**：Ray worker 层隔离训练、rollout、reward，方便独立扩展。
- **可观测**：支持 console、wandb、profile、rollout trace、checkpoint merge。

## 不适合什么

- 只想做普通分类/回归任务，直接用 PyTorch/Transformers 更轻。
- 非语言模型的传统 RL 环境，RLlib 或 Stable-Baselines3 更直接。
- 没有 GPU 却想完整跑 RL 训练，只能阅读和做小规模 CPU 代码检查。

## 下一步

- [01-installation.md](01-installation.md) - 安装环境
- [02-quick-start.md](02-quick-start.md) - 跑通 quickstart
- [03-core-concepts.md](03-core-concepts.md) - 深入组件概念

---

<!-- NAV_BOTTOM_START -->
> 阅读： [目录](../README.md#catalog) · [01 - 安装与配置 →](01-installation.md)
<!-- NAV_BOTTOM_END -->
