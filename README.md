# VERL 使用教程

> 基于 `verl-project/verl` v0.8.0 | 更新日期：2026-07-03

## 这是什么

这是一份面向实践者的 verl 框架使用教程。教程沿用“先跑通、再理解、再扩展”的学习路线，并将示例命令、配置键、算法目录和扩展 API 更新到 v0.8.0。

verl 是大模型后训练框架，核心目标是把策略模型、价值模型、参考模型、奖励函数/奖励模型、rollout 推理引擎和分布式训练后端组织成可扩展的 RLHF / Agentic RL / SFT 工作流。

这一版仍然保持原有阅读方式：先用 Part0 建立整体认识，再按工程训练或研究扩展两条路线深入。每篇文章之间增加了轻量跳转，方便顺读，也方便查阅。

## 本版更新重点

| 方向 | v0.8.0 版写法 |
| --- | --- |
| 官方仓库 | `https://github.com/verl-project/verl` |
| PPO/GRPO 入口 | `python -m verl.trainer.main_ppo` |
| SFT 入口 | `torchrun ... -m verl.trainer.sft_trainer` |
| Reward 配置 | 推荐 `reward.custom_reward_function.*`、`reward.reward_model.*`、`reward.reward_manager.*` |
| Rollout 后端 | `hf`、`vllm`、`sglang`、`trtllm` |
| 训练后端 | 默认 DP/FSDP，进阶可用 `model_engine=megatron`、`veomni`、`torchtitan`、`automodel` |
| 多轮/工具 | `actor_rollout_ref.rollout.multi_turn.*` + `tool_config_path` / `function_tool_path` / `agent.*` |
| 自定义扩展 | `register_adv_est`、`register_policy_loss`、`reward_manager.register` |

> 说明：verl 的 `main` 分支更新很快。本教程以最新稳定 release v0.8.0 为基准；若使用 `main`，优先对照源码中的 `verl/trainer/config/_generated_ppo_trainer.yaml`。

## 适用读者

| 读者类型 | 关注点 | 建议路径 |
| --- | --- | --- |
| 工程人员 | 跑通 PPO/GRPO/SFT/AgentRL、调 batch、调后端、排错 | [Part0](#part0-基础) -> [Part1](#part1-工程训练) -> [Appendix](#appendix-附录) |
| 研究人员 | 改 advantage、改 policy loss、改 reward manager、复现论文 | [Part0](#part0-基础) -> [Part2](#part2-研究扩展)，并回看 Part1 的可运行脚本 |
| 刚接触 RLHF 的学习者 | 先理解 actor/critic/rollout/reward，再逐步上手 | [00](part0-foundation/00-what-is-verl.md) -> [02](part0-foundation/02-quick-start.md) -> [03](part0-foundation/03-core-concepts.md) -> [05-2](part1-engineering-training/05-algorithms/05-2-grpo.md) -> [09](part1-engineering-training/09-data-and-reward/09-1-data-preprocess.md) |

## 教程结构

```text
verl-tutorial/
│
├── part0-foundation/              # 基础（必读）
│   ├── 00-what-is-verl.md         #   verl 是什么
│   ├── 01-installation.md         #   安装配置
│   ├── 02-quick-start.md          #   快速上手
│   ├── 03-core-concepts.md        #   核心概念
│   └── 04-configuration.md        #   配置系统
│
├── part1-engineering-training/    # 工程训练（面向工程人员）
│   ├── 05-algorithms/             #   内置算法使用
│   ├── 06-sft-and-tuning/         #   SFT 与微调
│   ├── 07-agent-rl/               #   AgentRL 训练
│   ├── 08-distributed-training/   #   分布式训练
│   ├── 09-data-and-reward/        #   数据与奖励
│   └── 10-operations/             #   运维与监控
│
├── part2-research-extension/      # 研究扩展（面向研究人员）
│   ├── 11-extension-overview.md   #   扩展机制总览
│   ├── 12-custom-algorithm/       #   自定义算法组件
│   ├── 13-modify-workers/         #   修改 Worker
│   └── 14-paper-reproduction/     #   论文复现
│
└── appendix/                      # 附录
    ├── A-config-reference.md      #   配置参数参考
    ├── B-common-errors.md         #   常见错误
    └── C-resources.md             #   进阶资源
```

## 阅读路径

### 工程人员路径

1. [00-04 基础章节](#part0-基础)：掌握 verl 的组件、配置树和最小命令。
2. [05 算法章节](#05-算法训练)：根据任务选择 PPO、GRPO、RLOO、ReMax、GPG、GSPO 等。
3. [08 分布式章节](#08-分布式训练)：确定 FSDP/FSDP2/Megatron/VeOmni 与 rollout 后端。
4. [09 数据与奖励章节](#09-数据与奖励)：准备 parquet、奖励函数、Reward Model。
5. [10 运维章节](#10-运维与调优)：监控、checkpoint、profiling、集群调度。
6. [06 SFT](#06-sft-与模型调参) / [07 AgentRL](#07-agent-rl)：按任务补齐监督微调或工具调用训练。

### 研究人员路径

1. [00-04 基础章节](#part0-基础)：确认 v0.8.0 的配置和运行入口。
2. [11 扩展机制总览](part2-research-extension/11-extension-overview.md)：建立扩展点地图。
3. [12 自定义算法](#12-自定义算法)：实现 advantage、policy loss、reward manager。
4. [13 修改 Worker](#13-修改-worker)：理解 Actor/Critic/Rollout Worker 的边界。
5. [14 论文复现](#14-论文复现)：把论文拆成配置、代码、数据、验证四部分。

## 完整目录

<details>
<summary>点击展开全部章节链接</summary>

### Part0 基础

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 00 | 什么是 verl | [进入](part0-foundation/00-what-is-verl.md) |
| 01 | 安装与配置 | [进入](part0-foundation/01-installation.md) |
| 02 | 快速上手 | [进入](part0-foundation/02-quick-start.md) |
| 03 | 核心概念 | [进入](part0-foundation/03-core-concepts.md) |
| 04 | 配置系统 | [进入](part0-foundation/04-configuration.md) |

### Part1 工程训练

#### 05 算法训练

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 05-1 | PPO 训练 | [进入](part1-engineering-training/05-algorithms/05-1-ppo.md) |
| 05-2 | GRPO 训练 | [进入](part1-engineering-training/05-algorithms/05-2-grpo.md) |
| 05-3 | REINFORCE++ / RLOO / ReMax 训练 | [进入](part1-engineering-training/05-algorithms/05-3-reinforce-pp.md) |
| 05-4 | RLOO / ReMax 详解 | [进入](part1-engineering-training/05-algorithms/05-4-rloo-remax.md) |
| 05-5 | 其他算法 | [进入](part1-engineering-training/05-algorithms/05-5-other-algos.md) |

#### 06 SFT 与模型调参

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 06-1 | SFT 基础训练 | [进入](part1-engineering-training/06-sft-and-tuning/06-1-sft-basics.md) |
| 06-2 | 模型规模与调参 | [进入](part1-engineering-training/06-sft-and-tuning/06-2-model-tuning.md) |

#### 07 Agent RL

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 07-1 | Tool Calling 训练 | [进入](part1-engineering-training/07-agent-rl/07-1-tool-calling.md) |
| 07-2 | 多轮对话 RL 训练 | [进入](part1-engineering-training/07-agent-rl/07-2-multi-turn.md) |
| 07-3 | Agent Loop 训练 | [进入](part1-engineering-training/07-agent-rl/07-3-agent-loop.md) |

#### 08 分布式训练

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 08-1 | 单机多卡训练 | [进入](part1-engineering-training/08-distributed-training/08-1-single-node.md) |
| 08-2 | 多机训练 | [进入](part1-engineering-training/08-distributed-training/08-2-multi-node.md) |
| 08-3 | 训练后端选择 | [进入](part1-engineering-training/08-distributed-training/08-3-backends.md) |
| 08-4 | 推理引擎配置 | [进入](part1-engineering-training/08-distributed-training/08-4-inference-engine.md) |

#### 09 数据与奖励

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 09-1 | 数据预处理 | [进入](part1-engineering-training/09-data-and-reward/09-1-data-preprocess.md) |
| 09-2 | 自定义数据集 | [进入](part1-engineering-training/09-data-and-reward/09-2-custom-dataset.md) |
| 09-3 | 奖励配置 | [进入](part1-engineering-training/09-data-and-reward/09-3-reward-config.md) |

#### 10 运维与调优

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 10-1 | 训练监控 | [进入](part1-engineering-training/10-operations/10-1-monitoring.md) |
| 10-2 | 检查点管理 | [进入](part1-engineering-training/10-operations/10-2-checkpoint.md) |
| 10-3 | 性能分析与调优 | [进入](part1-engineering-training/10-operations/10-3-profiling.md) |
| 10-4 | 集群调度 | [进入](part1-engineering-training/10-operations/10-4-cluster.md) |

### Part2 研究扩展

#### 11 扩展总览

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 11 | 扩展机制总览 | [进入](part2-research-extension/11-extension-overview.md) |

#### 12 自定义算法

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 12-1 | 扩展 Advantage Estimator | [进入](part2-research-extension/12-custom-algorithm/12-1-advantage.md) |
| 12-2 | 自定义 Policy Loss | [进入](part2-research-extension/12-custom-algorithm/12-2-policy-loss.md) |
| 12-3 | 自定义 Reward Manager | [进入](part2-research-extension/12-custom-algorithm/12-3-reward-manager.md) |
| 12-4 | 完整算法实现示例 | [进入](part2-research-extension/12-custom-algorithm/12-4-full-example.md) |

#### 13 修改 Worker

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 13-1 | 修改 Actor Worker | [进入](part2-research-extension/13-modify-workers/13-1-actor-worker.md) |
| 13-2 | 修改 Critic Worker | [进入](part2-research-extension/13-modify-workers/13-2-critic-worker.md) |
| 13-3 | 修改 Rollout Worker | [进入](part2-research-extension/13-modify-workers/13-3-rollout-worker.md) |

#### 14 论文复现

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| 14-1 | 论文复现工作流 | [进入](part2-research-extension/14-paper-reproduction/14-1-reproduce-workflow.md) |
| 14-2 | 案例研究 | [进入](part2-research-extension/14-paper-reproduction/14-2-case-studies.md) |

### Appendix 附录

| 章节 | 主题 | 入口 |
| --- | --- | --- |
| A | 配置参数参考 | [进入](appendix/A-config-reference.md) |
| B | 常见错误与解决 | [进入](appendix/B-common-errors.md) |
| C | 进阶学习资源 | [进入](appendix/C-resources.md) |

</details>

## 代码约定

```bash
# RL 主入口
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.name=vllm

# SFT 主入口
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m verl.trainer.sft_trainer \
  engine=fsdp \
  data.micro_batch_size_per_gpu=4
```

## 事实校验方法

本教程所有配置优先对照以下 v0.8.0 源文件：

- `verl/trainer/config/_generated_ppo_trainer.yaml`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/sft_trainer_engine.yaml`
- `verl/trainer/config/rollout/rollout.yaml`
- `verl/trainer/config/reward/reward.yaml`
- `examples/README.md`

## 许可证

本教程采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。

verl 框架采用 Apache 2.0 许可证。
