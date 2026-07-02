# VERL 使用教程

> 基于 `verl-project/verl` v0.8.0 | 更新日期：2026-07-02

## 这是什么

这是一份面向实践者的 verl 框架使用教程。它保留原教程“先跑通、再理解、再扩展”的学习路线，但把示例命令、配置键、算法目录和扩展 API 更新到 v0.8.0。

verl 是大模型后训练框架，核心目标是把策略模型、价值模型、参考模型、奖励函数/奖励模型、rollout 推理引擎和分布式训练后端组织成可扩展的 RLHF / Agentic RL / SFT 工作流。

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

> 说明：verl 的 `main` 分支更新很快。本教程以最新稳定 release v0.8.0 为基准；如果你使用 `main`，优先对照源码中的 `verl/trainer/config/_generated_ppo_trainer.yaml`。

## 适用读者

| 读者类型 | 你可能关心 | 建议路径 |
| --- | --- | --- |
| 工程人员 | 跑通 PPO/GRPO/SFT/AgentRL、调 batch、调后端、排错 | Part0 -> Part1 -> Appendix |
| 研究人员 | 改 advantage、改 policy loss、改 reward manager、复现论文 | Part0 -> Part2，并回看 Part1 的可运行脚本 |
| 刚接触 RLHF 的学习者 | 先理解 actor/critic/rollout/reward，再逐步上手 | 00 -> 02 -> 03 -> 05-2 -> 09 |

## 教程结构

```text
verl-tutorial/
├── part0-foundation/              # 基础：概念、安装、快速上手、配置
├── part1-engineering-training/    # 工程：算法、SFT、AgentRL、分布式、数据奖励、运维
├── part2-research-extension/      # 研究：扩展点、自定义算法、Worker 修改、论文复现
└── appendix/                      # 附录：配置速查、常见错误、资源索引
```

## 阅读路径

### 工程人员路径

1. `00-04`：掌握 verl 的组件、配置树和最小命令。
2. `05-algorithms`：根据任务选择 PPO、GRPO、RLOO、ReMax、GPG、GSPO 等。
3. `08-distributed-training`：确定 FSDP/FSDP2/Megatron/VeOmni 与 rollout 后端。
4. `09-data-and-reward`：准备 parquet、奖励函数、Reward Model。
5. `10-operations`：监控、checkpoint、profiling、集群调度。
6. `06/07`：根据任务补 SFT 或 AgentRL。

### 研究人员路径

1. `00-04`：确认 v0.8.0 的配置和运行入口。
2. `11-extension-overview`：建立扩展点地图。
3. `12-custom-algorithm`：实现 advantage、policy loss、reward manager。
4. `13-modify-workers`：理解 Actor/Critic/Rollout Worker 的边界。
5. `14-paper-reproduction`：把论文拆成配置、代码、数据、验证四部分。

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
