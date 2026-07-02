# 14-2 - 案例研究

本章给出“如何把论文映射到 verl v0.8.0”的案例。重点不是复制某个固定脚本，而是学会找到官方入口、确认配置键、做最小复现。

## 案例 1：DeepSeekMath / GRPO

### 对应关系

| 论文概念 | verl 配置 |
| --- | --- |
| group sampling | `actor_rollout_ref.rollout.n>1` |
| group-relative advantage | `algorithm.adv_estimator=grpo` |
| rule-based math reward | `reward.custom_reward_function.*` 或默认 GSM8K/MATH reward |
| critic-free | `critic.enable=False` 或使用 GRPO 示例默认路径 |

### 最小命令骨架

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  trainer.logger='["console"]'
```

先用小模型确认 reward 和优势计算，再换大模型和官方 GRPO 脚本。

## 案例 2：DAPO

DAPO 更适合从官方 recipe 或 examples 入手，不建议手写全部细节。

关键点：

```bash
algorithm.adv_estimator=grpo \
reward.reward_manager.name=dapo \
actor_rollout_ref.rollout.n=<较大采样数>
```

复现时优先检查：

1. 数据过滤与 prompt 模板；
2. reward manager 是否为 `dapo`；
3. 长度、overlong filtering、rollout.n；
4. 是否启用动态 batch；
5. 官方 recipe 所要求的 verl commit / tag。

## 案例 3：GPG

GPG 同时改 advantage 和 policy loss：

```bash
algorithm.adv_estimator=gpg \
actor_rollout_ref.actor.policy_loss.loss_mode=gpg
```

注意 GPG 不是“Generalized Policy Gradient”，官方示例称为 Group Policy Gradient。

## 案例 4：GSPO / SAPO / CISPO

这些主要是 policy loss 变体：

```bash
# GSPO
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean

# SAPO
actor_rollout_ref.actor.policy_loss.loss_mode=sapo \
+actor_rollout_ref.actor.policy_loss.tau_pos=1.0 \
+actor_rollout_ref.actor.policy_loss.tau_neg=1.05

# CISPO
actor_rollout_ref.actor.policy_loss.loss_mode=cispo \
actor_rollout_ref.actor.clip_ratio_low=10 \
actor_rollout_ref.actor.clip_ratio_high=0.2
```

## 案例 5：MTP

MTP 是 Multi-Token-Prediction，不是 Multi-Turn PPO。v0.8.0 的官方入口在 `examples/mtp_trainer/`，常见配置：

```bash
actor_rollout_ref.model.mtp.enable=True \
actor_rollout_ref.model.mtp.enable_train=True \
actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1 \
actor_rollout_ref.model.mtp.detach_encoder=True
```

MTP 对模型结构和后端要求更严格，先跑官方 MiMo/Qwen/DeepSeek 系列示例，再改模型。

## 案例 6：ReTool / Agentic RL

v0.8.0 中不要使用旧键 `data.tool_provider`。工具和 agent 应通过：

```bash
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.multi_turn.tool_config_path=/path/to/tool_config.yaml \
actor_rollout_ref.rollout.multi_turn.function_tool_path=/path/to/tools.py \
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

数据中可以加入：

```python
{"agent_name": "tool_agent", "extra_info": {"tools_kwargs": {...}}}
```

## 复现检查表

| 检查项 | 问题 |
| --- | --- |
| 版本 | 论文 recipe 要求的是 v0.8.0、main，还是固定 commit？ |
| 数据 | prompt 模板、过滤、ground truth 字段是否一致？ |
| 奖励 | reward function/manager 是否真的被调用？ |
| 算法 | 改的是 `adv_estimator` 还是 `policy_loss.loss_mode`？ |
| 后端 | FSDP/Megatron/VeOmni 与原实验是否一致？ |
| 采样 | `rollout.n`、温度、max length 是否一致？ |
| 监控 | 是否记录 score、length、KL、clipfrac、entropy？ |

复现不要一开始追求同等规模；先做 1 step 数值链路，再做小模型趋势，最后扩大模型和数据。
