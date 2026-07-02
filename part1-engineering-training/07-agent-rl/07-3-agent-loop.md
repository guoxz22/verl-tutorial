# 07-3 - Agent Loop 训练

<!-- NAV_START -->
> 阅读： [← 07-2 - 多轮对话 RL 训练](07-2-multi-turn.md) · [目录](../../README.md#完整目录) · [08-1 - 单机多卡训练 →](../08-distributed-training/08-1-single-node.md)
<!-- NAV_END -->

AgentLoop 是“怎么与环境交互”的代码层抽象。Tool Calling 只是 AgentLoop 的一种形态；搜索、浏览器、LangGraph、游戏环境、代码沙箱等更复杂的 loop 也可以放在这一层实现。

## v0.8.0 配置入口

```bash
actor_rollout_ref.rollout.agent.num_workers=8 \
actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/agent_loop.yaml
```

常见内置 loop：

| 名称 | 用途 |
| --- | --- |
| `single_turn_agent` | 默认单轮 rollout |
| `tool_agent` | 工具调用式多轮 rollout |

如果数据中有 `agent_name` 字段，rollout 可以按样本选择 agent；否则使用 `default_agent_loop`。

## 训练脚本骨架

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.return_raw_chat=True \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.function_tool_path=$PWD/tools.py \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  reward.custom_reward_function.path=$PWD/reward_fn.py \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=8 \
  trainer.logger='["console"]'
```

## 自定义 AgentLoop 的思路

1. 继承官方 agent loop 基类或参考 `verl/experimental/agent_loop/` 中的实现。
2. 保持 token 与文本的一致性：训练要用 rollout 引擎实际生成的 token。
3. 把外部环境调用封装成工具或 async client，避免阻塞 GPU。
4. 在 parquet 中加入必要的 `extra_info`，比如 task id、工具参数、ground truth。
5. 用 trace 先观察每轮消息，再放大 batch。

## AgentLoop 配置文件

复杂 agent 可以把参数放到 `agent_loop_config_path`：

```yaml
# agent_loop.yaml
tool_agent:
  max_steps: 4
  stop_on_tool_error: false
  trace_tool_output: true
```

启动时：

```bash
actor_rollout_ref.rollout.agent.agent_loop_config_path=$PWD/agent_loop.yaml
```

## 何时需要自定义 loop

| 场景 | 是否需要 |
| --- | --- |
| 只做数学答案验证 | 通常不需要，自定义 reward 即可 |
| 调用无状态函数工具 | 不一定，`function_tool_path` 足够 |
| 每个样本有独立沙箱/浏览器/数据库状态 | 需要 BaseTool 或自定义 AgentLoop |
| 需要 LangGraph/ReAct/规划器 | 通常需要自定义 AgentLoop |
| 要训练多 agent 协作 | 需要自定义 AgentLoop 和数据协议 |

## 常见坑

- 不要用 `data.tool_provider`。v0.8.0 的选择入口是 `agent.default_agent_loop` 或数据里的 `agent_name`。
- 工具不应返回无限长文本，需要设置 `max_tool_response_length`。
- 工具调用失败不是总是训练失败；先看 trace，再决定 reward 是否惩罚。
- 多轮任务先用 `rollout.n=1` 调试格式，再增加采样数。

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 07-2 - 多轮对话 RL 训练](07-2-multi-turn.md) · [目录](../../README.md#完整目录) · [08-1 - 单机多卡训练 →](../08-distributed-training/08-1-single-node.md)
<!-- NAV_BOTTOM_END -->
