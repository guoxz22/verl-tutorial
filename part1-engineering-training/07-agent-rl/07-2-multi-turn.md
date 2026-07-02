# 07-2 - 多轮对话 RL 训练

<!-- NAV_START -->
> 阅读： [← 07-1 - Tool Calling 训练](07-1-tool-calling.md) · [目录](../../README.md#catalog) · [07-3 - Agent Loop 训练 →](07-3-agent-loop.md)
<!-- NAV_END -->

多轮 RL 解决的是：一次样本不再只有 `prompt -> response`，而是可能包含多轮 assistant、tool、user 交互。v0.8.0 把这部分统一放在 `actor_rollout_ref.rollout.multi_turn` 下。

## 最小配置

```bash
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.multi_turn.format=hermes \
actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
data.return_raw_chat=True
```

也可以使用 vLLM 做部分 agentic rollout，但官方文档和示例对 SGLang 的多轮工具链覆盖更完整。第一次学习建议从 SGLang 开始。

## 关键参数

| 参数 | 含义 |
| --- | --- |
| `multi_turn.enable` | 是否启用多轮 rollout |
| `multi_turn.max_assistant_turns` | assistant 最多生成多少轮 |
| `multi_turn.max_user_turns` | 用户/环境最多返回多少轮 |
| `multi_turn.max_parallel_calls` | 并行工具调用上限 |
| `multi_turn.max_tool_response_length` | 工具返回 token 长度上限 |
| `multi_turn.tool_response_truncate_side` | 工具返回截断方向，默认 `middle` |
| `multi_turn.tokenization_sanity_check_mode` | 多轮增量 tokenization 校验，默认 `strict` |
| `multi_turn.use_inference_chat_template` | 是否使用推理模板处理历史消息 |

## 为什么需要 `data.return_raw_chat=True`

多轮训练需要保留原始 messages，rollout 才能在生成过程中不断拼接新消息、工具返回和 assistant 回复。因此数据读取阶段应保留 raw chat：

```bash
data.return_raw_chat=True
```

如果只保留 tokenized prompt，多轮 AgentLoop 很难正确恢复每轮消息边界。

## Tokenization sanity check

多轮 rollout 会采用“增量 tokenization”：每生成一轮 assistant，只把新增 assistant 内容加入 loss mask。v0.8.0 默认会做一致性检查：

```bash
actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=strict
```

可选值：

- `strict`：严格检查，适合新任务调试。
- `ignore_strippable`：忽略空白字符差异。
- `disable`：关闭检查，只在 chat template 行为已确认可靠时使用。

## 多轮长度控制

多轮训练最常见的问题是上下文爆炸。建议同时控制：

```bash
data.max_prompt_length=2048 \
data.max_response_length=2048 \
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=256
```

如果仍然 OOM，优先降低 `rollout.n`、`ppo_micro_batch_size_per_gpu` 和 max length，而不是先改算法。

## 数据样例

```python
{
  "data_source": "custom_multiturn",
  "prompt": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Solve this task."}
  ],
  "reward_model": {"style": "rule", "ground_truth": "..."},
  "extra_info": {"split": "train", "index": 0}
}
```

如果要工具调用，额外加入 `agent_name` 和工具配置，见 [07-1-tool-calling.md](07-1-tool-calling.md)。

## 不再推荐的旧写法

```bash
# 不要使用
actor_rollout_ref.rollout.multi_turn.server_mode=true
actor_rollout_ref.rollout.multi_turn.tool.dict.python_executor=true
```

v0.8.0 的 server/async 能力由 rollout `mode`、后端适配器、`agent` 和 `multi_turn` 子配置共同控制，不再用这个旧字段表达。

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 07-1 - Tool Calling 训练](07-1-tool-calling.md) · [目录](../../README.md#catalog) · [07-3 - Agent Loop 训练 →](07-3-agent-loop.md)
<!-- NAV_BOTTOM_END -->
