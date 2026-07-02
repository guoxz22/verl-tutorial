# 07-1 - Tool Calling 训练

<!-- NAV_START -->
> 阅读： [← 06-2 - 模型规模与调参](../06-sft-and-tuning/06-2-model-tuning.md) · [目录](../../README.md#catalog) · [07-2 - 多轮对话 RL 训练 →](07-2-multi-turn.md)
<!-- NAV_END -->

Tool Calling 的核心不是“给模型一个工具列表”这么简单，而是让 rollout 在生成过程中进入“模型 -> 工具 -> 模型”的闭环。v0.8.0 的关键配置在 `actor_rollout_ref.rollout.multi_turn` 和 `actor_rollout_ref.rollout.agent` 下。

## 三个概念

| 概念 | 作用 | 常用配置 |
| --- | --- | --- |
| `multi_turn` | 打开多轮 tokenization、工具调用和轮数限制 | `enable`、`tool_config_path`、`function_tool_path` |
| `agent` | 选择 rollout 时使用哪个 AgentLoop | `default_agent_loop`、`agent_loop_config_path` |
| tool config | 声明工具类、schema、执行参数 | YAML 文件或 `@function_tool` Python 文件 |

## 推荐配置骨架

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.return_raw_chat=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/path/to/tool_config.yaml \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

> v0.8.0 不使用旧写法 `actor_rollout_ref.rollout.multi_turn.tool.dict.python_executor=true`。工具应该通过 `tool_config_path` 或 `function_tool_path` 注册。

## Function Tool：最轻量的工具写法

适合无状态工具，例如计算器、天气查询、简单检索。

```python
# tools.py
from verl.tools.function_tool import function_tool

@function_tool("calculator")
def calculator(expression: str) -> str:
    """Evaluate a Python arithmetic expression.

    Args:
        expression: A Python expression such as "(3 + 4) * 5".
    """
    return str(eval(expression, {"__builtins__": {}}, {}))
```

启动时：

```bash
actor_rollout_ref.rollout.multi_turn.function_tool_path=/path/to/tools.py \
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

`@function_tool` 会根据类型标注和 Google 风格 docstring 推断 JSON schema。缺少类型标注或 `Args:` 描述时会在注册时失败。

## BaseTool：有状态工具写法

适合沙箱、浏览器、数据库会话、需要 create/release 生命周期的工具。结构通常是：

```yaml
# tool_config.yaml
tools:
  - class_name: my_project.tools.SandboxTool
    config:
      timeout: 10
    tool_schema:
      type: function
      function:
        name: run_python
        description: Run Python code in a sandbox.
        parameters:
          type: object
          properties:
            code:
              type: string
          required: [code]
```

启动时：

```bash
actor_rollout_ref.rollout.multi_turn.tool_config_path=/path/to/tool_config.yaml
```

## 数据要求

Tool Agent Loop 通常需要 parquet 中包含：

```python
{
  "data_source": "openai/gsm8k",
  "agent_name": "tool_agent",
  "prompt": [
    {"role": "system", "content": "You can use tools."},
    {"role": "user", "content": "..."}
  ],
  "reward_model": {"style": "rule", "ground_truth": "42"},
  "extra_info": {
    "tools_kwargs": {
      "calculator": {"create_kwargs": {}, "execute_kwargs": {}}
    }
  }
}
```

其中 `agent_name` 用于让 rollout 在不同样本上选择不同 agent。只做单一工具训练时，也可以用 `actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent` 作为默认值。

## 调试建议

- 先把 `trainer.logger=console`，确认 rollout 能生成工具调用标签。
- 打开 trace：`actor_rollout_ref.rollout.trace.backend=mlflow`，配合 MLflow 查看每轮对话。
- 如果日志出现 `Failed to decode tool call`，通常是模型没有按工具格式输出，不一定是训练异常。
- 控制工具返回长度：`multi_turn.max_tool_response_length` 和 `tool_response_truncate_side`。

## 下一步

- [07-2-multi-turn.md](07-2-multi-turn.md) - 多轮 tokenization 与轮数控制
- [07-3-agent-loop.md](07-3-agent-loop.md) - 自定义 AgentLoop

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 06-2 - 模型规模与调参](../06-sft-and-tuning/06-2-model-tuning.md) · [目录](../../README.md#catalog) · [07-2 - 多轮对话 RL 训练 →](07-2-multi-turn.md)
<!-- NAV_BOTTOM_END -->
