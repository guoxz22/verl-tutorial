# 13-3 - 修改 Rollout Worker

Rollout 负责生成 response。v0.8.0 的 rollout 代码已经按后端拆分，不再是旧路径 `verl/workers/rollout/vllm/`。

## 核心文件

| 后端 | 路径 | 主要类 |
| --- | --- | --- |
| HF | `verl/workers/rollout/hf_rollout.py` | `HFRollout` |
| vLLM | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | `ServerAdapter` |
| SGLang | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | `ServerAdapter` |
| TRTLLM | `verl/workers/rollout/trtllm_rollout/trtllm_rollout.py` | `ServerAdapter` |
| Base | `verl/workers/rollout/base.py` | `BaseRollout`, `get_rollout_class` |
| Replica | `verl/workers/rollout/replica.py` | async server replica registry |

## 先改配置，不先改代码

很多 rollout 需求不用改 Worker：

```bash
actor_rollout_ref.rollout.temperature=0.8 \
actor_rollout_ref.rollout.top_p=0.95 \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
+actor_rollout_ref.rollout.engine_kwargs.vllm.some_arg=value
```

只有在你要改变“请求如何构造、权重如何同步、server 如何适配、token 如何回传”时，才考虑改 rollout 代码。

## 自定义 rollout 的推荐方式

1. 继承 `BaseRollout` 或参考目标后端的 `ServerAdapter`。
2. 在独立模块中实现，不直接改官方文件。
3. 确认 `RolloutConfig` 能传入所需参数；新增参数放到 `rollout.custom` 或 `engine_kwargs`。
4. 在小模型上做 `rollout.n=1` smoke test。

伪代码：

```python
from verl.workers.rollout.base import BaseRollout

class MyRollout(BaseRollout):
    def __init__(self, model_path, config, tokenizer, model_hf_config, **kwargs):
        self.config = config
        self.tokenizer = tokenizer

    def generate_sequences(self, prompts, **kwargs):
        # 返回格式必须与 trainer 期望一致；真实实现请参考 vllm/sglang adapter
        raise NotImplementedError
```

## AgentRL 优先改 AgentLoop

如果你的需求是“多轮交互、工具调用、搜索、沙箱”，优先改：

```text
verl/experimental/agent_loop/
verl/tools/
actor_rollout_ref.rollout.multi_turn.*
actor_rollout_ref.rollout.agent.*
```

不要为了加工具调用去改 vLLM/SGLang adapter。

## 权重同步与 server replica

v0.8.0 的 async rollout 会涉及：

- training worker 到 rollout server 的权重更新；
- vLLM/SGLang/TRTLLM 的 replica 管理；
- checkpoint engine，如 `naive`、`nccl`、`nixl`；
- SGLang PD disaggregation 等高级模式。

这些属于高风险修改。建议先阅读：

```text
verl/workers/engine_workers.py
verl/workers/rollout/replica.py
verl/workers/config/rollout.py
```

## 常见错误

| 错误 | 说明 |
| --- | --- |
| `from verl.workers.rollout.vllm...` | 旧路径，v0.8.0 是 `vllm_rollout/` |
| 自定义类没有被调用 | 未接入 rollout registry 或配置仍指向内置后端 |
| 训练 loss 异常 | rollout 返回 token/logprob/mask 与 trainer 期望不一致 |
| 多轮工具错位 | 应改 AgentLoop / tool，而不是底层 rollout adapter |

## 下一步

- [14-1-reproduce-workflow.md](../14-paper-reproduction/14-1-reproduce-workflow.md)
- [07-3-agent-loop.md](../../part1-engineering-training/07-agent-rl/07-3-agent-loop.md)
