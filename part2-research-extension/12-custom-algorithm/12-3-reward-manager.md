# 12-3 - 自定义 Reward Manager

<!-- NAV_START -->
> 阅读： [← 12-2 - 自定义 Policy Loss](12-2-policy-loss.md) · [目录](../../README.md#catalog) · [12-4 - 完整算法实现示例 →](12-4-full-example.md)
<!-- NAV_END -->

Reward Manager 控制“如何把一批 rollout 样本变成 reward tensor”。v0.8.0 的注册器是：

```python
from verl.experimental.reward_loop.reward_manager import register
```

不是旧教程里的 `register_reward_manager`。`verl.workers.reward_manager` 目录仍然存在，用于 legacy manager；但 `reward.reward_manager.source=register` 在 PPO reward loop 中解析的是 `verl.experimental.reward_loop.reward_manager` 注册表。

## 官方抽象类签名

```python
class RewardManagerBase:
    def __init__(self, config, tokenizer, compute_score): ...
    async def run_single(self, data): ...
```

`run_single` 处理单条样本或一条 trajectory，应返回：

- `reward_score`：标量 reward。
- `reward_extra_info`：可选监控字段。

框架会把标量 reward 装配到 token-level reward tensor 中。

## 最小模板

```python
# custom_reward_manager.py
from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

@register("my_custom_rm")
class MyCustomRewardManager(RewardManagerBase):
    async def run_single(self, data: DataProto) -> dict:
        data_item = data[-1]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        score = 1.0 if str(ground_truth) in response_str else 0.0

        return {"reward_score": score, "reward_extra_info": {"exact_match": score}}
```

## 配置使用

```bash
python -m verl.trainer.main_ppo \
  reward.reward_manager.name=my_custom_rm
```

若使用 `source=register`，需要保证注册模块在 trainer 启动前被 import。更稳妥的方式是做成 package，在 `__init__.py` 里 import 注册类。

如果只想从独立文件加载类，也可以走 `importlib`：

```bash
python -m verl.trainer.main_ppo \
  reward.reward_manager.source=importlib \
  reward.reward_manager.module.path=$PWD/custom_reward_manager.py \
  reward.reward_manager.name=MyCustomRewardManager
```

## 何时写 Reward Function，何时写 Reward Manager

| 需求 | 推荐 |
| --- | --- |
| 每条样本独立打分 | `reward.custom_reward_function.path/name` |
| 需要批量处理、缓存、多个 reward 维度 | 自定义 Reward Manager |
| 要执行代码/沙箱且有并发控制 | Reward Manager + sandbox 配置 |
| 要对接 RM 输出并二次解析 | Reward Model + custom reward function/manager |
| 要给 reward_extra_info 做监控 | Reward Manager |

## 内置 manager

常见内置名：

```bash
reward.reward_manager.name=naive
reward.reward_manager.name=dapo
reward.reward_manager.name=gdpo
reward.reward_manager.name=rate_limited
reward.reward_manager.name=remote
```

具体可看 `verl/experimental/reward_loop/reward_manager/__init__.py` 和对应文件。`prime` 在 legacy `verl.workers.reward_manager` 里存在，但不是 v0.8.0 PPO reward loop 注册表的默认内置名。

## 常见错误

| 错误 | 处理 |
| --- | --- |
| `Unknown reward manager` | 注册模块没被 import，或名字写错 |
| reward 全 0 | 检查 `reward_model.ground_truth` 是否存在 |
| shape 不匹配 | reward tensor 应与 `responses` 对齐 |
| 多进程下状态丢失 | 不要依赖进程内全局变量保存跨 batch 状态 |

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 12-2 - 自定义 Policy Loss](12-2-policy-loss.md) · [目录](../../README.md#catalog) · [12-4 - 完整算法实现示例 →](12-4-full-example.md)
<!-- NAV_BOTTOM_END -->
