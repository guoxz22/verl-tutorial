# 12-3 - 自定义 Reward Manager

<!-- NAV_START -->
> 导航： [上一篇：12-2 - 自定义 Policy Loss](12-2-policy-loss.md) | [返回目录](../../README.md#完整目录) | [下一篇：12-4 - 完整算法实现示例](12-4-full-example.md)
<!-- NAV_END -->

Reward Manager 控制“如何把一批 rollout 样本变成 reward tensor”。v0.8.0 的注册器是：

```python
from verl.workers.reward_manager import register
```

不是旧教程里的 `register_reward_manager`。

## 官方抽象类签名

```python
class AbstractRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **kwargs): ...
    def __call__(self, data, return_dict=False): ...
```

`__call__` 应返回：

- `torch.Tensor`：形状通常与 `data.batch["responses"]` 一致，只在需要给 reward 的 token 位置非零。
- 或 `dict`：包含 `reward_tensor` 和 `reward_extra_info`。

## 最小模板

```python
# custom_reward_manager.py
from collections import defaultdict
from typing import Any

import torch
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("my_custom_rm")
class MyCustomRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            item = data[i]
            response_ids = item.batch["responses"]
            attention_mask = item.batch["attention_mask"]
            prompt_len = item.batch["prompts"].shape[-1]
            valid_response_len = attention_mask[prompt_len:].sum()
            valid_response_ids = response_ids[:valid_response_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            score = 1.0 if str(ground_truth) in response_str else 0.0
            reward_tensor[i, valid_response_len - 1] = score
            reward_extra_info["exact_match"].append(score)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
```

## 配置使用

```bash
python -m verl.trainer.main_ppo \
  reward.reward_manager.name=my_custom_rm
```

若 manager 放在独立文件里，需要保证模块被 import。更稳妥的方式是做成 package，在 `__init__.py` 里 import 注册类。

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
reward.reward_manager.name=prime
```

具体可看 `verl/workers/reward_manager/__init__.py` 和对应文件。

## 常见错误

| 错误 | 处理 |
| --- | --- |
| `Unknown reward manager` | 注册模块没被 import，或名字写错 |
| reward 全 0 | 检查 `reward_model.ground_truth` 是否存在 |
| shape 不匹配 | reward tensor 应与 `responses` 对齐 |
| 多进程下状态丢失 | 不要依赖进程内全局变量保存跨 batch 状态 |

---

<!-- NAV_BOTTOM_START -->
> 导航： [上一篇：12-2 - 自定义 Policy Loss](12-2-policy-loss.md) | [返回目录](../../README.md#完整目录) | [下一篇：12-4 - 完整算法实现示例](12-4-full-example.md)
<!-- NAV_BOTTOM_END -->
