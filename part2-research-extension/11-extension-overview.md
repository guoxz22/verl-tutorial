# 11 - 扩展机制总览

<!-- NAV_START -->
> 阅读： [← 10-4 - 集群调度](../part1-engineering-training/10-operations/10-4-cluster.md) · [目录](../README.md#完整目录) · [12-1 - 扩展 Advantage Estimator →](12-custom-algorithm/12-1-advantage.md)
<!-- NAV_END -->

verl 的扩展点很多，但对研究者最重要的是三类：算法数学、奖励系统、Worker/后端。先分清扩展层次，能少走很多弯路。

## 扩展点地图

| 扩展目标 | 首选位置 | 配置入口 |
| --- | --- | --- |
| 改 advantage 估计 | `verl.trainer.ppo.core_algos.register_adv_est` | `algorithm.adv_estimator=<name>` |
| 改 policy loss | `register_policy_loss` | `actor_rollout_ref.actor.policy_loss.loss_mode=<name>` |
| 改 reward 函数 | 独立 Python 文件 | `reward.custom_reward_function.path/name` |
| 改 reward manager | `verl.workers.reward_manager.register` | `reward.reward_manager.name=<name>` |
| 改 rollout 行为 | `verl/workers/rollout/*` 或 AgentLoop | `actor_rollout_ref.rollout.*` |
| 改 actor/critic 执行 | `verl/workers/engine_workers.py` 和 engine config | `model_engine=*`、`actor_rollout_ref.actor.*` |

## 注册器模式

### Advantage Estimator

```python
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("my_adv")
def compute_my_advantage(data, gamma=1.0, lam=1.0, **kwargs):
    # 返回包含 advantages/returns 的 DataProto 或张量结构；参考 core_algos.py 中 GAE/GRPO 实现
    return data
```

使用：

```bash
algorithm.adv_estimator=my_adv
```

### Policy Loss

```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_loss")
def my_loss(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config=None, rollout_log_probs=None):
    # 返回 (loss, metrics)
    ...
```

使用：

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=my_loss
```

### Reward Manager

```python
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("my_rm")
class MyRewardManager(AbstractRewardManager):
    ...
```

使用：

```bash
reward.reward_manager.name=my_rm
```

## 何时不该改源码

| 需求 | 更轻的做法 |
| --- | --- |
| 换数学题打分 | 写 `reward.custom_reward_function.path` |
| 换数据字段 | 改预处理脚本和 `data.reward_fn_key` |
| 换采样温度/长度 | 改 `actor_rollout_ref.rollout.*` |
| 加一个无状态工具 | 用 `@function_tool` |
| 调显存 | 改 batch/offload/TP，不要改 Worker |

## 扩展开发流程

1. 先找到 v0.8.0 官方已有实现：`core_algos.py`、`reward_manager/`、`workers/rollout/`。
2. 写最小扩展，不要一次改 trainer、worker、reward 三层。
3. 用小模型、小数据、`trainer.total_training_steps=1` 做 smoke test。
4. 输出自定义 metrics，确认新逻辑真的被调用。
5. 再扩大 batch 和节点。

## 新增配置键的规则

Hydra 结构化配置中，不存在的键需要加 `+`：

```bash
+algorithm.my_coef=0.5
+actor_rollout_ref.actor.policy_loss.my_tau=1.2
```

如果不加 `+`，很可能报 `Key is not in struct`。

## 下一步

- [12-1-advantage.md](12-custom-algorithm/12-1-advantage.md)
- [12-2-policy-loss.md](12-custom-algorithm/12-2-policy-loss.md)
- [12-3-reward-manager.md](12-custom-algorithm/12-3-reward-manager.md)

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 10-4 - 集群调度](../part1-engineering-training/10-operations/10-4-cluster.md) · [目录](../README.md#完整目录) · [12-1 - 扩展 Advantage Estimator →](12-custom-algorithm/12-1-advantage.md)
<!-- NAV_BOTTOM_END -->
