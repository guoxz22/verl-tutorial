# 11 - 扩展机制总览

本章介绍 verl 的扩展机制，帮助研究员理解如何自定义算法。

## 扩展点总览

verl 提供了多个扩展点，允许研究员实现自定义算法：

```
┌─────────────────────────────────────────────────────────────┐
│                    verl 扩展点                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐         │
│  │ Advantage Estimator │  │   Policy Loss       │         │
│  │    (算法核心)        │  │   (损失函数)        │         │
│  │                     │  │                     │         │
│  │ @register_adv_est   │  │ @register_policy_loss│        │
│  └─────────────────────┘  └─────────────────────┘         │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐         │
│  │   Reward Manager    │  │   Worker            │         │
│  │   (奖励计算)        │  │   (训练逻辑)        │         │
│  │                     │  │                     │         │
│  │ AbstractRewardMgr   │  │ ActorWorker         │         │
│  └─────────────────────┘  │ CriticWorker        │         │
│                           │ RolloutWorker       │         │
│                           └─────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 注册器模式

verl 使用注册器模式管理扩展：

### Advantage Estimator 注册

```python
from verl.trainer.ppo.core_algos import register_adv_est, get_adv_estimator_fn

# 注册自定义 Advantage Estimator
@register_adv_est("my_custom_adv")
def my_custom_advantage(data, config):
    """
    自定义 Advantage 计算逻辑

    Args:
        data: DataProto，包含 rewards, values 等
        config: 配置对象

    Returns:
        torch.Tensor: Advantage 值
    """
    rewards = data.batch['rewards']
    # ... 自定义计算逻辑
    advantages = rewards - rewards.mean()  # 简单示例
    return advantages

# 使用
# algorithm.adv_estimator=my_custom_adv
```

### Policy Loss 注册

```python
from verl.trainer.ppo.core_algos import register_policy_loss, PolicyLossFn

@register_policy_loss("my_custom_loss")
def my_custom_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str,
    config,
    rollout_log_probs=None,
) -> torch.Tensor:
    """
    自定义 Policy Loss

    Args:
        old_log_prob: 旧策略的 log probability
        log_prob: 当前策略的 log probability
        advantages: Advantage 值
        response_mask: Response 部分的掩码
        loss_agg_mode: 聚合模式
        config: 配置
        rollout_log_probs: Rollout 时的 log probs（可选）

    Returns:
        torch.Tensor: Loss 值
    """
    ratio = torch.exp(log_prob - old_log_prob)
    loss = -ratio * advantages
    loss = (loss * response_mask).sum() / response_mask.sum()
    return loss

# 使用
# actor_rollout_ref.actor.loss_type=my_custom_loss
```

## 核心文件位置

| 组件 | 文件位置 |
|------|----------|
| Advantage Estimator | `verl/trainer/ppo/core_algos.py` |
| Policy Loss | `verl/trainer/ppo/core_algos.py` |
| Reward Manager | `verl/workers/reward_manager/` |
| Actor Worker | `verl/workers/fsdp_workers.py` 或 `megatron_workers.py` |
| Critic Worker | `verl/workers/fsdp_workers.py` 或 `megatron_workers.py` |
| Rollout | `verl/workers/rollout/` |

## 扩展工作流

### 方式一：在 verl 源码中添加

1. 在 `core_algos.py` 中注册新组件
2. 在配置中添加新参数
3. 通过配置参数启用

### 方式二：外部模块注入

1. 创建外部 Python 文件
2. 导入并注册组件
3. 通过环境变量或配置加载

```python
# external_algorithm.py
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("external_adv")
def external_advantage(data, config):
    # 自定义实现
    pass
```

```bash
# 加载外部模块
export VERL_USE_EXTERNAL_MODULES=/path/to/external_algorithm.py
```

## 下一步

- [12-1-advantage.md](12-custom-algorithm/12-1-advantage.md) - 扩展 Advantage Estimator
- [12-2-policy-loss.md](12-custom-algorithm/12-2-policy-loss.md) - 自定义 Policy Loss
- [12-3-reward-manager.md](12-custom-algorithm/12-3-reward-manager.md) - 自定义 Reward Manager
