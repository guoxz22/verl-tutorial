# 12-2 - 自定义 Policy Loss

本章介绍如何实现自定义 Policy Loss。

## Policy Loss 基础

Policy Loss 决定了如何根据 Advantage 更新策略：

```
Loss = -E[A(s,a) * log π(a|s)]
```

verl 内置了多种 Policy Loss：

- 标准 PPO Clip Loss
- KL 惩罚 Loss（GRPO 使用）
- 熵正则化 Loss

## 实现自定义 Policy Loss

### 基本模板

```python
# custom_policy_loss.py
import torch
from verl.trainer.ppo.core_algos import register_policy_loss

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
        old_log_prob: 旧策略 log prob [batch_size, seq_len]
        log_prob: 当前策略 log prob [batch_size, seq_len]
        advantages: Advantage 值 [batch_size, seq_len]
        response_mask: Response 掩码 [batch_size, seq_len]
        loss_agg_mode: 聚合模式 ('token-mean', 'seq-mean', etc.)
        config: ActorConfig 配置
        rollout_log_probs: 可选的 rollout log probs

    Returns:
        loss: torch.Tensor 标量
    """
    # 计算 importance ratio
    ratio = torch.exp(log_prob - old_log_prob)

    # 自定义 loss 计算
    # 示例：简单的 policy gradient
    loss = -ratio * advantages

    # 应用掩码并聚合
    if loss_agg_mode == 'token-mean':
        loss = (loss * response_mask).sum() / response_mask.sum()
    elif loss_agg_mode == 'seq-mean':
        seq_loss = (loss * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)
        loss = seq_loss.mean()
    else:
        loss = loss.mean()

    return loss
```

### PPO Clip Loss 实现（参考）

```python
@register_policy_loss("ppo_clip")
def ppo_clip_loss(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode, config, rollout_log_probs=None
):
    """PPO Clip Loss"""
    ratio = torch.exp(log_prob - old_log_prob)

    # PPO clipping
    clip_ratio = config.clip_ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    # 取较小值
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    loss = -torch.min(surr1, surr2)

    # 聚合
    if loss_agg_mode == 'token-mean':
        loss = (loss * response_mask).sum() / response_mask.sum()
    else:
        loss = (loss * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)
        loss = loss.mean()

    return loss
```

### KL 惩罚 Loss（参考）

```python
@register_policy_loss("kl_loss")
def kl_penalty_loss(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode, config, rollout_log_probs=None
):
    """带 KL 惩罚的 Loss"""
    # 标准 policy loss
    ratio = torch.exp(log_prob - old_log_prob)
    policy_loss = -ratio * advantages

    # KL 散度
    # 需要 reference policy 的 log prob
    ref_log_prob = rollout_log_probs  # 假设传入
    kl = old_log_prob - ref_log_prob  # 近似 KL

    if config.kl_loss_type == 'low_var_kl':
        kl = (torch.exp(kl) - 1 - kl)  # 低方差 KL

    # 组合 loss
    kl_coef = config.kl_loss_coef
    total_loss = policy_loss + kl_coef * kl

    # 聚合
    if loss_agg_mode == 'token-mean':
        total_loss = (total_loss * response_mask).sum() / response_mask.sum()
    else:
        total_loss = (total_loss * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)
        total_loss = total_loss.mean()

    return total_loss
```

## 配置使用

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.loss_type=my_custom_loss \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    ...
```

## 添加配置参数

如果需要新的配置参数：

1. 在 `verl/workers/config.py` 的 `ActorConfig` 中添加：

```python
@dataclass
class ActorConfig:
    # 现有参数...

    # 自定义参数
    my_custom_param: float = 0.1
```

2. 在 loss 函数中使用：

```python
@register_policy_loss("my_custom_loss")
def my_custom_loss(..., config, ...):
    my_param = config.my_custom_param
    # 使用 my_param
```

3. 配置：

```bash
actor_rollout_ref.actor.my_custom_param=0.5
```

## 下一步

- [12-3-reward-manager.md](12-3-reward-manager.md) - 自定义 Reward Manager
