# 12-1 - 扩展 Advantage Estimator

本章详细介绍如何实现自定义 Advantage Estimator。

## Advantage Estimator 基础

Advantage 用于衡量某个动作相对于平均水平的优势程度：

```
A(s, a) = Q(s, a) - V(s)
```

verl 内置了多种 Advantage Estimator：

- `gae`: Generalized Advantage Estimation (PPO)
- `grpo`: Group-wise PPO
- `reinforce_plus_plus`: REINFORCE++
- `rloo`: Leave-One-Out baseline

## 实现自定义 Advantage Estimator

### 基本模板

```python
# custom_advantage.py
import torch
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("my_custom_adv")
def my_custom_advantage(data, config):
    """
    自定义 Advantage 计算

    Args:
        data: DataProto 对象
            - data.batch['rewards']: 奖励张量 [batch_size, seq_len]
            - data.batch['values']: Critic 估计值（可选）
            - data.batch['response_mask']: Response 掩码
        config: AlgoConfig 配置对象
            - config.gamma: 折扣因子
            - config.lam: GAE lambda

    Returns:
        advantages: torch.Tensor [batch_size, seq_len]
    """
    # 获取奖励
    rewards = data.batch['rewards']
    response_mask = data.batch['response_mask']

    # 自定义计算逻辑
    # 示例：简单的奖励归一化
    advantages = rewards - rewards.mean(dim=-1, keepdim=True)
    advantages = advantages / (rewards.std(dim=-1, keepdim=True) + 1e-8)

    # 应用掩码
    advantages = advantages * response_mask

    return advantages
```

### GAE 实现（参考）

```python
@register_adv_est("gae")
def compute_gae_advantage(data, config):
    """GAE 实现"""
    rewards = data.batch['rewards']
    values = data.batch['values']
    response_mask = data.batch['response_mask']

    gamma = config.gamma
    lam = config.lam

    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(seq_len)):
        if t < seq_len - 1:
            next_value = values[:, t + 1]
        else:
            next_value = 0

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = last_gae = delta + gamma * lam * last_gae

    # 应用掩码
    advantages = advantages * response_mask

    return advantages
```

### GRPO 实现（参考）

```python
@register_adv_est("grpo")
def compute_grpo_advantage(data, config):
    """GRPO 实现：使用 Group 内的均值和方差"""
    rewards = data.batch['rewards']
    response_mask = data.batch['response_mask']

    # 假设数据已按 group 排列
    # 每个 prompt 有 n 个 response
    n = config.rollout_n  # 每个 prompt 的 response 数

    # Reshape 为 [num_prompts, n, seq_len]
    batch_size = rewards.shape[0]
    num_prompts = batch_size // n
    rewards_grouped = rewards.view(num_prompts, n, -1)

    # 计算 group 均值和方差
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)
    group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

    # 归一化
    advantages = (rewards_grouped - group_mean) / group_std

    # Reshape 回原形状
    advantages = advantages.view(batch_size, -1)

    # 应用掩码
    advantages = advantages * response_mask

    return advantages
```

## 配置使用

### 方式一：修改源码后直接使用

```bash
# 修改 verl/trainer/ppo/core_algos.py 添加注册后
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=my_custom_adv \
    ...
```

### 方式二：外部模块注入

```python
# external/custom_adv.py
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("external_adv")
def external_advantage(data, config):
    # 实现
    pass
```

```bash
# 加载外部模块
export VERL_USE_EXTERNAL_MODULES=/path/to/external/custom_adv.py

# 使用
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=external_adv \
    ...
```

## 调试技巧

### 验证 Advantage 计算

```python
# 在自定义函数中添加调试代码
@register_adv_est("debug_adv")
def debug_advantage(data, config):
    advantages = ...  # 计算逻辑

    # 打印统计信息
    print(f"Advantages mean: {advantages.mean()}")
    print(f"Advantages std: {advantages.std()}")
    print(f"Advantages min: {advantages.min()}")
    print(f"Advantages max: {advantages.max()}")

    return advantages
```

### 单元测试

```python
import torch
from verl import DataProto
from tensordict import TensorDict

def test_custom_advantage():
    # 创建测试数据
    batch = TensorDict({
        'rewards': torch.randn(4, 10),
        'response_mask': torch.ones(4, 10),
    })
    data = DataProto(batch=batch)

    # 测试
    from custom_advantage import my_custom_advantage
    from verl.trainer.config import AlgoConfig

    config = AlgoConfig()
    advantages = my_custom_advantage(data, config)

    assert advantages.shape == (4, 10)
    print("Test passed!")

if __name__ == "__main__":
    test_custom_advantage()
```

## 下一步

- [12-2-policy-loss.md](12-2-policy-loss.md) - 自定义 Policy Loss
