# 13-2 - 修改 Critic Worker

本章介绍如何修改 Critic Worker。

## Critic Worker 基础

Critic Worker 负责：
- 加载和管理 Critic 模型
- 估计 Value Function
- 计算 Value Loss
- 更新 Critic 参数

## 修改示例：自定义 Value Loss

```python
# custom_critic_worker.py
from verl.workers.fsdp_workers import CriticWorker

class CustomCriticWorker(CriticWorker):
    """带自定义 Value Loss 的 Critic Worker"""

    def compute_value_loss(self, values, returns, mask):
        """自定义 Value Loss 计算"""
        # 标准 MSE Loss
        mse_loss = ((values - returns) ** 2) * mask

        # 添加 Huber Loss 选项
        if hasattr(self.config, 'use_huber_loss') and self.config.use_huber_loss:
            delta = self.config.huber_delta
            error = values - returns
            huber_loss = torch.where(
                error.abs() < delta,
                0.5 * error ** 2,
                delta * (error.abs() - 0.5 * delta)
            )
            loss = (huber_loss * mask).sum() / mask.sum()
        else:
            loss = mse_loss.sum() / mask.sum()

        return loss
```

## 修改示例：Value Function 正则化

```python
class CustomCriticWorker(CriticWorker):
    """带正则化的 Critic Worker"""

    def update_critic(self, data):
        metrics = super().update_critic(data)

        # 添加 Value 范数约束
        if hasattr(self.config, 'value_norm_reg'):
            values = data.batch['values']
            value_std = values.std()

            if value_std > self.config.value_norm_threshold:
                reg_loss = self.config.value_norm_reg * (value_std - 1.0) ** 2
                reg_loss.backward()
                metrics['critic/value_norm_reg'] = reg_loss.item()

        return metrics
```

## 不使用 Critic

对于 GRPO 等算法，可以禁用 Critic：

```bash
trainer.critic_warmup=0
```

## 下一步

- [13-3-rollout-worker.md](13-3-rollout-worker.md) - 修改 Rollout Worker
