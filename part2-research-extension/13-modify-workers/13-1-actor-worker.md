# 13-1 - 修改 Actor Worker

本章介绍如何修改 Actor Worker 以实现自定义训练逻辑。

## Actor Worker 基础

Actor Worker 负责：
- 加载和管理 Actor 模型
- 执行前向/后向传播
- 计算 Policy Loss
- 更新模型参数

## 核心文件

| 后端 | 文件位置 |
|------|----------|
| FSDP | `verl/workers/fsdp_workers.py` |
| Megatron | `verl/workers/megatron_workers.py` |
| 新引擎 | `verl/workers/engine_workers.py` |

## 主要方法

```python
class ActorRolloutRefWorker(Worker):
    def __init__(self, config, role):
        """初始化"""
        pass

    def load_model(self):
        """加载模型"""
        pass

    def forward(self, data):
        """前向传播"""
        pass

    def compute_log_prob(self, data):
        """计算 log probability"""
        pass

    def update_policy(self, data):
        """更新策略"""
        pass
```

## 修改示例：添加自定义正则化

```python
# custom_actor_worker.py
from verl.workers.fsdp_workers import ActorRolloutRefWorker

class CustomActorWorker(ActorRolloutRefWorker):
    """带自定义正则化的 Actor Worker"""

    def update_policy(self, data):
        """重写 policy 更新，添加自定义正则化"""
        # 调用原始逻辑
        metrics = super().update_policy(data)

        # 添加自定义正则化
        if hasattr(self.config, 'custom_l2_reg'):
            l2_reg = 0.0
            for param in self.actor_model.parameters():
                l2_reg += torch.norm(param, p=2)

            l2_loss = self.config.custom_l2_reg * l2_reg
            l2_loss.backward()

            metrics['custom/l2_reg'] = l2_reg.item()

        return metrics
```

## 修改示例：自定义梯度处理

```python
class CustomActorWorker(ActorRolloutRefWorker):
    """带自定义梯度处理的 Actor Worker"""

    def update_policy(self, data):
        metrics = super().update_policy(data)

        # 梯度裁剪后处理
        if hasattr(self.config, 'custom_grad_norm_target'):
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_model.parameters(),
                self.config.grad_clip
            )

            # 自定义梯度缩放
            if total_norm > self.config.custom_grad_norm_target:
                scale = self.config.custom_grad_norm_target / (total_norm + 1e-8)
                for param in self.actor_model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(scale)

            metrics['custom/grad_norm'] = total_norm.item()

        return metrics
```

## 注册自定义 Worker

```python
# 在配置中使用
# trainer.custom_actor_worker=/path/to/custom_actor_worker.py:CustomActorWorker
```

## 下一步

- [13-2-critic-worker.md](13-2-critic-worker.md) - 修改 Critic Worker
