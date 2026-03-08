# 12-4 - 完整算法实现示例

本章通过一个完整示例展示如何实现自定义 RL 算法。

## 目标：实现一个简单的 Contrastive RL 算法

### 算法概述

- 使用正负样本对比学习
- Advantage 基于正负样本的奖励差异
- Loss 使用 contrastive 形式

## 步骤 1：实现 Advantage Estimator

```python
# contrastive_rl/advantage.py
import torch
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("contrastive_adv")
def contrastive_advantage(data, config):
    """
    Contrastive Advantage:
    - 正样本（高奖励）获得正 Advantage
    - 负样本（低奖励）获得负 Advantage
    """
    rewards = data.batch['rewards']
    response_mask = data.batch['response_mask']

    # 获取正负样本阈值（可配置）
    threshold = getattr(config, 'contrastive_threshold', 0.5)

    # 二值化奖励
    binary_rewards = (rewards > threshold).float()

    # Contrastive Advantage: 正样本 +1，负样本 -1
    advantages = 2 * binary_rewards - 1

    # 归一化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 应用掩码
    advantages = advantages * response_mask

    return advantages
```

## 步骤 2：实现 Policy Loss

```python
# contrastive_rl/loss.py
import torch
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("contrastive_loss")
def contrastive_policy_loss(
    old_log_prob, log_prob, advantages, response_mask,
    loss_agg_mode, config, rollout_log_probs=None
):
    """
    Contrastive Policy Loss:
    - 最大化正样本的概率
    - 最小化负样本的概率
    """
    # 计算 ratio
    ratio = torch.exp(log_prob - old_log_prob)

    # Contrastive weighting
    # 正样本（advantages > 0）：增大 log prob
    # 负样本（advantages < 0）：减小 log prob
    weight = torch.sign(advantages)
    loss = -weight * log_prob

    # 可选：加入 margin
    margin = getattr(config, 'contrastive_margin', 0.1)
    if margin > 0:
        # Margin-based contrastive loss
        pos_mask = advantages > 0
        neg_mask = advantages < 0

        pos_loss = -log_prob * pos_mask.float()
        neg_loss = torch.relu(log_prob + margin) * neg_mask.float()

        loss = pos_loss + neg_loss

    # 聚合
    if loss_agg_mode == 'token-mean':
        loss = (loss * response_mask).sum() / response_mask.sum()
    else:
        loss = (loss * response_mask).sum(dim=-1) / response_mask.sum(dim=-1)
        loss = loss.mean()

    return loss
```

## 步骤 3：实现 Reward Manager

```python
# contrastive_rl/reward.py
import torch
from verl.workers.reward_manager import AbstractRewardManager, register_reward_manager
from verl import DataProto

@register_reward_manager("contrastive_rm")
class ContrastiveRewardManager(AbstractRewardManager):
    """
    Contrastive Reward Manager
    生成正负样本对并计算奖励
    """

    def __init__(self, config):
        super().__init__(config)
        self.positive_threshold = config.get('positive_threshold', 0.8)

    def __call__(self, data: DataProto) -> DataProto:
        # 获取模型生成的 response
        input_ids = data.batch['input_ids']
        responses = self._decode_responses(input_ids)

        # 获取 ground truth
        ground_truths = data.non_tensor_batch.get('ground_truth', [])

        # 计算奖励
        rewards = []
        for response, gt in zip(responses, ground_truths):
            # 计算匹配分数
            score = self._compute_match_score(response, gt)

            # 归一化到 [0, 1]
            reward = float(score >= self.positive_threshold)
            rewards.append(reward)

        data.batch['rewards'] = torch.tensor(rewards, dtype=torch.float32)
        return data

    def _compute_match_score(self, response, gt):
        """计算匹配分数"""
        # 简单示例：字符串包含
        return 1.0 if gt in response else 0.0
```

## 步骤 4：配置文件

```yaml
# contrastive_rl/config.yaml
algorithm:
  adv_estimator: contrastive_adv
  contrastive_threshold: 0.5

actor_rollout_ref:
  actor:
    loss_type: contrastive_loss
    contrastive_margin: 0.1

reward:
  reward_manager: contrastive_rm
  positive_threshold: 0.8
```

## 步骤 5：入口脚本

```python
# contrastive_rl/__init__.py
from .advantage import contrastive_advantage
from .loss import contrastive_policy_loss
from .reward import ContrastiveRewardManager

# 注册时会自动执行
__all__ = ['contrastive_advantage', 'contrastive_policy_loss', 'ContrastiveRewardManager']
```

```bash
# run_contrastive.sh
#!/bin/bash

# 加载自定义模块
export VERL_USE_EXTERNAL_MODULES=/path/to/contrastive_rl

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=contrastive_adv \
    algorithm.contrastive_threshold=0.5 \
    actor_rollout_ref.actor.loss_type=contrastive_loss \
    actor_rollout_ref.actor.contrastive_margin=0.1 \
    reward.reward_manager=contrastive_rm \
    reward.positive_threshold=0.8 \
    data.train_files=./data/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10
```

## 项目结构

```
contrastive_rl/
├── __init__.py
├── advantage.py      # Advantage Estimator
├── loss.py           # Policy Loss
├── reward.py         # Reward Manager
├── config.yaml       # 配置
└── run_contrastive.sh  # 运行脚本
```

## 下一步

- [13-modify-workers](../13-modify-workers/) - 修改 Worker
- [14-paper-reproduction](../14-paper-reproduction/) - 论文复现
