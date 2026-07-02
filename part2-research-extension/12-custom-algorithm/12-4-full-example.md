# 12-4 - 完整算法实现示例

本章示例实现一个极简 Contrastive RL 变体，用来串联三个扩展点：advantage、policy loss、reward manager。它不是新论文算法，只是展示 v0.8.0 的正确接线方式。

## 目录结构

```text
contrastive_rl/
├── __init__.py
├── advantage.py
├── loss.py
└── reward_manager.py
```

`__init__.py` 必须 import 三个模块，让注册器执行：

```python
from .advantage import compute_contrastive_advantage
from .loss import compute_contrastive_loss
from .reward_manager import ContrastiveRewardManager
```

## 1. Advantage

```python
# contrastive_rl/advantage.py
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("contrastive_adv")
def compute_contrastive_advantage(data, gamma=1.0, lam=1.0, **kwargs):
    # 教学示例：真实实现应参考 core_algos.py 中 GRPO/GPG 的 DataProto 字段处理
    rewards = data.batch["token_level_scores"].sum(dim=-1)
    baseline = rewards.mean()
    advantages = rewards - baseline
    data.batch["advantages"] = advantages.unsqueeze(-1).expand_as(data.batch["responses"]).float()
    data.batch["returns"] = data.batch["advantages"]
    return data
```

## 2. Policy Loss

```python
# contrastive_rl/loss.py
import torch
from verl.trainer.ppo.core_algos import register_policy_loss
from verl.utils.torch_functional import masked_mean

@register_policy_loss("contrastive_loss")
def compute_contrastive_loss(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config=None, rollout_log_probs=None):
    ratio = torch.exp(log_prob - old_log_prob)
    margin = getattr(getattr(config, "policy_loss", None), "margin", 0.0) if config is not None else 0.0
    loss_mat = -ratio * (advantages - margin)
    loss = masked_mean(loss_mat, response_mask)
    return loss, {"actor/contrastive_loss": loss.detach().item()}
```

## 3. Reward Manager

```python
# contrastive_rl/reward_manager.py
from collections import defaultdict
import torch
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("contrastive_rm")
class ContrastiveRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extra = defaultdict(list)
        for i in range(len(data)):
            item = data[i]
            valid_len = item.batch["attention_mask"][item.batch["prompts"].shape[-1]:].sum()
            response = self.tokenizer.decode(item.batch["responses"][:valid_len], skip_special_tokens=True)
            positive = item.non_tensor_batch.get("extra_info", {}).get("positive", "")
            score = 1.0 if positive and positive in response else 0.0
            reward_tensor[i, valid_len - 1] = score
            extra["positive_hit"].append(score)
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
```

## 4. 训练命令

```bash
PYTHONPATH=$PWD:$PYTHONPATH python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=contrastive_adv \
  actor_rollout_ref.actor.policy_loss.loss_mode=contrastive_loss \
  +actor_rollout_ref.actor.policy_loss.margin=0.1 \
  reward.reward_manager.name=contrastive_rm \
  data.train_files=$HOME/data/contrastive/train.parquet \
  data.val_files=$HOME/data/contrastive/test.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  trainer.logger='["console"]' \
  trainer.total_training_steps=1
```

## 5. 验证

- console 中应出现 `actor/contrastive_loss`。
- reward extra info 中应出现 `positive_hit`。
- 如果 `Unknown reward manager` 或 `Unknown policy loss`，说明 package 没有被 import。
- 如果 Hydra 报自定义键不存在，给新增键加 `+`。

## 6. 研究实现建议

教学示例为了短小，省略了很多严谨处理。真实论文复现时应：

1. 参考官方 `core_algos.py` 中最接近的 estimator/loss。
2. 保持 tensor shape 与 mask 完全一致。
3. 给每个自定义分支加 metrics。
4. 先固定一个 batch 做数值检查，再跑分布式。
