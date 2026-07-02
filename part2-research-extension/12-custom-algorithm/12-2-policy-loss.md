# 12-2 - 自定义 Policy Loss

Policy loss 决定 actor 如何根据 log probability、advantage 和 mask 更新。v0.8.0 的注册入口是 `register_policy_loss`，配置键是：

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=<name>
```

## 官方函数签名

```python
PolicyLossFn = Callable[
    [old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config, rollout_log_probs],
    tuple[torch.Tensor, dict[str, Any]],
]
```

最小模板：

```python
# my_policy_loss.py
import torch
from verl.trainer.ppo.core_algos import register_policy_loss
from verl.utils.torch_functional import masked_mean

@register_policy_loss("my_loss")
def compute_my_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config=None,
    rollout_log_probs: torch.Tensor | None = None,
):
    ratio = torch.exp(log_prob - old_log_prob)
    loss_mat = -ratio * advantages
    loss = masked_mean(loss_mat, response_mask)
    metrics = {
        "actor/my_pg_loss": loss.detach().item(),
        "actor/my_ratio_mean": masked_mean(ratio, response_mask).detach().item(),
    }
    return loss, metrics
```

> 如果你需要在独立文件中注册 loss，确保训练进程会 import 这个文件。最常见做法是在自定义 package 的 `__init__.py` 中 import，或在启动脚本中通过你自己的入口先 import 再调用 verl trainer。

## 使用配置

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.policy_loss.loss_mode=my_loss
```

新增自定义超参数要加 `+`：

```bash
+actor_rollout_ref.actor.policy_loss.my_clip=0.3
```

在函数里可以通过 `config.policy_loss.my_clip` 读取。

## 官方已注册 loss

```text
vanilla, dppo_tv, dppo_kl, gspo, sapo, gpg,
clip_cov, kl_cov, geo_mean, cispo, bypass_mode
```

其中：

- `vanilla` 是默认 PPO clipped surrogate。
- `gspo`、`sapo`、`cispo` 是 policy loss 变体，不是 advantage estimator。
- `geo_mean` 对应 GMPO。
- `dppo_tv` / `dppo_kl` 对应 DPPO 变体。

## 不要用的旧键

```bash
# 错误 / 旧写法
actor_rollout_ref.actor.loss_type=my_loss

# v0.8.0 写法
actor_rollout_ref.actor.policy_loss.loss_mode=my_loss
```

## 验证你的 loss 是否生效

1. 在 metrics 里输出唯一前缀，例如 `actor/my_pg_loss`。
2. 训练 1 step，检查 console 或 wandb 中是否出现该指标。
3. 临时设置一个明显不同的 loss，确认曲线变化。
4. 再和 `vanilla` 对照，避免只注册未调用。

## 常见错误

| 错误 | 原因 | 解决 |
| --- | --- | --- |
| `Unknown policy loss` | 自定义模块没有被 import | 确保注册代码在 trainer 启动前执行 |
| `Key policy_loss.my_x is not in struct` | 新增 Hydra 键没加 `+` | 用 `+actor_rollout_ref.actor.policy_loss.my_x=...` |
| loss 是 NaN | ratio 爆炸或 mask 错 | clamp ratio，检查 `response_mask` |
| 指标没有出现 | 返回 metrics key 不对或 loss 未调用 | 加唯一 metrics 前缀 |
