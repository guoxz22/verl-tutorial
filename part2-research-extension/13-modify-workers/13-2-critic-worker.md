# 13-2 - 修改 Critic Worker

<!-- NAV_START -->
> 阅读： [← 13-1 - 修改 Actor Worker](13-1-actor-worker.md) · [目录](../../README.md#catalog) · [13-3 - 修改 Rollout Worker →](13-3-rollout-worker.md)
<!-- NAV_END -->

Critic 负责估计 value，并在 PPO/GAE 这类 actor-critic 算法中提供 baseline。GRPO、RLOO、ReMax 等 critic-free 方法通常不需要改 Critic。

## v0.8.0 中 Critic 在哪里

| 文件 | 作用 |
| --- | --- |
| `verl/trainer/ppo/ray_trainer.py` | 调用 `_compute_values`、`_update_critic` |
| `verl/trainer/main_ppo.py` | role 到 `TrainingWorker` 的映射 |
| `verl/workers/engine_workers.py` | 统一 worker 路径 |
| `verl/workers/config/critic.py` | critic 配置 dataclass |
| `verl/workers/utils/losses.py` | `value_loss` 实现 |

旧版教程里的 `from verl.workers.fsdp_workers import CriticWorker` 不适用于 v0.8.0。

## 先通过配置调 Critic

```bash
critic.optim.lr=1e-5 \
critic.ppo_micro_batch_size_per_gpu=4 \
critic.ppo_max_token_len_per_gpu=32768 \
critic.fsdp.param_offload=False \
critic.fsdp.optimizer_offload=False
```

PPO 中也要设置 warmup：

```bash
trainer.critic_warmup=10
```

GRPO/RLOO/ReMax 通常不训练 critic：

```bash
trainer.critic_warmup=0
```

## 想改 Value Loss 怎么办

优先阅读 `verl/workers/utils/losses.py` 中的 `value_loss`。如果只是增加一种 value clipping、Huber loss 或额外 metrics，建议：

1. 先在本地 fork 中改 `value_loss`。
2. 保持输入输出字段不变。
3. 输出新的 `critic/*` metrics。
4. 只跑 PPO 小模型 smoke test。

伪代码：

```python
def my_value_loss(config, model_output, data, dp_group=None):
    # 参考官方 value_loss 的 mask、clip、聚合方式
    values = model_output["values"]
    returns = data["returns"]
    response_mask = data["response_mask"]
    ...
```

## 什么时候不需要 Critic

| 算法 | 是否需要 Critic |
| --- | --- |
| PPO / GAE | 通常需要 |
| GRPO | 不需要 |
| RLOO | 不需要 |
| ReMax | 不需要 |
| REINFORCE++ | 不需要 |
| GPG | 不需要 |

## 风险提示

Critic 改动容易影响：

- value/returns shape；
- mask 聚合；
- actor advantage 的尺度；
- checkpoint 中 critic 状态；
- 多机训练时的 DP 聚合。

因此，除非论文明确需要新的 value objective，否则优先把研究点放在 advantage 或 policy loss。

## 下一步

- [13-3-rollout-worker.md](13-3-rollout-worker.md)
- [12-1-advantage.md](../12-custom-algorithm/12-1-advantage.md)

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 13-1 - 修改 Actor Worker](13-1-actor-worker.md) · [目录](../../README.md#catalog) · [13-3 - 修改 Rollout Worker →](13-3-rollout-worker.md)
<!-- NAV_BOTTOM_END -->
