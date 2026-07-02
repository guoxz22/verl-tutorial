# 13-1 - 修改 Actor Worker

<!-- NAV_START -->
> 阅读： [← 12-4 - 完整算法实现示例](../12-custom-algorithm/12-4-full-example.md) · [目录](../../README.md#完整目录) · [13-2 - 修改 Critic Worker →](13-2-critic-worker.md)
<!-- NAV_END -->

v0.8.0 的 Actor/Ref/Rollout 逻辑主要集中在 `verl/workers/engine_workers.py`，底层模型实现放在 `verl/workers/engine/<backend>/`。旧版教程里提到的 `fsdp_workers.py`、`megatron_workers.py` 不再是本版主要入口。

## 先问：真的需要改 Worker 吗？

| 需求 | 推荐做法 |
| --- | --- |
| 改 PPO/GSPO/SAPO loss | 自定义 `register_policy_loss` |
| 改 advantage | 自定义 `register_adv_est` |
| 改 KL、clip、batch、offload | 改配置 |
| 加 actor metrics | 优先在 policy loss 或 trainer metrics 中加 |
| 改 forward/backward/权重同步 | 才考虑 Worker/Engine |

Worker 是高风险层，改了以后要同时考虑 Ray、分布式、checkpoint、rollout 权重同步。

## v0.8.0 相关文件

| 文件 | 作用 |
| --- | --- |
| `verl/workers/engine_workers.py` | `ActorRolloutRefWorker` 与统一 TrainingWorker 路径 |
| `verl/workers/engine/base.py` | engine 抽象 |
| `verl/workers/engine/fsdp/` | FSDP/FSDP2 实现 |
| `verl/workers/engine/megatron/` | Megatron 实现 |
| `verl/workers/engine/veomni/` | VeOmni 实现 |
| `verl/workers/config/actor.py` | actor 配置 dataclass |
| `verl/workers/utils/losses.py` | actor/critic loss 工具 |

## Actor Worker 关键方法

`engine_workers.py` 中应重点看：

```text
ActorRolloutRefWorker.compute_log_prob
ActorRolloutRefWorker.update_actor
ActorRolloutRefWorker.generate_sequences
```

这些方法分别对应：算旧策略/参考 logprob、更新 actor、调用 rollout 生成。

## 更推荐的扩展方式：自定义 policy loss

大部分“改 actor update”的论文不需要改 Worker：

```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("my_actor_loss")
def my_actor_loss(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, config=None, rollout_log_probs=None):
    ...
```

配置：

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=my_actor_loss
```

这样不会破坏 worker 的分布式生命周期。

## 必须改 Worker 时的流程

1. 在 v0.8.0 源码中新建分支，不直接在教程里复制旧类名。
2. 找到当前 `ActorRolloutRefWorker.update_actor` 的调用链。
3. 先加 metrics，不改行为，确认自定义代码被调用。
4. 再做最小行为修改。
5. 用 `trainer.total_training_steps=1` 做 smoke test。
6. 再跑 8 卡/多机。

## 不推荐的旧接线

不要假设存在：

```text
verl/workers/fsdp_workers.py
trainer.custom_actor_worker=/path/to/worker.py:ClassName
```

v0.8.0 没有把 `trainer.custom_actor_worker` 作为官方稳定配置入口。需要替换 worker 时，应先阅读 `main_ppo.py` 中 role 到 worker 的映射，再决定是否维护自己的入口脚本。

## 下一步

- [12-2-policy-loss.md](../12-custom-algorithm/12-2-policy-loss.md)
- [13-2-critic-worker.md](13-2-critic-worker.md)

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 12-4 - 完整算法实现示例](../12-custom-algorithm/12-4-full-example.md) · [目录](../../README.md#完整目录) · [13-2 - 修改 Critic Worker →](13-2-critic-worker.md)
<!-- NAV_BOTTOM_END -->
