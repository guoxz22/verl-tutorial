# 05-5 - 其他算法

<!-- NAV_START -->
> 导航： [上一篇：05-4 - RLOO / ReMax 详解](05-4-rloo-remax.md) | [返回目录](../../README.md#完整目录) | [下一篇：06-1 - SFT 基础训练](../06-sft-and-tuning/06-1-sft-basics.md)
<!-- NAV_END -->

本章不把所有算法写成论文综述，而是帮助读者在 v0.8.0 的官方示例目录中找到正确入口，并区分每个算法到底改的是 `adv_estimator` 还是 `policy_loss.loss_mode`。

## v0.8.0 算法目录

| 目录 | 算法 | 主要开关 |
| --- | --- | --- |
| `examples/ppo_trainer/` | PPO | `algorithm.adv_estimator=gae` |
| `examples/grpo_trainer/` | GRPO | `algorithm.adv_estimator=grpo` |
| `examples/rloo_trainer/` | RLOO | `algorithm.adv_estimator=rloo` |
| `examples/remax_trainer/` | ReMax | `algorithm.adv_estimator=remax` |
| `examples/reinforce_plus_plus_trainer/` | REINFORCE++ | `algorithm.adv_estimator=reinforce_plus_plus` |
| `examples/cispo_trainer/` | CISPO | `actor_rollout_ref.actor.policy_loss.loss_mode=cispo` |
| `examples/dppo_trainer/` | Divergence PPO | `policy_loss.loss_mode=dppo_tv` 或 `dppo_kl` |
| `examples/gdpo_trainer/` | GDPO | `algorithm.adv_estimator=gdpo` |
| `examples/gmpo_trainer/` | GMPO | `policy_loss.loss_mode=geo_mean` |
| `examples/gpg_trainer/` | Group Policy Gradient | `adv_estimator=gpg` + `loss_mode=gpg` |
| `examples/gspo_trainer/` | GSPO | `policy_loss.loss_mode=gspo` |
| `examples/sapo_trainer/` | SAPO | `policy_loss.loss_mode=sapo` |
| `examples/otb_trainer/` | Optimal Token Baseline | `adv_estimator=optimal_token_baseline` |
| `examples/mtp_trainer/` | Multi-Token-Prediction RL | `actor_rollout_ref.model.mtp.*` |
| `examples/on_policy_distillation_trainer/` | On-Policy Distillation | `distillation.enabled=True` |

> 不要把 `adv_estimator` 和 `policy_loss.loss_mode` 混在一起。前者决定 advantage 如何算，后者决定 policy loss 如何聚合/裁剪。

## Advantage Estimator 列表

v0.8.0 注册的主要 estimator：

```text
gae, grpo, reinforce_plus_plus, reinforce_plus_plus_baseline,
rloo, remax, opo, grpo_passk, gpg, rloo_vectorized,
grpo_vectorized, gdpo, optimal_token_baseline,
tir_optimal_token_baseline
```

典型用法：

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gdpo \
  reward.reward_manager.name=gdpo
```

## Policy Loss 列表

v0.8.0 注册的 policy loss：

```text
vanilla, dppo_tv, dppo_kl, gspo, sapo, gpg,
clip_cov, kl_cov, geo_mean, cispo, bypass_mode
```

典型用法：

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
```

## 选型建议

| 任务 | 首选 | 理由 |
| --- | --- | --- |
| 数学/代码可验证奖励，想快速起步 | GRPO | 不需要 critic，工程成本低 |
| 有 Reward Model，想走经典 RLHF | PPO | actor + critic 更完整 |
| 想减少 critic 成本但保留多样采样 | RLOO / ReMax | critic-free，样本效率和稳定性折中 |
| 大 MoE / 序列级稳定性 | GSPO / GMPO | 更关注序列级或几何平均聚合 |
| 复现特定论文 | 对应 examples 或 recipe | 先跑官方脚本，再改数据和模型 |
| 多 token prediction / speculative head | MTP | 只在支持 MTP 架构和后端时使用 |

## 正确查看示例

```bash
# 总索引
cat examples/README.md

# 算法示例
ls examples/grpo_trainer
ls examples/gspo_trainer
ls examples/mtp_trainer

# 研究型配方
ls recipe
```

## 学习顺序

1. 先把 GRPO 跑通，理解 `rollout.n`、`train_batch_size`、reward。
2. 再看 PPO，理解 critic 与 value loss。
3. 然后区分两类扩展：只改 advantage 的算法，以及只改 policy loss 的算法。
4. 最后再研究 MTP、on-policy distillation、fully async 等高级能力。

---

<!-- NAV_BOTTOM_START -->
> 导航： [上一篇：05-4 - RLOO / ReMax 详解](05-4-rloo-remax.md) | [返回目录](../../README.md#完整目录) | [下一篇：06-1 - SFT 基础训练](../06-sft-and-tuning/06-1-sft-basics.md)
<!-- NAV_BOTTOM_END -->
