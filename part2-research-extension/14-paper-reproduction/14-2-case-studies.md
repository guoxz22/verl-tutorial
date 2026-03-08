# 14-2 - 案例研究

本章通过实际案例展示如何复现论文。

## 案例 1：复现 DeepSeek-R1 (GRPO)

### 论文信息

- 论文：DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
- 算法：GRPO (Group Relative Policy Optimization)

### verl 对应

verl 已内置 GRPO 支持，直接使用：

```bash
#!/bin/bash
# reproduce_r1.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-32B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='r1_reproduction' \
    trainer.experiment_name='qwen2_5_32b_grpo'
```

## 案例 2：复现 DAPO

### 论文信息

- 论文：DAPO: An Open-Source RL Framework for Advanced Reasoning
- 算法：Dynamic Advantage Policy Optimization

### 需要自定义

DAPO 有一些特殊处理，需要扩展：

```python
# dapo_advantage.py
import torch
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("dapo_adv")
def dapo_advantage(data, config):
    """DAPO Advantage 计算"""
    rewards = data.batch['rewards']
    response_mask = data.batch['response_mask']

    # DAPO 特有：动态 Advantage 缩放
    # 根据 reward 分布调整 advantage
    reward_mean = rewards.mean()
    reward_std = rewards.std() + 1e-8

    # 动态阈值
    dynamic_threshold = reward_mean + config.dapo_alpha * reward_std

    # 计算 advantage
    advantages = (rewards - dynamic_threshold) / reward_std

    # DAPO 特有：clip advantage
    advantages = torch.clamp(advantages, -config.dapo_clip, config.dapo_clip)

    return advantages * response_mask
```

```bash
# reproduce_dapo.sh
export VERL_USE_EXTERNAL_MODULES=./dapo_advantage.py

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=dapo_adv \
    algorithm.dapo_alpha=0.5 \
    algorithm.dapo_clip=5.0 \
    ...
```

## 案例 3：复现 ReTool

### 论文信息

- 论文：ReTool: Reinforcement Learning for Strategic Tool Use
- 特点：多轮 + Tool Calling

### 使用 SGLang 多轮

```bash
#!/bin/bash
# reproduce_retool.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/retool/train.parquet \
    data.val_files=$HOME/data/retool/test.parquet \
    data.tool_provider=retool \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.tool.sandbox_fusion.enable=true \
    \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15
```

## 已知复现项目

| 项目 | 论文 | GitHub |
|------|------|--------|
| TinyZero | R1 Zero | github.com/Jiayi-Pan/TinyZero |
| Easy-R1 | Multi-modal | github.com/hiyouga/EasyR1 |
| Search-R1 | Search + Reasoning | github.com/PeterGriffinJin/Search-R1 |
| RAGEN | Agent | github.com/ZihanWang314/ragen |

## 复现建议

1. **从小模型开始**：先用 0.5B-3B 模型验证算法
2. **使用小数据集**：GSM8K (7K 样本) 适合快速验证
3. **监控关键指标**：reward, kl, entropy
4. **对比 baseline**：先跑 GRPO baseline，再比较自定义算法
5. **保存检查点**：便于断点续训和消融实验

## 下一步

- [附录 A](../../appendix/A-config-reference.md) - 配置参数参考
- [附录 B](../../appendix/B-common-errors.md) - 常见错误
