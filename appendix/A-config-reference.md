# A - 配置参数参考

本章提供 verl 配置参数的快速参考。

## Algorithm 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `algorithm.adv_estimator` | str | `gae` | Advantage 估计器：gae, grpo, reinforce_plus_plus, rloo |
| `algorithm.gamma` | float | 1.0 | 折扣因子 |
| `algorithm.lam` | float | 1.0 | GAE lambda |
| `algorithm.use_kl_in_reward` | bool | False | 是否在奖励中加入 KL 惩罚 |
| `algorithm.kl_penalty` | str | `kl` | KL 惩罚类型：kl, abs, mse, low_var_kl |
| `algorithm.kl_ctrl.type` | str | `fixed` | KL 控制器：fixed, adaptive |
| `algorithm.kl_ctrl.kl_coef` | float | 0.001 | KL 惩罚系数 |
| `algorithm.kl_ctrl.target_kl` | float | 0.1 | 目标 KL（adaptive 模式） |

## Data 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data.train_files` | str | - | 训练数据路径 |
| `data.val_files` | str | - | 验证数据路径 |
| `data.train_batch_size` | int | 1024 | 训练批次大小 |
| `data.max_prompt_length` | int | 512 | 最大 prompt 长度 |
| `data.max_response_length` | int | 512 | 最大 response 长度 |
| `data.shuffle` | bool | True | 是否打乱数据 |
| `data.filter_overlong_prompts` | bool | False | 过滤超长 prompt |
| `data.truncation` | str | `error` | 截断策略：error, left, right, middle |

## Actor 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `actor_rollout_ref.model.path` | str | - | 模型路径 |
| `actor_rollout_ref.actor.strategy` | str | `fsdp` | 训练策略：fsdp, fsdp2, megatron |
| `actor_rollout_ref.actor.optim.lr` | float | 1e-6 | 学习率 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | int | 256 | PPO mini batch 大小 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | int | 8 | 每 GPU micro batch 大小 |
| `actor_rollout_ref.actor.use_kl_loss` | bool | False | 是否使用 KL Loss |
| `actor_rollout_ref.actor.kl_loss_coef` | float | 0.001 | KL Loss 系数 |
| `actor_rollout_ref.actor.clip_ratio` | float | 0.2 | PPO clip 比率 |
| `actor_rollout_ref.actor.entropy_coeff` | float | 0.0 | 熵系数 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | bool | False | 梯度检查点 |

## Rollout 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `actor_rollout_ref.rollout.name` | str | `vllm` | 推理引擎：vllm, sglang, hf |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | int | 1 | Tensor Parallel 大小 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | float | 0.5 | GPU 内存利用率 |
| `actor_rollout_ref.rollout.n` | int | 1 | 每个 prompt 采样数 |
| `actor_rollout_ref.rollout.temperature` | float | 1.0 | 采样温度 |
| `actor_rollout_ref.rollout.top_p` | float | 1.0 | Top-p 采样 |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | int | 8192 | 最大批处理 tokens |

## Critic 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `critic.strategy` | str | `fsdp` | 训练策略 |
| `critic.optim.lr` | float | 1e-5 | 学习率 |
| `critic.ppo_micro_batch_size_per_gpu` | int | 8 | Micro batch 大小 |
| `critic.model.enable_gradient_checkpointing` | bool | False | 梯度检查点 |

## Trainer 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `trainer.n_gpus_per_node` | int | 8 | 每节点 GPU 数 |
| `trainer.nnodes` | int | 1 | 节点数 |
| `trainer.total_epochs` | int | 30 | 总 epoch 数 |
| `trainer.total_training_steps` | int | null | 总步数（优先于 epochs） |
| `trainer.critic_warmup` | int | 0 | Critic 预热步数 |
| `trainer.save_freq` | int | -1 | 保存频率 |
| `trainer.test_freq` | int | -1 | 验证频率 |
| `trainer.logger` | list | `["console","wandb"]` | 日志后端 |
| `trainer.project_name` | str | `verl_examples` | 项目名 |
| `trainer.experiment_name` | str | `gsm8k` | 实验名 |
| `trainer.resume_mode` | str | `auto` | 恢复模式：auto, disable, resume_path |

## Reward 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reward.reward_model.enable` | bool | False | 启用 Reward Model |
| `reward.reward_model.model_path` | str | - | Reward Model 路径 |
| `reward.num_workers` | int | 1 | 奖励计算并行度 |

## FSDP 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `actor_rollout_ref.actor.fsdp_config.param_offload` | bool | False | 参数 Offload |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | bool | False | 优化器 Offload |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | int | -1 | FSDP 分片大小 |
