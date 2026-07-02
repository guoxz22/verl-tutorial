# A - 配置参数参考

<!-- NAV_START -->
> 阅读： [← 14-2 - 案例研究](../part2-research-extension/14-paper-reproduction/14-2-case-studies.md) · [目录](../README.md#完整目录) · [B - 常见错误与解决 →](B-common-errors.md)
<!-- NAV_END -->

本附录按 v0.8.0 的 `verl/trainer/config/_generated_ppo_trainer.yaml`、`rollout/rollout.yaml`、`reward/reward.yaml` 和 `sft_trainer_engine.yaml` 整理。它不是完整复制配置文件，而是列出最常改、最容易写错的键。

## 版本基准

| 项目 | 值 |
| --- | --- |
| verl release | `v0.8.0` |
| RL 入口 | `python -m verl.trainer.main_ppo` |
| SFT 入口 | `torchrun ... -m verl.trainer.sft_trainer` |
| PPO 生成配置 | `verl/trainer/config/_generated_ppo_trainer.yaml` |
| SFT 配置 | `verl/trainer/config/sft_trainer_engine.yaml` |

## Algorithm

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `algorithm.adv_estimator` | `gae` | `gae`、`grpo`、`rloo`、`remax`、`gpg`、`gdpo`、`optimal_token_baseline` 等 |
| `algorithm.gamma` | `1.0` | 折扣因子 |
| `algorithm.lam` | `1.0` | GAE lambda |
| `algorithm.use_kl_in_reward` | `False` | 是否把 KL 惩罚加到 reward 中 |
| `algorithm.kl_penalty` | `kl` | `kl`、`abs`、`mse`、`low_var_kl`、`full` |
| `algorithm.kl_ctrl.type` | `fixed` | `fixed` 或 `adaptive` |
| `algorithm.kl_ctrl.kl_coef` | `0.001` | KL 系数 |
| `algorithm.use_pf_ppo` | `False` | Preference Feedback PPO 开关 |

## Data

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `data.train_files` | `~/data/rlhf/gsm8k/train.parquet` | 训练 parquet |
| `data.val_files` | `~/data/rlhf/gsm8k/test.parquet` | 验证 parquet |
| `data.prompt_key` | `prompt` | prompt 字段 |
| `data.reward_fn_key` | `data_source` | reward function 的 data_source 字段 |
| `data.train_batch_size` | `1024` | 每 step prompt 数 |
| `data.max_prompt_length` | `512` | prompt 最大 token |
| `data.max_response_length` | `512` | response 最大 token |
| `data.filter_overlong_prompts` | `False` | 是否过滤超长 prompt |
| `data.truncation` | `error` | 超长处理策略 |
| `data.return_raw_chat` | `True` | Agent/multi-turn 常需要保留 raw messages |
| `data.image_key` / `video_key` / `audio_key` | `images` / `videos` / `audios` | 多模态字段 |

## Actor / Model

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `actor_rollout_ref.model.path` | 必填 | actor/ref/rollout 使用的 HF 模型路径 |
| `actor_rollout_ref.model.use_remove_padding` | `False` | 去 padding 优化 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `False` | 梯度检查点 |
| `actor_rollout_ref.model.lora_rank` | `0` | LoRA rank，0 表示不启用 |
| `actor_rollout_ref.model.mtp.enable` | `False` | MTP 模块开关 |
| `actor_rollout_ref.actor.strategy` | `fsdp` | FSDP/FSDP2 路径常用；Megatron 等通过 `model_engine` |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | actor 学习率 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` | actor mini batch |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `null` | 每 GPU micro batch，优先使用 `_per_gpu` |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | 动态 batch token 上限 |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` | `vanilla`、`gspo`、`sapo`、`gpg`、`geo_mean`、`cispo` 等 |
| `actor_rollout_ref.actor.use_kl_loss` | `False` | actor loss 侧 KL |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | loss 侧 KL 类型 |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `False` | FSDP 参数 offload |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `False` | FSDP optimizer offload |

## Rollout

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `actor_rollout_ref.rollout.name` | 必填 | `hf`、`vllm`、`sglang`、`trtllm` |
| `actor_rollout_ref.rollout.mode` | `async` | 默认 async |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | rollout TP |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | 推理后端显存比例 |
| `actor_rollout_ref.rollout.n` | `1` | 每个 prompt 采样数 |
| `actor_rollout_ref.rollout.temperature` | `1.0` | 采样温度 |
| `actor_rollout_ref.rollout.top_p` | `1.0` | top-p |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | rollout batching token 上限 |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `null` | rollout logprob micro batch |
| `actor_rollout_ref.rollout.engine_kwargs.vllm` | `{}` | vLLM 专属参数，新增键要加 `+` |
| `actor_rollout_ref.rollout.engine_kwargs.sglang` | `{}` | SGLang 专属参数 |
| `actor_rollout_ref.rollout.engine_kwargs.trtllm` | `{}` | TRTLLM 专属参数 |

## Multi-turn / Agent

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `actor_rollout_ref.rollout.multi_turn.enable` | `False` | 多轮开关 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `null` | BaseTool YAML |
| `actor_rollout_ref.rollout.multi_turn.function_tool_path` | `null` | `@function_tool` Python 文件 |
| `actor_rollout_ref.rollout.multi_turn.max_user_turns` | `null` | 用户/环境轮数上限 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | `null` | assistant 轮数上限 |
| `actor_rollout_ref.rollout.multi_turn.format` | `hermes` | 工具调用格式 |
| `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode` | `strict` | 多轮 tokenization 检查 |
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` | 默认 AgentLoop |
| `actor_rollout_ref.rollout.agent.agent_loop_config_path` | `null` | AgentLoop 配置文件 |

## Critic

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `critic.enable` | `null` | 是否启用 critic；PPO 需要，GRPO 通常不需要 |
| `critic.model.path` | 继承/必填 | critic 模型路径 |
| `critic.optim.lr` | `1e-5` | critic 学习率 |
| `critic.ppo_micro_batch_size_per_gpu` | `null` | critic micro batch |
| `critic.ppo_max_token_len_per_gpu` | `32768` | critic 动态 batch token 上限 |
| `critic.fsdp.param_offload` | `False` | critic 参数 offload |

## Reward

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `reward.num_workers` | `8` | reward manager 并行 worker 数 |
| `reward.custom_reward_function.path` | `null` | 自定义 reward 文件 |
| `reward.custom_reward_function.name` | `compute_score` | reward 函数名 |
| `reward.reward_manager.source` | `register` | manager 来源 |
| `reward.reward_manager.name` | `naive` | manager 名称 |
| `reward.reward_manager.module.path` | `null` | 自定义 manager 模块路径 |
| `reward.reward_model.enable` | `False` | 是否启用 Reward Model |
| `reward.reward_model.model_path` | `null` | Reward Model 路径 |
| `reward.reward_model.rollout.name` | 必填（启用 RM 时） | RM 推理后端 |
| `reward.reward_model.enable_resource_pool` | `False` | RM 是否单独资源池 |
| `reward.sandbox_fusion.url` | `null` | Sandbox Fusion 地址 |

## Trainer

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `trainer.nnodes` | `1` | 节点数 |
| `trainer.n_gpus_per_node` | `8` | 每节点 GPU 数 |
| `trainer.total_epochs` | `30` | 总 epoch |
| `trainer.total_training_steps` | `null` | 指定后优先于 epochs |
| `trainer.logger` | `['console','wandb']` | 日志后端 |
| `trainer.project_name` | `verl_examples` | 项目名 |
| `trainer.experiment_name` | `gsm8k` | 实验名 |
| `trainer.save_freq` | `-1` | 保存频率 |
| `trainer.test_freq` | `-1` | 验证频率 |
| `trainer.resume_mode` | `auto` | `auto`、`disable`、`resume_path` |
| `trainer.resume_from_path` | `null` | resume 路径 |
| `trainer.val_before_train` | `True` | 训练前验证 |

## SFT

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `data.micro_batch_size_per_gpu` | `4` | SFT 每 GPU micro batch |
| `data.messages_key` | `messages` | SFT messages 字段 |
| `data.max_length` | `1024` | SFT 总长度 |
| `engine` | `fsdp` | SFT engine |
| `engine.ulysses_sequence_parallel_size` | 视 engine 配置 | Ulysses 序列并行 |
| `model.path` | 必填 | SFT 模型路径 |
| `optim.lr` | 视配置 | SFT 学习率 |

## 迁移速查

| 旧写法 | v0.8.0 推荐写法 |
| --- | --- |
| `reward.reward_function.path` | `reward.custom_reward_function.path` |
| `reward.reward_function.name` | `reward.custom_reward_function.name` |
| `reward.reward_model.path` | `reward.reward_model.model_path` |
| `reward.reward_manager=my_rm` | `reward.reward_manager.name=my_rm` |
| `actor_rollout_ref.actor.loss_type` | `actor_rollout_ref.actor.policy_loss.loss_mode` |
| `data.micro_batch_size` in SFT | `data.micro_batch_size_per_gpu` |
| `actor_rollout_ref.rollout.multi_turn.tool.dict.*` | `tool_config_path` 或 `function_tool_path` |
| `data.tool_provider` | `agent.default_agent_loop` 或数据字段 `agent_name` |

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 14-2 - 案例研究](../part2-research-extension/14-paper-reproduction/14-2-case-studies.md) · [目录](../README.md#完整目录) · [B - 常见错误与解决 →](B-common-errors.md)
<!-- NAV_BOTTOM_END -->
