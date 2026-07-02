# 04 - 配置系统

<!-- NAV_START -->
> 阅读： [← 03 - 核心概念](03-core-concepts.md) · [目录](../README.md#catalog) · [05-1 - PPO 训练 →](../part1-engineering-training/05-algorithms/05-1-ppo.md)
<!-- NAV_END -->

verl 使用 Hydra 管理配置。学习 verl 配置有一个原则：**不要凭记忆写键名，优先查生成配置**。

v0.8.0 最重要的事实源：

```text
verl/trainer/config/_generated_ppo_trainer.yaml
verl/trainer/config/ppo_trainer.yaml
verl/trainer/config/sft_trainer_engine.yaml
verl/trainer/config/rollout/rollout.yaml
verl/trainer/config/reward/reward.yaml
```

## 命令行覆盖

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  trainer.n_gpus_per_node=8
```

Hydra 的命令行覆盖是最常见、最适合教程和实验记录的方式。

## YAML 配置

```yaml
# config.yaml
algorithm:
  adv_estimator: grpo

data:
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  train_batch_size: 256

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-0.5B-Instruct
  rollout:
    name: vllm
    n: 8

trainer:
  n_gpus_per_node: 8
  logger: [console]
```

运行：

```bash
python -m verl.trainer.main_ppo --config-path . --config-name config
```

## 新增键要加 `+`

如果键不在结构化配置里，需要加 `+`：

```bash
+actor_rollout_ref.actor.policy_loss.tau_pos=1.0
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[1,8,16,32]'
```

没有 `+` 时常见报错：`Key is not in struct`。

## 配置树总览

```text
algorithm                         # advantage、KL、PF-PPO
actor_rollout_ref.model           # actor/ref/rollout 共享模型配置
actor_rollout_ref.actor           # actor update、policy loss、FSDP/Megatron/VeOmni
actor_rollout_ref.rollout         # rollout backend、sampling、multi_turn、agent
actor_rollout_ref.ref             # reference logprob
critic                            # value model
reward                            # reward function / manager / model / sandbox
trainer                           # 节点、日志、checkpoint、验证
```

## Algorithm

```bash
algorithm.adv_estimator=grpo
algorithm.gamma=1.0
algorithm.lam=1.0
algorithm.use_kl_in_reward=False
algorithm.kl_penalty=kl
algorithm.kl_ctrl.type=fixed
algorithm.kl_ctrl.kl_coef=0.001
```

常见 estimator：`gae`、`grpo`、`rloo`、`remax`、`gpg`、`gdpo`、`optimal_token_baseline`。

## Actor 与 Policy Loss

```bash
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
```

`policy_loss.loss_mode` 可选：`vanilla`、`gspo`、`sapo`、`gpg`、`geo_mean`、`cispo`、`dppo_tv`、`dppo_kl` 等。

## Rollout

```bash
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=1.0
```

后端：`hf`、`vllm`、`sglang`、`trtllm`。

后端专属参数放在 `engine_kwargs`：

```bash
+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton
```

## Multi-turn / Agent

```bash
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.tool_config_path=/path/to/tool_config.yaml
actor_rollout_ref.rollout.multi_turn.function_tool_path=/path/to/tools.py
actor_rollout_ref.rollout.multi_turn.max_user_turns=2
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

旧写法 `multi_turn.tool.dict.*` 和 `multi_turn.server_mode` 不适用于 v0.8.0 教程。

## Reward

```bash
reward.num_workers=8
reward.custom_reward_function.path=$PWD/reward_fn.py
reward.custom_reward_function.name=compute_score
reward.reward_manager.name=naive
```

Reward Model：

```bash
reward.reward_model.enable=True
reward.reward_model.model_path=$HOME/models/reward_model
reward.reward_model.rollout.name=vllm
reward.reward_model.rollout.tensor_model_parallel_size=1
```

## Trainer

```bash
trainer.nnodes=1
trainer.n_gpus_per_node=8
trainer.total_epochs=15
trainer.total_training_steps=null
trainer.save_freq=20
trainer.test_freq=5
trainer.logger='["console","wandb"]'
trainer.resume_mode=auto
```

## SFT 配置

SFT 使用单独入口和配置：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m verl.trainer.sft_trainer \
  engine=fsdp \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.messages_key=messages \
  data.micro_batch_size_per_gpu=4 \
  model.path=Qwen/Qwen2.5-0.5B-Instruct
```

## 调试配置

打印最终配置：

```bash
python -m verl.trainer.main_ppo --cfg job
```

只跑少量 step：

```bash
trainer.total_training_steps=1 trainer.save_freq=-1 trainer.test_freq=-1 trainer.logger='["console"]'
```

## 迁移提示

| 看到旧键 | 改成 |
| --- | --- |
| `reward.reward_function.*` | `reward.custom_reward_function.*` |
| `actor_rollout_ref.actor.loss_type` | `actor_rollout_ref.actor.policy_loss.loss_mode` |
| `data.micro_batch_size` in SFT | `data.micro_batch_size_per_gpu` |
| `data.tool_provider` | `actor_rollout_ref.rollout.agent.default_agent_loop` / 数据字段 `agent_name` |

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 03 - 核心概念](03-core-concepts.md) · [目录](../README.md#catalog) · [05-1 - PPO 训练 →](../part1-engineering-training/05-algorithms/05-1-ppo.md)
<!-- NAV_BOTTOM_END -->
