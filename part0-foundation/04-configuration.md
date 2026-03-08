# 04 - 配置系统

verl 使用 **Hydra** 作为配置管理框架，支持灵活的参数覆盖和组合。

## 配置方式

### 方式一：命令行参数（推荐）

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/train.parquet \
    trainer.n_gpus_per_node=8
```

### 方式二：YAML 配置文件

创建 `config.yaml`：

```yaml
algorithm:
  adv_estimator: grpo
  gamma: 1.0

data:
  train_files: ./data/train.parquet
  max_prompt_length: 512

trainer:
  n_gpus_per_node: 8
  total_epochs: 15
```

运行：

```bash
python -m verl.trainer.main_ppo --config-path . --config-name config
```

### 方式三：混合使用

```bash
python -m verl.trainer.main_ppo \
    --config-path ./configs \
    --config-name my_config \
    trainer.n_gpus_per_node=4  # 覆盖配置文件中的值
```

## 配置结构

verl 的配置采用分层结构，主要包含以下模块：

```
config/
├── algorithm/          # 算法配置
├── actor_rollout_ref/  # Actor、Rollout、Reference 配置
├── critic/             # Critic 配置
├── data/               # 数据配置
├── reward/             # 奖励配置
└── trainer/            # 训练器配置
```

## 核心配置详解

### 1. Algorithm 配置

```bash
# 算法类型
algorithm.adv_estimator=grpo      # gae | grpo | reinforce_plus_plus | rloo

# 折扣因子
algorithm.gamma=1.0               # 未来奖励折扣
algorithm.lam=1.0                 # GAE 的 λ 参数

# KL 控制
algorithm.use_kl_in_reward=False  # 是否在奖励中加入 KL 惩罚
algorithm.kl_penalty=kl           # kl | abs | mse | low_var_kl

# KL 控制器
algorithm.kl_ctrl.type=fixed      # fixed | adaptive
algorithm.kl_ctrl.kl_coef=0.001   # KL 惩罚系数
algorithm.kl_ctrl.target_kl=0.1   # 目标 KL（adaptive 模式）
```

### 2. Data 配置

```bash
# 数据路径
data.train_files=./data/train.parquet
data.val_files=./data/val.parquet

# 数据量限制
data.train_max_samples=-1         # -1 表示使用全部数据
data.val_max_samples=-1

# 长度限制
data.max_prompt_length=512        # 最大 prompt 长度
data.max_response_length=1024     # 最大 response 长度

# 批次大小
data.train_batch_size=1024        # 每次迭代的总批次大小

# 数据处理
data.shuffle=True                 # 是否打乱数据
data.filter_overlong_prompts=True # 过滤超长 prompt
data.truncation=error             # error | left | right | middle
```

### 3. Actor/Rollout/Reference 配置

```bash
# 模型路径
actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct

# Actor 训练配置
actor_rollout_ref.actor.strategy=fsdp        # fsdp | fsdp2 | megatron
actor_rollout_ref.actor.optim.lr=1e-6        # 学习率
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40
actor_rollout_ref.actor.grad_clip=1.0        # 梯度裁剪
actor_rollout_ref.actor.clip_ratio=0.2       # PPO clip 比率

# KL Loss（GRPO 必需）
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl

# 优化
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.model.use_remove_padding=True

# FSDP Offload
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# Rollout 配置
actor_rollout_ref.rollout.name=vllm           # vllm | sglang | hf
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.rollout.n=5                 # 每个 prompt 生成 n 个 response
actor_rollout_ref.rollout.temperature=1.0

# Reference 配置
actor_rollout_ref.ref.fsdp_config.param_offload=True  # 节省内存
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40
```

### 4. Critic 配置

```bash
critic.strategy=fsdp
critic.optim.lr=1e-5
critic.ppo_micro_batch_size_per_gpu=40
critic.fsdp_config.param_offload=False
```

**注意**：GRPO 通常不需要 Critic，设置 `trainer.critic_warmup=0`。

### 5. Reward 配置

```bash
# 函数奖励（默认）
reward.reward_model.enable=False

# 模型奖励
reward.reward_model.enable=True
reward.reward_model.path=./models/reward_model
reward.reward_model.enable_resource_pool=False
```

### 6. Trainer 配置

```bash
# 分布式配置
trainer.nnodes=1                  # 节点数
trainer.n_gpus_per_node=8         # 每节点 GPU 数

# 训练控制
trainer.total_epochs=15           # 总 epoch 数
trainer.total_training_steps=null # 或指定总步数
trainer.critic_warmup=0           # Critic 预热步数

# 日志
trainer.logger='["console","wandb"]'
trainer.project_name='my_project'
trainer.experiment_name='my_experiment'

# 验证
trainer.val_before_train=True
trainer.test_freq=5               # 每 5 步验证一次

# 检查点
trainer.save_freq=20              # 每 20 步保存一次
trainer.default_local_dir=checkpoints/${trainer.project_name}/${trainer.experiment_name}
trainer.resume_mode=auto          # auto | disable | resume_path
```

## 配置模板

### GRPO 训练模板

```bash
#!/bin/bash
# run_grpo.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=1.0 \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/val.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.logger='["console","wandb"]'
```

### PPO 训练模板

```bash
#!/bin/bash
# run_ppo.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.02 \
    \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/val.parquet \
    data.train_batch_size=512 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    \
    critic.strategy=fsdp \
    critic.optim.lr=1e-5 \
    \
    trainer.critic_warmup=10 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=30
```

## 高级配置

### 序列并行

```bash
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
```

### LoRA 微调

```bash
actor_rollout_ref.model.lora_rank=64
actor_rollout_ref.model.lora_alpha=128
actor_rollout_ref.model.target_modules='["q_proj","v_proj"]'
```

### 多模态模型

```bash
data.image_key=images
actor_rollout_ref.model.path=Qwen/Qwen2-VL-7B-Instruct
```

## 配置调试

### 打印完整配置

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    --cfg job
```

### 查看默认配置

```python
from omegaconf import OmegaConf
from verl.trainer.config import get_default_config

config = get_default_config()
print(OmegaConf.to_yaml(config))
```

## 下一步

- 查看 `examples/` 目录中的实际配置示例
- 阅读 Part1 学习各类训练任务的配置
- 阅读 Part2 学习如何扩展配置
