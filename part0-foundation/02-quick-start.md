# 02 - 快速上手

本章带你用 5 分钟运行第一个 GRPO 训练任务。

## 前置准备

### 1. 准备数据

verl 使用 Parquet 格式的数据文件。我们以 GSM8K 数学数据集为例。

**下载数据**：

```bash
# 创建数据目录
mkdir -p ~/data/gsm8k

# 下载预处理好的数据（或使用自己的数据）
# 数据格式见下一节说明
```

**数据格式要求**：

数据文件需要包含 `data_source`（或 `prompt`）字段，存储输入提示词。

```python
# 示例：gsm8k/train.parquet 的结构
import pandas as pd
import pyarrow.parquet as pq

# 读取数据
df = pq.read_table('train.parquet').to_pandas()
print(df.columns)  # 应包含 'data_source' 或 'prompt' 字段
print(df.head())
```

**最小测试数据生成**：

```python
# create_test_data.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 创建简单的测试数据
data = {
    'data_source': [
        'What is 2 + 3?',
        'Calculate 10 - 4.',
        'What is 5 * 6?',
    ]
}

df = pd.DataFrame(data)
table = pa.Table.from_pandas(df)
pq.write_table(table, 'test_prompts.parquet')
print(f"Created test data with {len(df)} samples")
```

```bash
python create_test_data.py
```

### 2. 准备模型

verl 支持 Hugging Face 格式的模型。推荐使用 Qwen2.5 系列模型。

```bash
# 模型会自动从 Hugging Face 下载
# 或预先下载到本地
# huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
```

## 运行第一个训练

### 最小可运行示例（单 GPU）

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./test_prompts.parquet \
    data.val_files=./test_prompts.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_quickstart' \
    trainer.experiment_name='first_run' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=1
```

### 完整 GRPO 训练脚本（8 GPU）

创建文件 `run_grpo_gsm8k.sh`：

```bash
#!/bin/bash
set -x

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15
```

运行：

```bash
chmod +x run_grpo_gsm8k.sh
./run_grpo_gsm8k.sh
```

## 训练输出解读

### 控制台输出

```
ray init kwargs: {...}
TaskRunner hostname: xxx, PID: xxx
{
    'algorithm': {'adv_estimator': 'grpo', ...},
    'trainer': {'n_gpus_per_node': 8, ...},
    ...
}
[ActorRollout] Initializing model...
[Rollout] Generating responses...
[Reward] Computing rewards...
[PPO] Updating policy...
```

### 关键指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| `train/reward` | 平均奖励 | 应随训练上升 |
| `train/actor_loss` | Policy Loss | 应趋于稳定 |
| `train/entropy` | 策略熵 | 不应过快下降 |
| `train/kl` | KL 散度 | 控制在目标范围内 |

### 检查点保存

训练检查点默认保存在：

```
checkpoints/verl_grpo_example_gsm8k/qwen2_7b_function_rm/
├── actor/
│   ├── step_0/
│   ├── step_20/
│   └── ...
└── critic/
    └── ...
```

## 使用 WandB 监控

**1. 配置 WandB**

```bash
wandb login
```

**2. 启用 WandB 日志**

```bash
trainer.logger='["console","wandb"]' \
trainer.project_name='my_project' \
trainer.experiment_name='my_experiment'
```

## 从检查点恢复

```bash
python -m verl.trainer.main_ppo \
    ... \
    trainer.resume_mode=auto \
    # 或指定路径
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=checkpoints/xxx/actor/step_100
```

## 下一步

- 阅读 [03-core-concepts.md](03-core-concepts.md) 理解 verl 的核心概念
- 阅读 [04-configuration.md](04-configuration.md) 深入了解配置系统
- 查看 `examples/` 目录了解更多示例
