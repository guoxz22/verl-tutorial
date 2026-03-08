# 10-1 - 训练监控

本章介绍如何监控 verl 训练过程。

## 支持的日志后端

| 后端 | 配置 | 特点 |
|------|------|------|
| Console | `["console"]` | 默认，输出到终端 |
| WandB | `["wandb"]` | 在线可视化 |
| TensorBoard | `["tensorboard"]` | 本地可视化 |
| MLflow | `["mlflow"]` | 企业级管理 |

## 配置方式

### 单一后端

```bash
trainer.logger='["console"]'
```

### 多后端

```bash
trainer.logger='["console","wandb","tensorboard"]'
```

### 完整配置

```bash
python -m verl.trainer.main_ppo \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='my_project' \
    trainer.experiment_name='my_experiment' \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.log_val_generations=10 \
    ...
```

## WandB 配置

### 登录

```bash
wandb login
```

### 配置

```bash
trainer.logger='["console","wandb"]'
trainer.project_name='verl_experiments'
trainer.experiment_name='grpo_qwen2_7b'
```

### 环境变量

```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=verl_project
export WANDB_ENTITY=your_entity
```

## TensorBoard 配置

```bash
trainer.logger='["console","tensorboard"]'
```

启动 TensorBoard：

```bash
tensorboard --logdir=runs/
```

## 关键监控指标

### 训练指标

| 指标 | 含义 | 正常范围 |
|------|------|----------|
| `train/reward` | 平均奖励 | 持续上升 |
| `train/actor_loss` | Policy Loss | 趋于稳定 |
| `train/critic_loss` | Value Loss | 下降 |
| `train/entropy` | 策略熵 | 缓慢下降 |
| `train/kl` | KL 散度 | < 目标值 |
| `train/clip_ratio` | PPO Clip 比率 | 0.1-0.3 |

### 性能指标

| 指标 | 含义 |
|------|------|
| `train/step_time` | 每步时间 |
| `train/samples_per_second` | 吞吐量 |
| `train/gpu_memory` | GPU 内存使用 |

### 验证指标

| 指标 | 含义 |
|------|------|
| `val/reward` | 验证奖励 |
| `val/accuracy` | 验证准确率（可验证任务） |

## 日志记录频率

```bash
# 验证频率
trainer.test_freq=5  # 每 5 步验证

# 训练前验证
trainer.val_before_train=True

# 记录验证生成
trainer.log_val_generations=10  # 记录 10 条生成
```

## 下一步

- [10-2-checkpoint.md](10-2-checkpoint.md) - 检查点管理
