# 10-2 - 检查点管理

本章介绍如何保存、恢复和管理训练检查点。

## 保存检查点

### 基本配置

```bash
# 保存频率（每 N 步）
trainer.save_freq=20

# 保存路径
trainer.default_local_dir=checkpoints/${trainer.project_name}/${trainer.experiment_name}

# 最大保存数量
trainer.max_actor_ckpt_to_keep=5
trainer.max_critic_ckpt_to_keep=5
```

### 保存内容

```bash
# 保存内容配置
actor_rollout_ref.actor.checkpoint.save_contents="['model', 'optimizer', 'extra']"
critic.checkpoint.save_contents="['model', 'optimizer']"
```

## 恢复训练

### 自动恢复

```bash
trainer.resume_mode=auto  # 自动从最新检查点恢复
```

### 禁用恢复

```bash
trainer.resume_mode=disable  # 从头开始
```

### 指定路径恢复

```bash
trainer.resume_mode=resume_path
trainer.resume_from_path=checkpoints/my_project/my_experiment/actor/step_100
```

## 检查点结构

```
checkpoints/
└── my_project/
    └── my_experiment/
        ├── actor/
        │   ├── step_0/
        │   │   ├── model/
        │   │   └── optimizer/
        │   ├── step_20/
        │   └── ...
        └── critic/
            ├── step_0/
            └── ...
```

## 加载内容配置

```bash
# 加载内容（可与保存内容不同）
actor_rollout_ref.actor.checkpoint.load_contents="['model']"  # 只加载模型
```

## 完整示例

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=./checkpoints/grpo_gsm8k \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode=auto \
    ...
```

## HDFS 支持

```bash
# 保存到 HDFS
trainer.default_hdfs_dir=hdfs://path/to/checkpoints

# 加载后删除本地副本
trainer.del_local_ckpt_after_load=True
```

## 下一步

- [10-3-profiling.md](10-3-profiling.md) - 性能分析
