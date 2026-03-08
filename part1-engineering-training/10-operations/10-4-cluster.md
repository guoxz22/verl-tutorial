# 10-4 - 集群调度

本章介绍如何在集群上运行 verl 训练。

## Slurm 调度

### 基本脚本

```bash
#!/bin/bash
#SBATCH --job-name=verl_grpo
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err

set -x

# 加载环境
module load cuda/12.4
source activate verl

# 运行训练
srun python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.total_epochs=10
```

### 提交任务

```bash
sbatch run_grpo.sbatch
```

### 查看任务

```bash
squeue -u $USER
sacct -j <job_id>
```

## SkyPilot 调度

### 配置文件

```yaml
# sky.yaml
name: verl-grpo

resources:
  cloud: aws
  accelerators: A100:8
  memory: 512+

num_nodes: 4

setup: |
  pip install verl

run: |
  python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ...
```

### 运行

```bash
sky launch sky.yaml
```

## Ray 集群

### 启动 Head 节点

```bash
ray up cluster.yaml
```

### 提交任务

```bash
ray job submit --address=http://<head-ip>:8265 \
    -- python -m verl.trainer.main_ppo ...
```

## 容错配置

### 自动恢复

```bash
trainer.resume_mode=auto
```

### ESI 配置

```bash
# ESI（弹性服务器实例）自动保存
trainer.esi_redundant_time=300  # 提前 5 分钟保存
```

## 监控

### Ray Dashboard

```bash
# 访问 http://<head-ip>:8265
```

### Slurm 监控

```bash
# 实时日志
tail -f log_<job_id>.out

# 资源使用
sstat -j <job_id>
```

## 下一步

- Part2: [研究扩展](../../part2-research-extension/) - 自定义算法
