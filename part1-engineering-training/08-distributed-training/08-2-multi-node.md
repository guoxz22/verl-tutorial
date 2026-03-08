# 08-2 - 多机训练

本章介绍多机分布式训练配置。

## Ray 集群配置

### 启动 Ray 集群

**Head 节点**：

```bash
ray start --head \
    --num-gpus=8 \
    --port=6379 \
    --dashboard-port=8265
```

**Worker 节点**：

```bash
ray start --address=<head_ip>:6379 \
    --num-gpus=8
```

### 验证集群

```bash
# 查看集群状态
ray status

# 应显示类似：
# Node status:
#  1 node_xxxx (head)
#  3 node_yyyy (worker)
# Resources:
#  node:xxxx: 1.0
#  node:yyyy: 3.0
#  GPU: 32.0
```

## 训练脚本

### 4 机 32 卡

```bash
#!/bin/bash
# run_multinode_4x8.sh

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=2048 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='multinode_4x8'
```

### 使用 Slurm

```bash
#!/bin/bash
#SBATCH --job-name=verl_grpo
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00

set -x

# 加载环境
source activate verl

# 启动训练
srun python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/gsm8k/train.parquet \
    data.val_files=/data/gsm8k/test.parquet \
    data.train_batch_size=2048 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.total_epochs=10
```

## 关键配置

```bash
# 节点配置
trainer.nnodes=4                    # 节点数
trainer.n_gpus_per_node=8           # 每节点 GPU 数

# Ray 配置
ray_kwargs.ray_init.num_cpus=null   # CPU 数量
ray_kwargs.ray_init.runtime_env={}

# 超时配置
actor_rollout_ref.nccl_timeout=600  # NCCL 超时（秒）
```

## 网络优化

### NCCL 配置

```bash
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0
```

### 检查网络

```bash
# 测试节点间通信
ray health-check <head_ip>:6379

# 检查 NCCL
python -c "import torch; print(torch.distributed.is_nccl_available())"
```

## 常见问题

### 节点间通信失败

**解决**：
1. 检查防火墙设置
2. 确认端口 6379、8265 开放
3. 检查 NCCL 配置

### 同步问题

**解决**：
1. 增加 `nccl_timeout`
2. 检查网络带宽
3. 确保所有节点使用相同环境

## 下一步

- [08-3-backends.md](08-3-backends.md) - 后端选择
