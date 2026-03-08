# B - 常见错误与解决

本章列出 verl 使用中的常见错误及解决方案。

## 安装问题

### CUDA 版本不匹配

**错误**：
```
RuntimeError: CUDA version mismatch
```

**解决**：
```bash
# 检查版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 重新安装 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Flash Attention 安装失败

**错误**：
```
flash_attn installation failed
```

**解决**：
```bash
# 确保环境正确
pip install ninja packaging

# 安装
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## 内存问题

### OOM in Rollout

**错误**：
```
OutOfMemoryError: CUDA out of memory during rollout
```

**解决**：
```bash
# 降低 GPU 内存利用率
actor_rollout_ref.rollout.gpu_memory_utilization=0.4

# 减少 max_num_batched_tokens
actor_rollout_ref.rollout.max_num_batched_tokens=4096

# 增加 Tensor Parallel
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

### OOM in Training

**错误**：
```
OutOfMemoryError: CUDA out of memory during training
```

**解决**：
```bash
# 启用梯度检查点
actor_rollout_ref.model.enable_gradient_checkpointing=True

# 减小 micro batch size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8

# 启用 offload
actor_rollout_ref.ref.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

## Ray 问题

### Ray 初始化失败

**错误**：
```
RuntimeError: Unable to connect to Ray
```

**解决**：
```bash
# 停止现有 Ray
ray stop --force

# 重新启动
ray start --head

# 检查状态
ray status
```

### Ray Worker 超时

**错误**：
```
RayTimeoutError: Worker did not respond within timeout
```

**解决**：
```bash
# 增加 NCCL 超时
actor_rollout_ref.nccl_timeout=1200

# 增加 Ray 超时
trainer.ray_wait_register_center_timeout=600
```

## 分布式问题

### NCCL 通信错误

**错误**：
```
NCCL error: unhandled system error
```

**解决**：
```bash
# 设置 NCCL 配置
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# 检查网络
python -c "import torch; print(torch.distributed.is_nccl_available())"
```

### 多机同步问题

**错误**：
```
RuntimeError: Default process group has not been initialized
```

**解决**：
1. 检查所有节点环境一致
2. 确认端口开放
3. 检查 SSH 免密登录

## 训练问题

### Loss NaN

**错误**：
```
RuntimeError: Loss is nan
```

**解决**：
```bash
# 降低学习率
actor_rollout_ref.actor.optim.lr=1e-7

# 启用梯度裁剪
actor_rollout_ref.actor.grad_clip=0.5

# 检查奖励范围
# 确保 reward 没有异常值
```

### KL 散度过大

**症状**：
```
train/kl 持续增大，策略崩溃
```

**解决**：
```bash
# 增大 KL 系数
actor_rollout_ref.actor.kl_loss_coef=0.01

# 使用 adaptive KL
algorithm.kl_ctrl.type=adaptive
algorithm.kl_ctrl.target_kl=0.05

# 降低学习率
actor_rollout_ref.actor.optim.lr=5e-7
```

### Reward 不增长

**症状**：
```
train/reward 不上升或下降
```

**解决**：
1. 检查奖励函数是否正确
2. 检查数据格式是否正确
3. 尝试更大的学习率
4. 检查是否有足够的探索（temperature）

## 推理问题

### vLLM 启动失败

**错误**：
```
vLLM initialization failed
```

**解决**：
```bash
# 检查 vLLM 版本
pip show vllm

# 使用 V1
export VLLM_USE_V1=1

# 检查模型兼容性
# 某些模型需要特定 vLLM 版本
```

### SGLang 多轮问题

**错误**：
```
SGLang multi-turn error
```

**解决**：
```bash
# 确保启用多轮配置
actor_rollout_ref.rollout.multi_turn.enable=true

# 检查工具配置
# 确保工具正确注册
```

## 调试技巧

### 启用详细日志

```bash
export VERL_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=DEBUG
```

### 单 GPU 调试

```bash
# 使用最小配置
trainer.n_gpus_per_node=1
data.train_batch_size=8
actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

### 检查数据

```python
import pyarrow.parquet as pq

df = pq.read_table('train.parquet').to_pandas()
print(df.head())
print(df.columns)
print(len(df))
```
