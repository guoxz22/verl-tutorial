# 01 - 安装与配置

## 环境要求

| 依赖 | 版本要求 |
|------|----------|
| Python | >= 3.10 |
| CUDA | >= 12.8 |
| cuDNN | >= 9.10.0 |

**硬件支持**：NVIDIA GPU（推荐）、AMD GPU（ROCm）、华为 Ascend NPU

## 安装方式

### 方式一：使用 Docker（推荐）

Docker 是最简单的安装方式，所有依赖已预装。

**1. 拉取镜像**

```bash
# vLLM 版本
docker pull verlai/verl:vllm011.latest

# SGLang 版本
docker pull verlai/verl:sgl055.latest
```

**2. 启动容器**

```bash
# 创建容器
docker create --runtime=nvidia --gpus all --net=host \
    --shm-size="10g" --cap-add=SYS_ADMIN \
    -v $(pwd):/workspace/verl \
    --name verl verlai/verl:vllm011.latest sleep infinity

# 启动并进入
docker start verl
docker exec -it verl bash
```

**3. 安装 verl**

```bash
# 在容器内
git clone https://github.com/volcengine/verl.git && cd verl
pip3 install --no-deps -e .
```

### 方式二：从源码安装

适用于需要自定义环境或修改 verl 源码的场景。

**1. 创建 Conda 环境**

```bash
conda create -n verl python==3.12
conda activate verl
```

**2. 安装依赖**

```bash
# 克隆仓库
git clone https://github.com/volcengine/verl.git
cd verl

# 安装全部依赖（包含 Megatron-LM）
bash scripts/install_vllm_sglang_mcore.sh

# 或仅安装 FSDP 依赖（更快）
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

**3. 安装 verl**

```bash
pip install --no-deps -e .
```

### 方式三：指定推理引擎安装

如果需要同时支持 vLLM 和 SGLang：

```bash
pip install -e .[vllm]
pip install -e .[sglang]
```

## 后端选择指南

### 训练后端

| 后端 | 特点 | 推荐场景 |
|------|------|----------|
| **FSDP** | PyTorch 原生，配置简单 | 研究、原型开发、中小模型 |
| **FSDP2** | 更高性能，支持 CPU Offload | 内存受限、需要更高吞吐 |
| **Megatron-LM** | 大规模分布式优化 | 超大模型（70B+）、多机训练 |

**启用 FSDP2**（推荐）：

```bash
actor_rollout_ref.ref.strategy=fsdp2 \
actor_rollout_ref.actor.strategy=fsdp2 \
critic.strategy=fsdp2
```

### 推理引擎

| 引擎 | 特点 | 推荐场景 |
|------|------|----------|
| **vLLM** | 高吞吐，稳定 | 标准训练任务 |
| **SGLang** | 多轮对话、Agent 支持 | AgentRL、多轮训练 |
| **TRTLLM** | NVIDIA 优化 | 生产部署 |

**vLLM 性能优化**：启用 V1 架构

```bash
export VLLM_USE_V1=1
```

## 可选组件安装

### Flash Attention（推荐）

```bash
pip install flash-attn --no-build-isolation
```

### NVIDIA Apex（Megatron-LM 需要）

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
MAX_JOBS=32 pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
```

### Liger Kernel（优化训练效率）

```bash
pip install liger-kernel
```

## 验证安装

**1. 检查版本**

```bash
python -c "import verl; print(verl.__version__)"
```

**2. 检查 GPU 可用性**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

**3. 运行最小测试**

```bash
# 快速验证（需要至少 1 GPU）
python -c "
from verl.trainer.main_ppo import main
print('verl import successful')
"
```

## 常见安装问题

### 问题 1：PyTorch 版本冲突

**症状**：安装 vLLM/SGLang 后 PyTorch 被降级

**解决**：先安装推理框架，再安装 verl

```bash
# 先安装 vLLM（会自动安装兼容的 PyTorch）
pip install vllm

# 再安装 verl（不覆盖依赖）
pip install --no-deps -e .
```

### 问题 2：CUDA 版本不匹配

**症状**：`RuntimeError: CUDA version mismatch`

**解决**：检查 CUDA 版本一致性

```bash
# 检查系统 CUDA
nvcc --version

# 检查 PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

### 问题 3：tensordict 版本冲突

**症状**：`ImportError: cannot import name ... from tensordict`

**解决**：安装正确版本

```bash
pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
```

### 问题 4：Ray 启动失败

**症状**：`RuntimeError: Unable to connect to Ray`

**解决**：清理 Ray 会话并重启

```bash
ray stop --force
ray start --head
```

## 环境变量配置

推荐在 `~/.bashrc` 或训练脚本中设置：

```bash
# vLLM 优化
export VLLM_USE_V1=1

# Tokenizer 并行
export TOKENIZERS_PARALLELISM=false

# NCCL 调试（需要时开启）
export NCCL_DEBUG=WARN

# Ray 配置
export RAY_DEDUP_LOGS=0
```

## 下一步

安装完成后，继续阅读 [02-quick-start.md](02-quick-start.md) 运行第一个训练任务。
