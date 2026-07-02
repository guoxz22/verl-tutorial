# 01 - 安装与配置

<!-- NAV_START -->
> 阅读： [← 00 - 什么是 verl](00-what-is-verl.md) · [目录](../README.md#catalog) · [02 - 快速上手 →](02-quick-start.md)
<!-- NAV_END -->

本章给出一套能复现 v0.8.0 示例的安装路线，并说明何时使用 Docker，何时手动配置环境。

## 环境要求

verl v0.8.0 官方安装文档给出的基础要求：

| 项目 | 建议 |
| --- | --- |
| Python | `>=3.10`，官方示例常用 3.12 |
| CUDA | `>=12.8` |
| cuDNN | `>=9.10.0` |
| 训练后端 | FSDP/FSDP2 入门，Megatron-LM 扩展大模型，VeOmni/TorchTitan/Automodel 属于进阶后端 |
| Rollout 后端 | vLLM、SGLang、TGI；v0.8.0 代码中也支持 `trtllm` rollout |

> 最稳妥的学习方式是先用 Docker 跑通官方 quickstart，再在自己的环境里逐步替换依赖。

## 方式一：Docker（推荐）

官方 Docker 镜像发布节奏比教程快，具体 tag 请以 `docker/README.md` 和 Docker Hub `verlai/verl` 为准。启动方式如下：

```bash
docker create --runtime=nvidia --gpus all --net=host \
  --shm-size="10g" --cap-add=SYS_ADMIN \
  -v $PWD:/workspace/verl \
  --name verl <image:tag> sleep infinity

docker start verl
docker exec -it verl bash
```

进入容器后安装 verl 源码：

```bash
git clone https://github.com/verl-project/verl.git
cd verl
git checkout v0.8.0
pip install --no-deps -e .
```

若需要自行安装 inference 依赖：

```bash
pip install -e ".[vllm]"
pip install -e ".[sglang]"
pip install -e ".[trtllm]"  # 仅在准备使用 TensorRT-LLM rollout 时需要
```

## 方式二：自定义 Conda 环境

适合不能使用 Docker 的机器。建议先装 inference 框架，再装 verl，避免 vLLM/SGLang 覆盖 PyTorch 版本后才发现冲突。

```bash
conda create -n verl python=3.12 -y
conda activate verl

git clone https://github.com/verl-project/verl.git
cd verl
git checkout v0.8.0

# FSDP + vLLM/SGLang + Megatron 常用依赖脚本
bash scripts/install_vllm_sglang_mcore.sh

# 只跑 FSDP 可以关闭 Megatron 依赖
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install --no-deps -e .
```

## 快速验证

```bash
python - <<'PY'
import verl
print('verl import ok')
PY

python -m verl.trainer.main_ppo --help | head
```

如果 `main_ppo --help` 能输出 Hydra 帮助信息，说明入口可用。

## 模型下载建议

官方 quickstart 使用 `Qwen/Qwen2.5-0.5B-Instruct`。如果国内访问 Hugging Face 慢，可以设置 ModelScope：

```bash
export VERL_USE_MODELSCOPE=True
```

或者提前把模型下载到本地，再把路径传给：

```bash
actor_rollout_ref.model.path=/path/to/model
critic.model.path=/path/to/model
```

## 常见安装选择

| 场景 | 建议 |
| --- | --- |
| 第一次学习 | Docker + vLLM |
| 单机 8 卡研究 | FSDP2 + vLLM/SGLang |
| 多机大模型 | Megatron + vLLM/SGLang/TRTLLM |
| AgentRL / Tool Calling | SGLang 或 vLLM async rollout，优先看 `multi_turn` 配置 |
| 视觉/多模态 | 先用官方 VLM 示例，不要从零拼配置 |
| NPU/Ascend | 参考官方 `docs/ascend_tutorial`，不要直接套 CUDA 脚本 |

## 下一步

- [02-quick-start.md](02-quick-start.md) - 跑通最小 PPO/GRPO 示例
- [04-configuration.md](04-configuration.md) - 理解 Hydra 配置树

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 00 - 什么是 verl](00-what-is-verl.md) · [目录](../README.md#catalog) · [02 - 快速上手 →](02-quick-start.md)
<!-- NAV_BOTTOM_END -->
