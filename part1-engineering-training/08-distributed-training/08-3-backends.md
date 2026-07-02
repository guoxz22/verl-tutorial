# 08-3 - 训练后端选择

verl v0.8.0 把训练后端收敛到统一 worker / model engine 体系。学习时不要一开始就追最大规模，先按“能跑 -> 能扩 -> 能调”的顺序选后端。

## 后端对比

| 后端 | 适合场景 | 典型开关 |
| --- | --- | --- |
| FSDP / FSDP2 | 单机或中小规模研究、入门首选 | `actor_rollout_ref.actor.strategy=fsdp2` |
| Megatron-LM | 大模型、多机、TP/PP/CP/EP | `model_engine=megatron` |
| VeOmni | MoE、多模态、NPU/GPU 进阶场景 | `model_engine=veomni` |
| TorchTitan | TorchTitan 生态实验 | `model_engine=torchtitan` |
| Automodel | 新统一 engine 路径，适合跟官方示例 | `engine=automodel` 或 `model_engine=automodel` 相关示例 |

## FSDP / FSDP2

默认 DP/FSDP 路径最适合调算法和数据。

```bash
python -m verl.trainer.main_ppo \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```

常用调参：

| 参数 | 作用 |
| --- | --- |
| `ppo_micro_batch_size_per_gpu` | 限制每张卡一次 update 的样本数 |
| `ppo_max_token_len_per_gpu` | 动态 batch 时限制每张卡 token 数 |
| `fsdp_config.param_offload` | 参数 offload，省显存但降速 |
| `fsdp_config.optimizer_offload` | 优化器 offload，省显存但降速 |
| `fsdp_config.ulysses_sequence_parallel_size` | 序列并行 |

## Megatron

Megatron 适合 30B、70B、MoE 或多机规模。官方示例通常使用：

```bash
python -m verl.trainer.main_ppo \
  model_engine=megatron \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.context_parallel_size=1 \
  actor_rollout_ref.actor.megatron.use_mbridge=True
```

大模型不要只改 actor，还要同步 ref/critic 的并行配置：

```bash
actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
critic.megatron.tensor_model_parallel_size=2
```

## VeOmni / TorchTitan / Automodel

这些后端在 v0.8.0 中已经进入官方 examples 和生成配置，但使用门槛高于 FSDP。建议原则：

1. 先从 `examples/grpo_trainer/*veomni*.sh`、`examples/sft/*automodel*.sh` 复制可运行脚本。
2. 不要手写完整配置树；用官方脚本中的 env vars 调模型、节点、并行度。
3. 只在你确认模型族、硬件和依赖支持时使用。

## 选择建议

| 你现在的目标 | 推荐 |
| --- | --- |
| 学习 verl / 跑 0.5B-8B | FSDP2 + vLLM |
| 做算法实验 | FSDP2，少改后端 |
| 训练 30B+ dense | Megatron 或 VeOmni |
| MoE / router replay | Megatron / VeOmni，直接参考官方脚本 |
| NPU | MindSpeed / VeOmni，参考官方 Ascend 文档 |

## 配置检查清单

- `trainer.n_gpus_per_node * trainer.nnodes` 与 Ray 集群资源一致。
- rollout TP 不要超过可用 GPU，并保证和训练 worker 放置不冲突。
- actor、ref、critic 的策略/并行配置要匹配任务需要。
- 优先使用 `_per_gpu` 后缀的 micro batch 配置。
- 每次换后端，先跑 1-2 step smoke test。
