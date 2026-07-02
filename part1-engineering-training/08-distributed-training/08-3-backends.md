# 08-3 - 训练后端选择

<!-- NAV_START -->
> 阅读： [← 08-2 - 多机训练](08-2-multi-node.md) · [目录](../../README.md#catalog) · [08-4 - 推理引擎配置 →](08-4-inference-engine.md)
<!-- NAV_END -->

verl v0.8.0 把训练后端收敛到统一 worker / model engine 体系。学习时不要一开始就追最大规模，先按“能跑 -> 能扩 -> 能调”的顺序选后端。

## 后端对比

| 后端 | 适合场景 | 典型开关 |
| --- | --- | --- |
| FSDP / FSDP2 | 单机或中小规模研究、入门首选 | `actor_rollout_ref.actor.strategy=fsdp2` |
| Megatron-LM | 大模型、多机、TP/PP/CP/EP | `model_engine=megatron` |
| VeOmni | MoE、多模态、NPU/GPU 进阶场景 | `model_engine=veomni` |
| TorchTitan | TorchTitan 生态实验 | `model_engine=torchtitan` |
| Automodel | NeMo Automodel 后端，v0.8.0 主要参考 SFT 示例 | `engine=automodel` |

训练后端和 rollout 后端是两件事。`model_engine`、actor/ref/critic 的 strategy 解决“参数如何训练”；`actor_rollout_ref.rollout.name` 解决“样本如何生成”。不要因为用了 Megatron 就默认 rollout 也变成 Megatron，也不要把 SFT 的 `engine=automodel` 写进 PPO 的 `model_engine`。

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

大模型场景不应只改 actor，还要同步 ref/critic 的并行配置：

```bash
actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
critic.megatron.tensor_model_parallel_size=2
```

## VeOmni / TorchTitan / Automodel

VeOmni / TorchTitan 已进入 v0.8.0 的 PPO `model_engine` 配置；Automodel 有 engine 实现和 SFT 示例，常见入口是 `engine=automodel`，不要把它当成主 PPO 生成配置里的 `model_engine=automodel`。建议原则：

1. 先从 `examples/grpo_trainer/*veomni*.sh`、`examples/sft/*automodel*.sh` 复制可运行脚本。
2. 不要手写完整配置树；用官方脚本中的 env vars 调模型、节点、并行度。
3. 只在模型族、硬件和依赖支持都确认后使用。

## 训练后端 × rollout 后端

更实际的选择通常是组合选择：

| 组合 | 适合场景 | 说明 |
| --- | --- | --- |
| FSDP2 + vLLM | 入门、GRPO、常规 PPO | 最推荐的学习组合，配置量少 |
| FSDP2 + SGLang | AgentRL、多轮、工具调用 | rollout 侧更适合复杂 serving 能力 |
| Megatron + vLLM/SGLang | 30B+ dense、多机训练 | 训练并行交给 Megatron，生成仍由 rollout engine 负责 |
| Megatron + TRTLLM | NVIDIA 高性能推理栈探索 | 依赖、镜像和构建要求更高 |
| VeOmni + vLLM/SGLang | MoE、多模态、NPU/GPU 进阶 | 直接从官方 veomni 示例复制起步 |
| TorchTitan + vLLM | TorchTitan 生态实验 | 适合跟随官方示例验证，不建议初学者手写 |
| Automodel + SFT | NeMo Automodel SFT | 主要看 SFT 的 `engine=automodel` 示例 |

组合选定后，还要决定 Trainer 数据流：常规情况用 `main_ppo`；遇到 controller 数据传输瓶颈或多输出 agent loop，再看 [08-5 Trainer 数据流](08-5-trainer-dataflow.md) 中的 `main_ppo_sync` 和 fully async。

## 选择建议

| 当前目标 | 推荐 |
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
- 如果性能瓶颈不在训练后端，而在 rollout / controller 数据流，不要盲目加 TP/PP，先看 [08-4](08-4-inference-engine.md) 和 [08-5](08-5-trainer-dataflow.md)。

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 08-2 - 多机训练](08-2-multi-node.md) · [目录](../../README.md#catalog) · [08-4 - 推理引擎配置 →](08-4-inference-engine.md)
<!-- NAV_BOTTOM_END -->
