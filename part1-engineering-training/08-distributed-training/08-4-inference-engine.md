# 08-4 - 推理引擎配置

<!-- NAV_START -->
> 阅读： [← 08-3 - 训练后端选择](08-3-backends.md) · [目录](../../README.md#catalog) · [08-5 - Trainer 数据流与同步/异步训练 →](08-5-trainer-dataflow.md)
<!-- NAV_END -->

Rollout 后端负责生成训练样本，是 verl 性能和稳定性的关键。v0.8.0 的配置入口是：

```bash
actor_rollout_ref.rollout.name=<hf|vllm|sglang|trtllm>
```

## 后端对比

| 后端 | 适合场景 | 备注 |
| --- | --- | --- |
| `hf` | 调试、最小依赖、小模型 | 慢，但排查问题直接 |
| `vllm` | 通用高吞吐 rollout | 入门首选，FSDP 常用 |
| `sglang` | 多轮、工具、复杂 serving 能力 | AgentRL 常用 |
| `trtllm` | NVIDIA TensorRT-LLM 高性能 rollout | 依赖和镜像要求更高 |

## vLLM

```bash
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.max_num_batched_tokens=8192
```

传递 vLLM 专属参数时使用 `engine_kwargs`，新键要加 `+`：

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[1,8,16,32,64,128]'
```

## SGLang

```bash
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
```

AgentRL / Tool Calling 常配：

```bash
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.multi_turn.tool_config_path=/path/to/tool_config.yaml \
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

SGLang 也有 Prefill-Decode disaggregation 相关配置：

```bash
actor_rollout_ref.rollout.disaggregation.enabled=True
```

仅在理解集群放置和 SGLang 版本要求时使用。

## TRTLLM

v0.8.0 代码和 worker 文档支持 TensorRT-LLM rollout：

```bash
actor_rollout_ref.rollout.name=trtllm
```

通常还需要安装 `.[trtllm]` 或使用官方 TensorRT-LLM Docker，并通过 `engine_kwargs.trtllm.*` 传递 TensorRT-LLM 参数：

```bash
+actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_timeout_iters=32 \
+actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_max_tokens_ratio=0.5
```

不要把旧教程里的 `trtllm-build --model-path ...` 当成 verl 必需步骤；具体构建与 serving 参数以官方 `docs/workers/trtllm_worker.rst` 和示例脚本为准。

## HF

```bash
actor_rollout_ref.rollout.name=hf \
actor_rollout_ref.rollout.n=1
```

HF rollout 主要用于调试，不建议作为高吞吐训练配置。

## sync、async 与 standalone rollout

v0.8.0 的 `rollout.yaml` 里有几个容易混淆的键：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `actor_rollout_ref.rollout.mode` | `async` | rollout 后端模式；源码注释是 `sync: LLM, async: AsyncLLM` |
| `actor_rollout_ref.rollout.nnodes` | `0` | 独立 rollout server 节点数；one-step-off / fully async 训练需要大于 0 |
| `actor_rollout_ref.rollout.n_gpus_per_node` | `${trainer.n_gpus_per_node}` | 独立 rollout server 每节点 GPU 数 |
| `actor_rollout_ref.rollout.calculate_log_probs` | `False` | 是否在 rollout 侧计算 log probs；fully async 训练会打开 |
| `actor_rollout_ref.rollout.top_k` | `-1` | vLLM 使用 `-1`，HF rollout 通常用 `0` |

最重要的区别是：`actor_rollout_ref.rollout.mode=async` 不等于 fully async training。它只是 rollout engine 使用 AsyncLLM。真正的 fully async 路径在 `verl.experimental.fully_async_policy`，还会引入 top-level `rollout.*` 和 `async_training.*` 配置。

## 参数层级不要混

| 层级 | 例子 | 作用 |
| --- | --- | --- |
| `trainer.*` | `trainer.nnodes`、`trainer.n_gpus_per_node` | 训练侧资源 |
| `actor_rollout_ref.rollout.*` | `name`、`mode`、`tensor_model_parallel_size`、`gpu_memory_utilization` | actor 关联的 rollout 引擎配置 |
| top-level `rollout.*` | `rollout.nnodes`、`rollout.total_rollout_steps` | fully async 配置里的独立 rollout 服务 |
| `async_training.*` | `staleness_threshold`、`trigger_parameter_sync_step` | fully async 的训练/rollout 解耦控制 |

如果只是普通 PPO/GRPO，不要急着写 top-level `rollout.nnodes`。当你开始使用 fully async 时，再跟着 [08-5 Trainer 数据流](08-5-trainer-dataflow.md) 区分这些层级。

## 性能调参顺序

1. 先确认生成正确：`rollout.n=1`，小 batch。
2. 再增大 `rollout.n`，观察 reward 方差和吞吐。
3. 调 `tensor_model_parallel_size`，让单次生成不 OOM。
4. 调 `gpu_memory_utilization`，给训练和 rollout 留出余量。
5. 最后再加 engine-specific kwargs。

## 常见错误

| 现象 | 处理 |
| --- | --- |
| rollout OOM | 降低 `n`、`max_num_batched_tokens`、`gpu_memory_utilization` |
| vLLM/SGLang 参数不生效 | 检查是否加了 `+actor_rollout_ref.rollout.engine_kwargs.<backend>.*` |
| 多轮工具不触发 | 检查 `multi_turn.enable`、工具路径、`actor_rollout_ref.rollout.agent.default_agent_loop` |
| TRTLLM import/build 失败 | 使用官方 Docker 或确认 `pip install -e .[trtllm]` 完成 |
| fully async 没有 rollout 资源 | 检查 top-level `rollout.nnodes` / `rollout.n_gpus_per_node`，不是只看 `trainer.*` |

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 08-3 - 训练后端选择](08-3-backends.md) · [目录](../../README.md#catalog) · [08-5 - Trainer 数据流与同步/异步训练 →](08-5-trainer-dataflow.md)
<!-- NAV_BOTTOM_END -->
