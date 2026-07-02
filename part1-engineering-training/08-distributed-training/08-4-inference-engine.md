# 08-4 - 推理引擎配置

<!-- NAV_START -->
> 导航： [上一篇：08-3 - 训练后端选择](08-3-backends.md) | [返回目录](../../README.md#完整目录) | [下一篇：09-1 - 数据预处理](../09-data-and-reward/09-1-data-preprocess.md)
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
| 多轮工具不触发 | 检查 `multi_turn.enable`、工具路径、`agent.default_agent_loop` |
| TRTLLM import/build 失败 | 使用官方 Docker 或确认 `pip install -e .[trtllm]` 完成 |

---

<!-- NAV_BOTTOM_START -->
> 导航： [上一篇：08-3 - 训练后端选择](08-3-backends.md) | [返回目录](../../README.md#完整目录) | [下一篇：09-1 - 数据预处理](../09-data-and-reward/09-1-data-preprocess.md)
<!-- NAV_BOTTOM_END -->
