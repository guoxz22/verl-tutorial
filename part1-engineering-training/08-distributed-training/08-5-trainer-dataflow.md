# 08-5 - Trainer 数据流与同步/异步训练

<!-- NAV_START -->
> 阅读： [← 08-4 - 推理引擎配置](08-4-inference-engine.md) · [目录](../../README.md#catalog) · [09-1 - 数据预处理 →](../09-data-and-reward/09-1-data-preprocess.md)
<!-- NAV_END -->

前面两章分别回答了“训练用什么后端”和“rollout 用什么推理引擎”。本章回答第三个问题：同样是 PPO/GRPO 训练，数据到底走哪条 Trainer 路径？

v0.8.0 里至少要区分三条路径：

| 路径 | 入口 | 适合场景 | 一句话理解 |
| --- | --- | --- | --- |
| 标准单控制器 | `python -m verl.trainer.main_ppo` | 入门、算法实验、常规 PPO/GRPO | 最稳定、最适合学习的主路径 |
| TransferQueue 同步路径 | `python -m verl.trainer.main_ppo_sync` | controller 数据传输成为瓶颈、agent loop 多输出、不同 prompt 需要不同 `n` | 仍是同步 PPO，但数据不再都挤在 RayPPOTrainer 里 |
| fully async 实验路径 | `python -m verl.experimental.fully_async_policy.fully_async_main` | 训练与 rollout 解耦、独立 rollout server、多机高吞吐探索 | trainer 和 rollouter 并行推进，用参数同步控制 staleness |

不要把 `actor_rollout_ref.rollout.mode=async` 和 fully async training 混在一起。前者只是 rollout 后端使用 AsyncLLM；后者是 `verl.experimental.fully_async_policy` 下的一整套训练/rollout 解耦流程。

## 标准路径：main_ppo

大多数读者应该先掌握这一条。PPO、GRPO、RLOO、ReMax、DAPO 类配置通常都从 `main_ppo` 进入：

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  trainer.n_gpus_per_node=8
```

可以把它理解为一条清晰的教学流水线：

```text
DataLoader
  -> rollout 生成 response
  -> actor/ref/critic/reward 计算 logprob、value、reward
  -> advantage 与 return
  -> actor/critic update
  -> metrics、validation、checkpoint
```

这条路径的优点是概念集中、配置最少、调试最直接。只要不是明确遇到 controller 侧数据搬运瓶颈，或者正在做训练/rollout 解耦实验，就不要过早离开 `main_ppo`。

## 同步增强路径：main_ppo_sync + TransferQueue

`main_ppo_sync.py` 的文件注释把它定义为 synchronous PPO trainer with colocated actor and rollout。它和 `main_ppo.py` 的差别不是“变成 fully async”，而是引入 TransferQueue 与 ReplayBuffer 来改造数据传输：

1. 用 TransferQueue 做 zero-padding / zero-copy data transfer。
2. 用 ReplayBuffer 从 TransferQueue 采样。
3. 支持每个 prompt 不同的 `n` 采样。
4. 支持每个 agent loop 产生多个输出。

启动前需要安装 TransferQueue。v0.8.0 源码里的缺失提示是：

```bash
pip install TransferQueue==0.1.6
```

最小入口仍然沿用 `ppo_trainer` 配置族：

```bash
python -m verl.trainer.main_ppo_sync \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8
```

它的核心变化可以这样看：

```text
prompt / uid / meta
  -> agent loop 或 rollout worker 写入 TransferQueue
  -> ReplayBuffer / sampler 取出可训练样本
  -> logprob、ref_log_prob、value、reward、advantage 等字段写回 TransferQueue
  -> trainer 在字段齐备后消费并更新模型
```

TransferQueue 的价值不在于多一个缓存名字，而在于把样本级元数据、生产状态、消费状态和数据平面分开管理。官方 v0.8.0 文档里给它的定位是缓解单控制器 `RayPPOTrainer` 的数据传输瓶颈，尤其是多模态、大规模、多输出 agent loop 这类容易让 `DataProto` 往返变重的场景。

## fully async 路径

fully async 位于 `verl/experimental/fully_async_policy/`，入口是：

```bash
python -m verl.experimental.fully_async_policy.fully_async_main
```

默认配置文件是 `fully_async_ppo_trainer.yaml`；Megatron 路径使用：

```bash
python -m verl.experimental.fully_async_policy.fully_async_main \
  --config-path=config \
  --config-name=fully_async_ppo_megatron_trainer.yaml
```

它的系统形态和标准 PPO 不同：

```text
rollout server 持续生成样本
  -> message queue / rollout buffer
  -> trainer 按 require_batches 消费
  -> checkpoint engine 同步新参数
  -> rollouter 带着受控 staleness 继续生成
```

关键配置通常集中在三组：

| 配置组 | 常用键 | 作用 |
| --- | --- | --- |
| rollout 服务资源 | `rollout.nnodes`、`rollout.n_gpus_per_node`、`rollout.total_rollout_steps` | 给独立 rollout server 分配资源和生成步数 |
| async 控制 | `async_training.staleness_threshold`、`trigger_parameter_sync_step`、`require_batches`、`partial_rollout` | 控制样本陈旧度、参数同步频率和中断后是否恢复生成 |
| logprob 对齐 | `actor_rollout_ref.rollout.calculate_log_probs=True`、`actor_rollout_ref.actor.use_rollout_log_probs=True` | fully async 训练需要使用 rollout 侧 log probs |

一个能帮助定位参数层级的命令骨架是：

```bash
python -m verl.experimental.fully_async_policy.fully_async_main \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.actor.use_rollout_log_probs=True \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node=8 \
  rollout.total_rollout_steps=100 \
  async_training.staleness_threshold=0.5 \
  async_training.trigger_parameter_sync_step=4 \
  async_training.require_batches=1 \
  async_training.partial_rollout=True
```

这里的 `rollout.nnodes` 是 fully async 配置里的独立 rollout 服务资源；普通 colocated PPO 里常见的是 `trainer.nnodes` 和 `actor_rollout_ref.rollout.*`。v0.8.0 的 `fully_async_main.py` 会把 top-level `rollout.nnodes` 写回 `actor_rollout_ref.rollout.nnodes`，用于启动独立 rollout server。

## 何时选哪条路径

| 你的目标 | 建议 |
| --- | --- |
| 第一次跑通 PPO/GRPO | `main_ppo` |
| 改 advantage、policy loss、reward manager | `main_ppo`，降低系统变量 |
| Agent loop 产生多个输出，或者每个 prompt 的采样数不一样 | `main_ppo_sync` |
| 多模态/大规模训练里 controller 数据搬运明显吃紧 | 评估 `main_ppo_sync` + TransferQueue |
| 想把 rollout server 和 trainer 资源彻底拆开 | `fully_async_main` |
| Megatron + fully async | 使用 `fully_async_ppo_megatron_trainer.yaml` 起步 |

## 常见误区

| 误区 | 正确理解 |
| --- | --- |
| `rollout.mode=async` 就是 fully async | 不是。它只表示 rollout 后端使用 AsyncLLM |
| `main_ppo_sync` 会自动变成异步训练 | 不是。源码注释明确是 synchronous PPO trainer |
| fully async 不需要 rollout log probs | v0.8.0 fully async 配置要求 `calculate_log_probs=True`，actor 使用 rollout log probs |
| 普通训练也要设置 top-level `rollout.nnodes` | 不需要。普通路径主要看 `trainer.*` 和 `actor_rollout_ref.rollout.*` |
| `actor_rollout_ref.rollout.nnodes=0` 总是错误 | 普通 colocated 路径可以是 0；one-step-off / fully async 独立 rollout server 才要求大于 0 |
| TransferQueue 是所有 v0.8.0 训练的默认数据系统 | 不是。它是 `main_ppo_sync` 等路径使用的增强数据流 |

## 排错提示

- 找不到 `transfer_queue`：按源码提示安装 `pip install TransferQueue==0.1.6`。
- fully async 启动时报缺少 `async_training`：确认使用的是 `fully_async_ppo_trainer.yaml` 或 `fully_async_ppo_megatron_trainer.yaml`。
- fully async rollout server 没有资源：检查 `rollout.nnodes`、`rollout.n_gpus_per_node` 和 Ray 集群资源。
- rollout log probs 缺失：检查 `actor_rollout_ref.rollout.calculate_log_probs=True` 和 `actor_rollout_ref.actor.use_rollout_log_probs=True`。
- vLLM fully async 示例异常：优先对照官方 shell 示例；v0.8.0 的 vLLM async 示例会设置 `VLLM_USE_V1=1`。

## 下一步

学完本章后，08 章的选择顺序就完整了：

1. [08-3](08-3-backends.md)：先选训练后端。
2. [08-4](08-4-inference-engine.md)：再选 rollout 后端。
3. 本章：最后决定 trainer 数据流。

接下来进入 [09-1 数据预处理](../09-data-and-reward/09-1-data-preprocess.md)，把数据格式和 reward 字段准备好。

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 08-4 - 推理引擎配置](08-4-inference-engine.md) · [目录](../../README.md#catalog) · [09-1 - 数据预处理 →](../09-data-and-reward/09-1-data-preprocess.md)
<!-- NAV_BOTTOM_END -->
