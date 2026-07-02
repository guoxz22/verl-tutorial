# 09-3 - 奖励配置

<!-- NAV_START -->
> 阅读： [← 09-2 - 自定义数据集](09-2-custom-dataset.md) · [目录](../../README.md#完整目录) · [10-1 - 训练监控 →](../10-operations/10-1-monitoring.md)
<!-- NAV_END -->

v0.8.0 推荐把奖励相关配置统一写在 `reward.*` 下。旧脚本里可能还能看到顶层 `custom_reward_function.*` 或 `reward_model.*`，那是兼容字段；新教程统一使用当前 `reward/reward.yaml` 的结构。

## 三种奖励来源

| 类型 | 用途 | 配置入口 |
| --- | --- | --- |
| 函数奖励 | 数学、代码、格式、可验证任务 | `reward.custom_reward_function.*` |
| Reward Manager | 控制如何批量计算/聚合奖励 | `reward.reward_manager.*` |
| Reward Model | 使用模型打分，支持 discriminative/generative RM | `reward.reward_model.*` |

## 自定义函数奖励

```python
# reward_fn.py
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if ground_truth is None:
        return 0.0
    return 1.0 if str(ground_truth).strip() in solution_str else 0.0
```

启动：

```bash
python -m verl.trainer.main_ppo \
  reward.custom_reward_function.path=$PWD/reward_fn.py \
  reward.custom_reward_function.name=compute_score
```

函数参数应包含：

- `data_source`：数据来源，默认来自 `data.reward_fn_key=data_source`。
- `solution_str`：模型生成文本。
- `ground_truth`：parquet 中 `reward_model.ground_truth`。
- `extra_info`：预处理脚本塞入的额外字段。

函数可以返回 float，也可以返回 dict；返回 dict 时应包含 `score`。

## Reward Manager

默认 manager 是 `naive`：

```bash
reward.reward_manager.name=naive
```

DAPO / GDPO 等算法会使用自己的 manager：

```bash
reward.reward_manager.name=dapo
reward.reward_manager.name=gdpo
```

自定义 manager 见 [12-3-reward-manager.md](../../part2-research-extension/12-custom-algorithm/12-3-reward-manager.md)。
v0.8.0 的 PPO reward loop 默认从 `verl.experimental.reward_loop.reward_manager` 注册表查找 manager；独立文件里的 manager 可以用 `reward.reward_manager.source=importlib` 加载。

## Reward Model

```bash
python -m verl.trainer.main_ppo \
  reward.reward_model.enable=True \
  reward.reward_model.model_path=$HOME/models/reward_model \
  reward.reward_model.rollout.name=vllm \
  reward.reward_model.rollout.tensor_model_parallel_size=1 \
  reward.reward_model.rollout.gpu_memory_utilization=0.5
```

如果 Reward Model 单独占资源池：

```bash
reward.reward_model.enable_resource_pool=True \
reward.reward_model.nnodes=1 \
reward.reward_model.n_gpus_per_node=2
```

## 函数奖励 + Reward Model

Generative Reward Model 或“模型先打分、函数再解析”的场景会同时使用两者：

```bash
reward.reward_model.enable=True \
reward.reward_model.model_path=$HOME/models/genrm \
reward.custom_reward_function.path=$PWD/parse_genrm_score.py \
reward.custom_reward_function.name=compute_score
```

## 并行度

```bash
reward.num_workers=8
```

这个值控制 reward manager 并行 worker 数。工具调用或代码执行型 reward 如果很慢，优先调这个值和外部 sandbox 并发限制。

## 数据字段要求

默认 reward manager 会从每条样本中读取：

```python
{
  "data_source": "openai/gsm8k",
  "reward_model": {"style": "rule", "ground_truth": "42"},
  "extra_info": {"split": "train", "index": 0}
}
```

若数据源字段不是 `data_source`：

```bash
data.reward_fn_key=my_source_key
```

## 常见错误

| 错误写法 | v0.8.0 推荐写法 |
| --- | --- |
| `reward.reward_function.path` | `reward.custom_reward_function.path` |
| `reward.reward_function.name` | `reward.custom_reward_function.name` |
| `reward.reward_model.path` | `reward.reward_model.model_path` |
| `reward.reward_manager=my_rm` | `reward.reward_manager.name=my_rm` |

---

<!-- NAV_BOTTOM_START -->
> 阅读： [← 09-2 - 自定义数据集](09-2-custom-dataset.md) · [目录](../../README.md#完整目录) · [10-1 - 训练监控 →](../10-operations/10-1-monitoring.md)
<!-- NAV_BOTTOM_END -->
