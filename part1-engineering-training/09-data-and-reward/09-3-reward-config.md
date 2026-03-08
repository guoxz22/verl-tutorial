# 09-3 - 奖励配置

本章介绍如何配置奖励函数。

## 奖励类型

| 类型 | 说明 | 配置 |
|------|------|------|
| 函数奖励 | 基于规则的奖励 | 默认，需自定义 |
| 模型奖励 | 使用 Reward Model | `reward.reward_model.enable=True` |

## 函数奖励

### 默认奖励函数

verl 默认使用函数奖励，需要实现 `reward_function`：

```python
# reward_function.py
def reward_function(data_source, solution_str, ground_truth, **kwargs):
    """
    计算奖励值

    Args:
        data_source: 原始输入
        solution_str: 模型生成的回答
        ground_truth: 标准答案

    Returns:
        float: 奖励值
    """
    # 示例：数学答案匹配
    try:
        # 提取数字答案
        predicted = extract_answer(solution_str)
        expected = float(ground_truth)

        if abs(predicted - expected) < 1e-6:
            return 1.0  # 正确
        else:
            return 0.0  # 错误
    except:
        return 0.0  # 格式错误

def extract_answer(text):
    """从文本中提取答案"""
    # 实现答案提取逻辑
    import re
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(',', ''))
    return None
```

### 配置使用

```bash
python -m verl.trainer.main_ppo \
    reward.reward_function.path=./reward_function.py \
    reward.reward_function.name=reward_function \
    ...
```

## 模型奖励

### 启用 Reward Model

```bash
reward.reward_model.enable=True
reward.reward_model.model_path=sfairXC/FsfairX-LLaMA3-RM-v0.1
```

### 完整配置

```bash
python -m verl.trainer.main_ppo \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$HOME/models/FsfairX-LLaMA3-RM-v0.1 \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.gpu_memory_utilization=0.8 \
    reward.reward_model.rollout.prompt_length=2048 \
    reward.reward_model.rollout.response_length=1024 \
    ...
```

### 资源池配置

将 Reward Model 放在独立 GPU 上：

```bash
reward.reward_model.enable_resource_pool=True
reward.reward_model.nnodes=1
reward.reward_model.n_gpus_per_node=2
```

## 混合奖励

可以同时使用函数奖励和模型奖励：

```bash
# 模型奖励作为 baseline，函数奖励作为主要信号
reward.reward_model.enable=True
reward.reward_function.path=./reward_function.py
```

## 奖励计算配置

```bash
# 奖励计算并行度
reward.num_workers=8

# 奖励归一化
algorithm.norm_adv_by_std_in_grpo=True

# KL 惩罚（在奖励中）
algorithm.use_kl_in_reward=True
algorithm.kl_ctrl.kl_coef=0.01
```

## 下一步

- [10-operations](../10-operations/) - 运维与监控
