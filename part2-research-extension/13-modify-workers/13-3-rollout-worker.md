# 13-3 - 修改 Rollout Worker

本章介绍如何修改 Rollout Worker。

## Rollout Worker 基础

Rollout Worker 负责：
- 使用 Actor 模型生成 Response
- 计算 log probability
- 支持多种推理引擎（vLLM, SGLang, HF）

## 核心文件

| 引擎 | 文件位置 |
|------|----------|
| vLLM | `verl/workers/rollout/vllm/` |
| SGLang | `verl/workers/rollout/sglang/` |
| HF | `verl/workers/rollout/hf/` |

## 修改示例：自定义采样策略

```python
# custom_rollout.py
from verl.workers.rollout.vllm.vllm_rollout import VLLMRollout

class CustomRollout(VLLMRollout):
    """自定义 Rollout 采样策略"""

    def generate_sequences(self, prompts):
        """重写生成逻辑"""
        # 获取配置
        temperature = self.config.temperature
        top_p = self.config.top_p
        custom_top_k = getattr(self.config, 'custom_top_k', -1)

        # 自定义采样参数
        sampling_params = self.sampling_params.clone()
        if custom_top_k > 0:
            sampling_params.top_k = custom_top_k

        # 生成
        outputs = self.model.generate(
            prompts,
            sampling_params=sampling_params,
        )

        return outputs
```

## 修改示例：添加多样性奖励

```python
class DiverseRollout(VLLMRollout):
    """鼓励多样性的 Rollout"""

    def generate_sequences(self, prompts):
        # 标准生成
        outputs = super().generate_sequences(prompts)

        # 计算多样性奖励
        if hasattr(self.config, 'diversity_bonus'):
            diversity_rewards = self._compute_diversity(outputs)
            for i, output in enumerate(outputs):
                output.diversity_reward = diversity_rewards[i]

        return outputs

    def _compute_diversity(self, outputs):
        """计算输出多样性"""
        texts = [output.text for output in outputs]
        # 简单的 n-gram 多样性
        from collections import Counter

        rewards = []
        for text in texts:
            ngrams = self._get_ngrams(text, n=2)
            unique_ratio = len(set(ngrams)) / max(len(ngrams), 1)
            rewards.append(unique_ratio)

        return rewards

    def _get_ngrams(self, text, n=2):
        tokens = text.split()
        return tuple(zip(*[tokens[i:] for i in range(n)]))
```

## 多轮 Rollout 修改

```python
class CustomMultiturnRollout(SGLangRollout):
    """自定义多轮 Rollout"""

    def generate_multiturn(self, data, env):
        """多轮生成逻辑"""
        conversations = []
        rewards = []

        for turn in range(self.config.max_turns):
            # 生成响应
            response = self.generate_single_turn(data)

            # 环境交互
            obs, reward, done, info = env.step(response)

            conversations.append(response)
            rewards.append(reward)

            if done:
                break

        return conversations, rewards
```

## 下一步

- [14-paper-reproduction](../../14-paper-reproduction/) - 论文复现
