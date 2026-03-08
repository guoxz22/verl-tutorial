# 12-3 - 自定义 Reward Manager

本章介绍如何实现自定义 Reward Manager。

## Reward Manager 基础

Reward Manager 负责计算每个 Response 的奖励值。verl 提供了抽象基类：

```python
from verl.workers.reward_manager import AbstractRewardManager
```

## 实现自定义 Reward Manager

### 基本模板

```python
# custom_reward_manager.py
import torch
from verl.workers.reward_manager import AbstractRewardManager, register_reward_manager
from verl import DataProto

@register_reward_manager("my_custom_rm")
class MyCustomRewardManager(AbstractRewardManager):
    """
    自定义 Reward Manager
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def __call__(self, data: DataProto) -> DataProto:
        """
        计算奖励

        Args:
            data: DataProto 对象，包含：
                - data.batch['input_ids']: 输入 token IDs
                - data.batch['attention_mask']: 注意力掩码
                - data.non_tensor_batch['data_source']: 原始 prompt
                - 其他自定义字段

        Returns:
            data: 更新后的 DataProto，包含：
                - data.batch['rewards']: 奖励张量
        """
        # 获取原始 prompt 和生成的 response
        prompts = data.non_tensor_batch.get('data_source', [])
        input_ids = data.batch['input_ids']

        # 解码 response
        responses = self._decode_responses(input_ids, self.tokenizer)

        # 计算奖励
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self._compute_single_reward(prompt, response)
            rewards.append(reward)

        # 存储奖励
        data.batch['rewards'] = torch.tensor(rewards, dtype=torch.float32)

        return data

    def _compute_single_reward(self, prompt: str, response: str) -> float:
        """
        计算单个样本的奖励

        Args:
            prompt: 输入提示词
            response: 模型生成的回答

        Returns:
            float: 奖励值
        """
        # 自定义奖励计算逻辑
        # 示例：简单的长度奖励
        reward = len(response) / 100.0
        return reward

    def _decode_responses(self, input_ids, tokenizer):
        """解码 response"""
        responses = []
        for ids in input_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            responses.append(text)
        return responses
```

### 数学验证奖励（示例）

```python
@register_reward_manager("math_reward")
class MathRewardManager(AbstractRewardManager):
    """数学答案验证奖励"""

    def __init__(self, config):
        super().__init__(config)
        # 可以加载额外的验证工具
        pass

    def __call__(self, data: DataProto) -> DataProto:
        prompts = data.non_tensor_batch.get('data_source', [])
        ground_truths = data.non_tensor_batch.get('ground_truth', [])

        input_ids = data.batch['input_ids']
        responses = self._decode(input_ids)

        rewards = []
        for prompt, response, gt in zip(prompts, responses, ground_truths):
            # 提取答案
            predicted = self._extract_answer(response)
            expected = self._parse_ground_truth(gt)

            # 验证
            if self._verify_answer(predicted, expected):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        data.batch['rewards'] = torch.tensor(rewards, dtype=torch.float32)
        return data

    def _extract_answer(self, text: str) -> str:
        """从文本中提取答案"""
        import re
        # 匹配 #### 后的答案
        match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '')
        return None

    def _parse_ground_truth(self, gt: str) -> str:
        """解析标准答案"""
        return gt.strip()

    def _verify_answer(self, predicted, expected) -> bool:
        """验证答案是否正确"""
        if predicted is None or expected is None:
            return False
        try:
            p = float(predicted)
            e = float(expected)
            return abs(p - e) < 1e-6
        except:
            return predicted.strip() == expected.strip()
```

### 代码执行奖励（示例）

```python
@register_reward_manager("code_reward")
class CodeRewardManager(AbstractRewardManager):
    """代码执行验证奖励"""

    def __init__(self, config):
        super().__init__(config)
        self.timeout = config.get('timeout', 10)

    def __call__(self, data: DataProto) -> DataProto:
        prompts = data.non_tensor_batch.get('data_source', [])
        input_ids = data.batch['input_ids']
        responses = self._decode(input_ids)

        rewards = []
        for prompt, response in zip(prompts, responses):
            code = self._extract_code(response)
            if code:
                reward = self._execute_code(code)
            else:
                reward = 0.0
            rewards.append(reward)

        data.batch['rewards'] = torch.tensor(rewards, dtype=torch.float32)
        return data

    def _extract_code(self, text: str) -> str:
        """提取代码块"""
        import re
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1)
        return None

    def _execute_code(self, code: str) -> float:
        """执行代码并返回奖励"""
        try:
            # 安全执行代码
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            return 1.0  # 执行成功
        except Exception as e:
            return 0.0  # 执行失败
```

## 配置使用

```bash
python -m verl.trainer.main_ppo \
    reward.reward_manager=my_custom_rm \
    ...
```

## 下一步

- [12-4-full-example.md](12-4-full-example.md) - 完整算法实现示例
