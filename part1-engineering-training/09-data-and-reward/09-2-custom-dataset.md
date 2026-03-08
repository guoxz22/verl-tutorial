# 09-2 - 自定义数据集

本章介绍如何创建和使用自定义数据集。

## 自定义数据集类

### 创建自定义类

```python
# custom_dataset.py
from verl.utils.dataset.rl_dataset import RLHFDataset

class MyCustomDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, processor, config, **kwargs):
        super().__init__(data_files, tokenizer, processor, config, **kwargs)

    def __getitem__(self, item):
        # 自定义数据处理逻辑
        data = self.data[item]

        # 返回格式
        return {
            'input_ids': ...,
            'attention_mask': ...,
            'position_ids': ...,
            'data_source': ...,
        }
```

### 注册自定义类

```bash
python -m verl.trainer.main_ppo \
    data.custom_cls.path=/path/to/custom_dataset.py \
    data.custom_cls.name=MyCustomDataset \
    ...
```

## 完整示例

### 数学数据集

```python
# math_custom_dataset.py
import pandas as pd
import pyarrow.parquet as pq
from verl.utils.dataset.rl_dataset import RLHFDataset

class MathCustomDataset(RLHFDataset):
    """自定义数学数据集，包含答案解析"""

    def __init__(self, data_files, tokenizer, processor, config, **kwargs):
        super().__init__(data_files, tokenizer, processor, config, **kwargs)

        # 加载额外数据
        self.answers = self._load_answers()

    def _load_answers(self):
        # 加载标准答案
        answers = {}
        for item in self.data.to_pandas().itertuples():
            answers[item.uid] = item.answer
        return answers

    def __getitem__(self, item):
        data = super().__getitem__(item)

        # 添加额外字段
        uid = self.data[item]['uid']
        data['ground_truth'] = self.answers.get(uid, '')

        return data
```

### 使用配置

```bash
python -m verl.trainer.main_ppo \
    data.train_files=./data/math_train.parquet \
    data.custom_cls.path=./math_custom_dataset.py \
    data.custom_cls.name=MathCustomDataset \
    ...
```

## 数据字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `data_source` / `prompt` | str | 输入提示词 |
| `input_ids` | tensor | Token IDs |
| `attention_mask` | tensor | 注意力掩码 |
| `position_ids` | tensor | 位置 IDs |
| `images` | list | 图像数据（多模态） |
| `ground_truth` | str | 标准答案（可选） |

## 下一步

- [09-3-reward-config.md](09-3-reward-config.md) - 奖励配置
