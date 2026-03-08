# 09-1 - 数据预处理

本章介绍如何准备 verl 训练所需的数据。

## 数据格式

### 基本格式

verl 使用 Parquet 格式存储数据：

```python
import pandas as pd
import pyarrow.parquet as pq

# 基本数据结构
data = {
    'data_source': [  # 或 'prompt'
        'What is 2+2?',
        'Calculate 10*5.',
    ]
}

df = pd.DataFrame(data)
df.to_parquet('train.parquet')
```

### 对话格式

```python
data = {
    'messages': [
        [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'What is 2+2?'},
            {'role': 'assistant', 'content': '4'}
        ],
    ]
}
```

### 多模态格式

```python
data = {
    'messages': [
        [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'Describe this.'}
            ]},
            {'role': 'assistant', 'content': 'This is...'}
        ]
    ],
    'images': [
        ['path/to/image.jpg']  # 或 base64 编码
    ]
}
```

## 数据预处理脚本

verl 提供了多个数据预处理脚本：

```bash
ls examples/data_preprocess/
# aime2024_multiturn_w_tool.py
# dapo_multiturn_w_tool.py
# geo3k.py
# gsm8k.py
# math_dataset.py
# ...
```

### GSM8K 预处理

```bash
python examples/data_preprocess/gsm8k.py \
    --local_save_dir ~/data/gsm8k
```

### MATH 预处理

```bash
python examples/data_preprocess/math_dataset.py \
    --local_dir ~/data/math
```

### 多轮数据预处理

```bash
python examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir ~/data/gsm8k_multiturn
```

## 自定义数据

### 创建自定义数据集

```python
# create_custom_dataset.py
import pandas as pd
import json

# 从 JSON 加载
with open('raw_data.json', 'r') as f:
    raw_data = json.load(f)

# 转换格式
data = {
    'data_source': [],
    'answer': [],  # 用于奖励计算
}

for item in raw_data:
    data['data_source'].append(item['question'])
    data['answer'].append(item['answer'])

# 保存
df = pd.DataFrame(data)
df.to_parquet('train.parquet')
```

### 使用自定义字段

```bash
# 指定 prompt 字段
data.prompt_key=data_source

# 指定 image 字段
data.image_key=images
```

## 数据配置

```bash
# 数据路径
data.train_files=./data/train.parquet
data.val_files=./data/val.parquet

# 支持多个文件
data.train_files="['./data/train1.parquet', './data/train2.parquet']"

# 采样限制
data.train_max_samples=-1  # -1 表示全部
data.val_max_samples=1000

# 长度限制
data.max_prompt_length=512
data.max_response_length=1024

# 过滤
data.filter_overlong_prompts=True
data.truncation=error  # error | left | right | middle

# 打乱
data.shuffle=True
data.seed=42
```

## 下一步

- [09-2-custom-dataset.md](09-2-custom-dataset.md) - 自定义数据集
- [09-3-reward-config.md](09-3-reward-config.md) - 奖励配置
