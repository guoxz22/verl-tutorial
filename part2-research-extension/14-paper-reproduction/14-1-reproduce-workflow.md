# 14-1 - 论文复现工作流

本章介绍使用 verl 复现论文的系统化方法。

## 复现工作流

```
┌─────────────────────────────────────────────────────────────┐
│                    论文复现工作流                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 理解论文                                                │
│     └─► 算法核心、超参数、实验设置                          │
│                                                             │
│  2. 确定复现范围                                            │
│     └─► 哪些是 verl 已支持的？哪些需要扩展？               │
│                                                             │
│  3. 准备数据                                                │
│     └─► 预处理脚本、数据格式转换                            │
│                                                             │
│  4. 配置基线                                                │
│     └─► 使用内置算法建立 baseline                          │
│                                                             │
│  5. 实现修改                                                │
│     └─► Advantage / Loss / Reward Manager                   │
│                                                             │
│  6. 调试验证                                                │
│     └─► 单元测试、小规模实验                                │
│                                                             │
│  7. 大规模实验                                              │
│     └─► 完整训练、结果对比                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 步骤 1：理解论文

创建复现清单：

```markdown
# 论文复现清单

## 基本信息
- 论文标题：
- 发表时间：
- 代码链接（如有）：

## 算法核心
- 算法类型：PPO / GRPO / 其他
- 核心创新点：
  1.
  2.

## 超参数
- 学习率：
- Batch size：
- 采样数 n：
- KL 系数：
- 其他：

## 实验设置
- 模型：
- 数据集：
- 评估指标：

## verl 对应
- 已支持：[列出 verl 已支持的部分]
- 需实现：[列出需要自定义的部分]
```

## 步骤 2：确定复现范围

| 论文组件 | verl 支持 | 需要实现 |
|----------|-----------|----------|
| PPO 算法 | ✅ | - |
| GRPO 算法 | ✅ | - |
| 自定义 Advantage | - | ✅ |
| 自定义 Loss | - | ✅ |
| 自定义 Reward | - | ✅ |

## 步骤 3：准备数据

```bash
# 使用 verl 预处理脚本
python examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k
python examples/data_preprocess/math_dataset.py --local_dir ./data/math
```

## 步骤 4：配置基线

```bash
# 先用内置 GRPO 建立 baseline
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/gsm8k/train.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=5
```

## 步骤 5：实现修改

参见 [12-custom-algorithm](../12-custom-algorithm/) 章节的详细说明。

## 步骤 6：调试验证

### 单元测试

```python
def test_custom_algorithm():
    """测试自定义算法组件"""
    from custom_advantage import my_advantage

    # 创建测试数据
    data = create_test_data()

    # 测试
    advantages = my_advantage(data, config)

    # 验证
    assert advantages.shape == expected_shape
    assert not torch.isnan(advantages).any()
```

### 小规模实验

```bash
# 用小数据集快速验证
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=my_custom_adv \
    data.train_max_samples=100 \
    data.val_max_samples=10 \
    trainer.total_epochs=1 \
    ...
```

## 步骤 7：大规模实验

```bash
# 完整训练
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=my_custom_adv \
    data.train_files=./data/full_train.parquet \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.total_epochs=15 \
    trainer.logger='["console","wandb"]'
```

## 下一步

- [14-2-case-studies.md](14-2-case-studies.md) - 案例研究
