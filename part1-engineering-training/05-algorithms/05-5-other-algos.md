# 05-5 - 其他算法

verl 还支持多种其他 RL 算法变体，本章简要介绍。

## 算法列表

| 算法 | 目录 | 特点 |
|------|------|------|
| CISPO | `examples/cispo_trainer/` | Clipped IS-weight PPO |
| DPPO | `examples/dppo_trainer/` | Diffusion PPO |
| FAPO | `examples/fapo_trainer/` | Flow-based PPO |
| GMPO | `examples/gmpo_trainer/` | Grouped Multi-head PPO |
| GPG | `examples/gpg_trainer/` | Generalized Policy Gradient |
| GSPO | `examples/gspo_trainer/` | Grouped Shaped PPO |
| SAPO | `examples/sapo_trainer/` | Self-Adaptive PPO |
| OTB | `examples/otb_trainer/` | Off-policy Training with Buffer |
| MTP | `examples/mtp_trainer/` | Multi-Turn PPO |

## CISPO (Clipped IS-weight PPO)

CISPO 使用 Importance Sampling 权重裁剪来稳定训练。

```bash
# 查看 CISPO 示例
cat examples/cispo_trainer/run_qwen2-7b.sh
```

**特点**：
- 裁剪 IS 权重而非 ratio
- 更稳定的梯度估计
- 适合分布偏移较大的场景

## DPPO (Diffusion PPO)

DPPO 用于扩散模型的 RL 训练。

```bash
# 查看 DPPO 示例
cat examples/dppo_trainer/run_example.sh
```

**特点**：
- 适配扩散模型
- 支持 Classifier-free Guidance
- 图像生成任务

## FAPO (Flow-based PPO)

FAPO 用于 Flow-based 模型的 RL 训练。

```bash
# 查看 FAPO 示例
ls examples/fapo_trainer/
```

**特点**：
- 支持 Normalizing Flow
- 生成任务优化

## GMPO (Grouped Multi-head PPO)

GMPO 使用多头策略网络。

```bash
# 查看 GMPO 示例
ls examples/gmpo_trainer/
```

**特点**：
- 多头策略
- 多样性探索

## GPG (Generalized Policy Gradient)

GPG 是通用的 Policy Gradient 实现。

```bash
# 查看 GPG 示例
ls examples/gpg_trainer/
```

**特点**：
- 通用框架
- 易于扩展

## GSPO (Grouped Shaped PPO)

GSPO 使用 Group-based 奖励塑形。

```bash
# 查看 GSPO 示例
ls examples/gspo_trainer/
```

**特点**：
- 奖励塑形
- Group-based 优化

## SAPO (Self-Adaptive PPO)

SAPO 自适应调整超参数。

```bash
# 查看 SAPO 示例
ls examples/sapo_trainer/
```

**特点**：
- 自适应 KL 惩罚
- 自适应学习率

## OTB (Off-policy Training with Buffer)

OTB 支持离策略训练。

```bash
# 查看 OTB 示例
ls examples/otb_trainer/
```

**特点**：
- 经验回放
- 数据重用
- 样本效率高

## MTP (Multi-Turn PPO)

MTP 专门用于多轮对话训练。

```bash
# 查看 MTP 示例
ls examples/mtp_trainer/
```

**特点**：
- 多轮对话支持
- 状态追踪
- 对话历史管理

## 使用通用训练器

大多数算法可以通过 `main_ppo.py` 配置：

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=<algorithm_name> \
    ...
```

## 查看具体示例

```bash
# 列出所有示例
ls examples/

# 查看特定算法的 README
cat examples/grpo_trainer/README.md
cat examples/ppo_trainer/README.md

# 查看运行脚本
cat examples/grpo_trainer/run_qwen2-7b.sh
```

## 算法选择指南

```
                ┌─────────────────────┐
                │ 有 Reward Model?    │
                └──────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
             Yes                        No
              │                         │
              ▼                         ▼
        ┌─────────┐              ┌─────────────┐
        │  PPO    │              │ 可验证奖励? │
        └─────────┘              └──────┬──────┘
                                        │
                           ┌────────────┴────────────┐
                           │                         │
                          Yes                        No
                           │                         │
                           ▼                         ▼
                     ┌───────────┐            ┌───────────┐
                     │ GRPO/     │            │ REINFORCE │
                     │ RLOO/     │            │ 或使用 RM │
                     │ ReMax     │            └───────────┘
                     └───────────┘
```

## 下一步

- [06-sft-and-tuning](../06-sft-and-tuning/) - SFT 与微调
- [07-agent-rl](../07-agent-rl/) - AgentRL 训练
