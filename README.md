# VERL 使用教程

> 基于 verl v0.7.0 | 更新日期：2026-03-08

## 这是什么

这是一份面向实践者的 verl 框架使用教程。verl 是字节跳动开源的大语言模型强化学习训练框架，支持 PPO、GRPO 等多种算法，以及分布式训练、AgentRL 等场景。


## 适用读者

| 读者类型 | 特征 | 学习目标 |
|---------|------|----------|
| **研究员** | 熟悉RL/LLM理论，工程能力有限 | 论文复现、修改框架实现自定义算法 |
| **工程人员** | 需要完成训练任务，不修改框架 | 使用内置功能完成各类RL训练任务 |

## 教程结构

```
verl-tutorial/
│
├── part0-foundation/              # 基础（必读）
│   ├── 00-what-is-verl.md         #   verl是什么
│   ├── 01-installation.md         #   安装配置
│   ├── 02-quick-start.md          #   快速上手
│   ├── 03-core-concepts.md        #   核心概念
│   └── 04-configuration.md        #   配置系统
│
├── part1-engineering-training/    # 工程训练（面向工程人员）
│   ├── 05-algorithms/             #   内置算法使用
│   ├── 06-sft-and-tuning/         #   SFT与微调
│   ├── 07-agent-rl/               #   AgentRL训练
│   ├── 08-distributed-training/   #   分布式训练
│   ├── 09-data-and-reward/        #   数据与奖励
│   └── 10-operations/             #   运维与监控
│
├── part2-research-extension/      # 研究扩展（面向研究员）
│   ├── 11-extension-overview.md   #   扩展机制总览
│   ├── 12-custom-algorithm/       #   自定义算法组件
│   ├── 13-modify-workers/         #   修改Worker
│   └── 14-paper-reproduction/     #   论文复现
│
└── appendix/                      # 附录
    ├── A-config-reference.md      #   配置参数参考
    ├── B-common-errors.md         #   常见错误
    └── C-resources.md             #   进阶资源
```

## 阅读路径

### 工程人员路径

```
Part0 基础 → Part1 工程训练 → 附录
```

**推荐顺序**：
1. 00-04：掌握基础概念和配置
2. 05-algorithms：选择适合的算法
3. 08-distributed-training：配置训练环境
4. 09-data-and-reward：准备数据
5. 10-operations：监控和运维
6. 06/07：根据需要学习SFT或AgentRL

### 研究员路径

```
Part0 基础 → Part2 研究扩展 → 附录
```

**推荐顺序**：
1. 00-04：掌握基础概念和配置
2. 11-extension-overview：了解扩展点
3. 12-custom-algorithm：学习自定义组件
4. 13-modify-workers：深入Worker修改
5. 14-paper-reproduction：复现实战

> 研究员可能需要参考 Part1 中的具体配置示例

## 前置知识

- **必须**：Python 基础、PyTorch 基础
- **推荐**：强化学习基础概念（Policy Gradient、PPO）
- **可选**：分布式训练概念、Hydra 配置框架

## 代码约定

```bash
# 命令行示例
python -m verl.trainer.main_ppo config.key=value

# 配置文件示例
algorithm:
  adv_estimator: grpo
```


## 许可证

本教程采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。

verl 框架采用 Apache 2.0 许可证。
