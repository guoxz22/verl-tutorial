# C - 进阶学习资源

本章提供进一步学习 verl 的资源。

## 官方资源

### 文档

- [verl 官方文档](https://verl.readthedocs.io/)
- [HybridFlow 论文](https://arxiv.org/abs/2409.19256)
- [GitHub 仓库](https://github.com/volcengine/verl)

### 示例代码

- [examples/](https://github.com/volcengine/verl/tree/main/examples) - 官方示例
- [recipe/](https://github.com/verl-project/verl-recipe) - 算法配方

## 社区项目

### 基于 verl 的项目

| 项目 | 描述 | 链接 |
|------|------|------|
| TinyZero | R1 Zero 复现 | github.com/Jiayi-Pan/TinyZero |
| SkyThought | Sky-T1-7B RL | github.com/NovaSky-AI/SkyThought |
| Easy-R1 | 多模态 RL | github.com/hiyouga/EasyR1 |
| OpenManus-RL | Agent RL | github.com/OpenManus/OpenManus-RL |
| RAGEN | Agent 训练 | github.com/ZihanWang314/ragen |
| Search-R1 | 搜索+推理 | github.com/PeterGriffinJin/Search-R1 |
| verl-agent | 长程 Agent | github.com/langfengQ/verl-agent |
| ToRL | Tool RL | github.com/GAIR-NLP/ToRL |

## 算法学习

### PPO 相关

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - PPO 原论文
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) - PPO for LLM

### GRPO 相关

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) - GRPO 原论文
- [DAPO](https://dapo-sia.github.io/) - DAPO 算法

### 其他算法

- [REINFORCE++](https://arxiv.org/abs/2401.08667)
- [RLOO](https://arxiv.org/abs/2402.14740)
- [ReMax](https://arxiv.org/abs/2310.02256)

## 系统学习

### 分布式训练

- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

### 推理引擎

- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://github.com/sgl-project/sglang)

## 博客和教程

- [verl Blog](https://verl-project.github.io/)
- [Awesome ML-SYS Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial) - verl 相关教程
- [verl 多轮代码解析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)

## 性能调优

- [verl Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
- [DeepSeek 671B Performance](https://verl.readthedocs.io/en/latest/perf/dpsk.html)

## 贡献

### 如何贡献

1. Fork [verl 仓库](https://github.com/volcengine/verl)
2. 阅读 [贡献指南](https://github.com/volcengine/verl/blob/main/CONTRIBUTING.md)
3. 提交 Pull Request

### 报告问题

- [GitHub Issues](https://github.com/volcengine/verl/issues)

## 持续更新

verl 正在快速发展，建议：

1. 关注 [GitHub Releases](https://github.com/volcengine/verl/releases)
2. 加入 [Slack 社区](https://join.slack.com/t/verl-project/)
3. 关注 [Twitter](https://twitter.com/verl_project)

---

## 教程总结

本教程涵盖了 verl 框架的核心使用方法：

**Part 0 基础**：安装、配置、核心概念
**Part 1 工程训练**：内置算法使用、分布式训练、数据处理
**Part 2 研究扩展**：自定义算法、Worker 修改、论文复现

希望这份教程能帮助你：

- 工程人员：快速完成 RL 训练任务
- 研究员：高效实现和验证新算法

如有问题，请查阅官方文档或在 GitHub 提 Issue。

Happy Training!
