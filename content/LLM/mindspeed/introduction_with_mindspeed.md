---
title: "Mindspeed基本介绍"
description: "MindSpeed是一个为昇腾生态系统定制的Megatron适配框架。"
date: 2025-11-10
---

MindSpeed的核心目的是高效地适配和加速基于Megatron-LM的大模型，使其能够在华为昇腾AI硬件上进行训练和推理。其主要功能包括：

- 框架兼容性：将原生Megatron-LM代码适配到昇腾NPU架构，确保功能正确性。
- 性能优化：通过算子融合、内存优化和通信加速等技术提升训练和推理效率。
- 多级加速：提供可配置的优化层级（如基础兼容性、亲和性增强、全面加速），用户可根据需求启用。
- 生态集成：与CANN和MindSpore等昇腾软件栈深度集成，实现无缝的软硬件协同优化。

简而言之，MindSpeed使基于Megatron的大模型能够在昇腾设备上正确运行、快速运行和可靠运行。

## 一行代码适配Megatron代码

```python
import torch
import mindspeed.megatron_adaptor # 新增代码行
```

## 加速特性层级说明

MindSpeed Core加速特性分为三个层级。用户可以通过在启动脚本中设置`--optimization-level {level}`参数来根据实际需求选择要启用的优化级别。该参数支持以下配置：

| 等级 | 等级名称 | 描述 |
|-------|------------|-------------|
| 0     | 基础功能兼容性 | 提供Megatron-LM框架与NPU的基础功能兼容性。 |
| 1     | 亲和性增强 🔥 | 在L0基础上，启用部分融合算子和昇腾友好的计算重写。 |
| 2     | 加速特性增强 🔥🔥 | 默认值。在L0和L1基础上启用更丰富的加速特性，加速特性通常通过特定参数启用。详情请参阅"特性介绍"部分。 |

## MindSpeed Core生态系统

在MindSpeed Core加速库的基础上，还提供了以下专业库：

- 大语言模型库: [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)
- 多模态模型库: [MindSpeed MM](https://gitcode.com/Ascend/MindSpeed-MM)
- 强化学习加速库: [MindSpeed RL](https://gitcode.com/Ascend/MindSpeed-RL)
