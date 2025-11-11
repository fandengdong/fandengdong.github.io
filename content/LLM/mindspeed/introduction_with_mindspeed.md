---
title: "Mindspeed基本介绍"
description: "MindSpeed is a Megatron adaptation framework tailored for the Ascend ecosystem."
date: 2025-11-10
---

The core purpose of MindSpeed is to efficiently adapt and accelerate large models based on Megatron-LM for training and inference on Huawei Ascend AI hardware. Its key functionalities include:

- Framework Compatibility: Adapting native Megatron-LM code to the Ascend NPU architecture to ensure correct functionality.
- Performance Optimization: Enhancing training and inference efficiency through techniques such as operator fusion, memory optimization, and communication acceleration.
- Multi-Level Acceleration: Offering configurable optimization tiers (e.g., basic compatibility, affinity enhancement, full acceleration) that users can enable based on their needs.
- Ecosystem Integration: Deep integration with Ascend software stacks like CANN and MindSpore for seamless hardware-software co-optimization.

In short, MindSpeed enables Megatron-based large models to run correctly, run fast, and run reliably on Ascend devices.

## One line code to adapt Megatron codes

```python
import torch
import mindspeed.megatron_adaptor # 新增代码行
```

## Acceleration Feature Tier Description

MindSpeed Core acceleration features are divided into three levels. Users can select the optimization level to enable based on actual requirements by setting the `--optimization-level {level}` parameter in the launch script. This parameter supports the following configurations:

| Level | Level Name | Description |
|-------|------------|-------------|
| 0     | Basic Function Compatibility | Provides basic functional compatibility of the Megatron-LM framework with NPUs. |
| 1     | Affinity Enhancement 🔥 | On top of L0, enables partial fusion operators and Ascend-friendly computation rewriting. |
| 2     | Acceleration Feature Enhancement 🔥🔥 | Default value. Enables richer acceleration features based on L0 and L1, with acceleration features typically enabled via specific parameters. Refer to the "Feature Introduction" section for details. |

## MindSpeed Core ecosystem

Building upon the MindSpeed Core acceleration library, additional specialized libraries are available:

- Large Language Model Library: [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)
- Multimodal Model Library: [MindSpeed MM](https://gitcode.com/Ascend/MindSpeed-MM)
- Reinforcement Learning Acceleration Library: [MindSpeed RL](https://gitcode.com/Ascend/MindSpeed-RL)
