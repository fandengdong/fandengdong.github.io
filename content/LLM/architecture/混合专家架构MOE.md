---
title: "MOE - 混合专家架构"
date: 2025-12-24
math: true
---

混合专家（Mixture of Experts, MoE）是一种**条件计算（conditional computation）**架构，旨在在不显著增加计算成本的前提下扩展模型容量。它已被广泛应用于现代大语言模型（如 Google 的 GLaM、Mixtral、Qwen-MoE、DeepSeek-MoE 等）。

## MoE 的核心思想

- 🎯 目标：
    - 扩大模型参数量（提升表达能力）
    - 保持每次前向计算的 FLOPs 基本不变（高效推理）
- 🔑 关键机制：
    - 模型由多个“专家”（Experts）组成，每个专家是一个子网络（如 FFN）。
    - 引入一个门控网络（Gating Network），根据输入动态决定激活哪几个专家。
    - 通常只激活 Top-K 个专家（如 K=1 或 K=2），其余专家不参与计算。
> ✅ 这样：总参数量 = 所有专家参数之和（很大），但每 token 计算量 ≈ K 个专家（很小）。
