---
title: "GDN - Qwen3-Next中引入的一种状态空间模型变体"
date: 2025-01-09
math: true
---

Ref: [知乎](https://zhuanlan.zhihu.com/p/1970797248171475540)

Gated DeltaNet 是 Qwen3-Next（通义千问 3 的下一代模型）中引入的一种**状态空间模型（State Space Model, SSM）变体**，用于增强 Transformer 架构在长序列建模中的能力。它融合了 **Delta Rule（增量学习规则）**、**门控机制（Gating）** 和 **状态空间建模思想**，旨在解决传统注意力机制在处理超长上下文时的计算和内存瓶颈。

下面我们将从**由来、演变、集成方式、数学形式**到 **PyTorch 实现** 一步步解析 Gated DeltaNet。

---

## 一、由来与动机

### 1.1 背景：Transformer 的局限
- 标准 Transformer 使用自注意力机制，其复杂度为 $O(L^2)$，其中 $L$ 是序列长度。
- 对于超长上下文（如 100K+ tokens），注意力机制变得不可扩展。

### 1.2 状态空间模型（SSM）的兴起
- SSM（如 S4、S5、Mamba）提供了一种线性复杂度 $O(L)$ 的序列建模范式。
- 其核心是维护一个**隐状态** $h_t$，通过递推更新：
  $$
  h_t = \bar{A} h_{t-1} + \bar{B} x_t
  $$
  $$
  y_t = C h_t
  $$
  其中 $\bar{A}, \bar{B}$ 是离散化后的系统矩阵。

### 1.3 Delta Rule 与 Hebbian 学习
- Delta Rule 是一种增量学习规则，形式为：
  $$
  \Delta W \propto (y_{\text{target}} - y) x^\top
  $$
- 在神经记忆模型中，Delta Rule 可被解释为**快速联想记忆更新**：新输入 $x_t$ 与当前输出误差共同驱动权重更新。

### 1.4 Gated DeltaNet 的提出
- Gated DeltaNet 将 Delta Rule 的思想嵌入到 SSM 框架中，形成一种**可学习的、门控控制的记忆更新机制**。
- 它不是直接更新权重，而是维护一个**键值记忆矩阵** $M_t$，并通过门控机制进行增量更新。

---

## 二、Gated DeltaNet 的数学形式

Gated DeltaNet 的核心是维护一个**外部记忆矩阵** $M_t \in \mathbb{R}^{d_k \times d_v}$，并在每个时间步通过输入 $x_t$ 更新它。

设输入序列为 $\{x_t\}_{t=1}^L$，每个 token 映射为：
- 查询向量：$q_t = W_q x_t$
- 键向量：$k_t = W_k x_t$
- 值向量：$v_t = W_v x_t$

### 2.1 记忆更新规则（Delta Rule 风格）
$$
M_t = \lambda_t M_{t-1} + \eta_t k_t v_t^\top
$$
其中：
- $\lambda_t \in [0,1]$ 是**遗忘门**（forget gate），控制旧记忆的保留程度；
- $\eta_t \in [0,1]$ 是**写入门**（write gate），控制新信息的写入强度；
- $k_t v_t^\top$ 是外积形式的“联想记忆”项，类似 Hebbian 学习。

### 2.2 输出计算
$$
o_t = q_t^\top M_t
$$
即用当前查询 $q_t$ 从记忆矩阵 $M_t$ 中读取信息。

### 2.3 门控机制（Gating）
$$
\lambda_t = \sigma(g_\lambda^\top x_t), \quad \eta_t = \sigma(g_\eta^\top x_t)
$$
其中 $\sigma$ 是 sigmoid，$g_\lambda, g_\eta$ 是可学习向量（或通过小型 MLP 实现）。

> 注意：实际实现中，为避免显式存储 $M_t$（其大小为 $d_k \times d_v$），通常采用**低秩近似**或**递推展开**技巧。但在 Gated DeltaNet 中，由于 $d_k, d_v$ 通常较小（如 64），直接维护 $M_t$ 是可行的。

---

## 三、在 Transformer 中的集成方式

Gated DeltaNet 通常作为 **替代或补充标准自注意力模块** 的组件，集成在 Transformer 层中：

- **方案 A（替代）**：完全用 Gated DeltaNet 替代 Multi-Head Attention（MHA），形成纯 SSM 架构。
- **方案 B（混合）**：在 MHA 之后添加 Gated DeltaNet 作为“记忆增强”模块（类似 Linear Transformer 或 RWKV 的思路）。
- **Qwen3-Next 采用的是混合架构**：在某些层使用 Gated DeltaNet，尤其在处理长上下文时。

其位置通常如下：
```text
[Input] → LayerNorm → GatedDeltaNet → Residual → LayerNorm → FFN → Residual → [Output]
```

---

## 四、PyTorch 实现

下面是一个简化但功能完整的 Gated DeltaNet 模块实现（支持 batch 和序列维度）(注意：仅供参考，某些细节可能不同）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedDeltaNet(nn.Module):
    def __init__(self, d_model: int, d_k: int = 64, d_v: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_v, bias=False)

        # Gating parameters (simplified: use linear + sigmoid)
        self.gate_proj = nn.Linear(d_model, 2, bias=True)  # outputs [logit_lambda, logit_eta]

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.zeros_(self.gate_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_v)
        """
        B, L, D = x.shape

        q = self.W_q(x)  # (B, L, d_k)
        k = self.W_k(x)  # (B, L, d_k)
        v = self.W_v(x)  # (B, L, d_v)

        # Compute gates: lambda (forget), eta (write)
        gate_logits = self.gate_proj(x)  # (B, L, 2)
        lambda_t = torch.sigmoid(gate_logits[..., 0])  # (B, L)
        eta_t = torch.sigmoid(gate_logits[..., 1])     # (B, L)

        # Initialize memory M: (B, d_k, d_v)
        M = torch.zeros(B, self.d_k, self.d_v, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            # Update memory: M_t = lambda_t * M_{t-1} + eta_t * (k_t ⊗ v_t)
            kt = k[:, t].unsqueeze(2)      # (B, d_k, 1)
            vt = v[:, t].unsqueeze(1)      # (B, 1, d_v)
            outer = kt @ vt                # (B, d_k, d_v)

            lambda_t_i = lambda_t[:, t].view(B, 1, 1)  # (B, 1, 1)
            eta_t_i = eta_t[:, t].view(B, 1, 1)        # (B, 1, 1)

            M = lambda_t_i * M + eta_t_i * outer

            # Read: o_t = q_t^T M
            qt = q[:, t].unsqueeze(1)      # (B, 1, d_k)
            ot = qt @ M                    # (B, 1, d_v)
            outputs.append(ot.squeeze(1))

        output = torch.stack(outputs, dim=1)  # (B, L, d_v)
        return output
```

### 说明：
- 该实现是**顺序递推**的，因此时间复杂度为 $O(L \cdot d_k \cdot d_v)$，空间复杂度为 $O(d_k \cdot d_v)$（不随 L 增长）。
- 若 `d_k = d_v = 64`，则每步更新仅需约 4K 参数操作，远低于注意力的 $O(L^2)$。
- 实际部署中可进一步优化（如使用 `torch.compile` 或 CUDA kernel）。

---

## 五、与 Mamba / SSM 的区别

| 特性 | Mamba | Gated DeltaNet |
|------|-------|----------------|
| 核心机制 | 连续 SSM + 选择性扫描 | 外积记忆 + 门控增量更新 |
| 状态形式 | 向量 $h_t \in \mathbb{R}^N$ | 矩阵 $M_t \in \mathbb{R}^{d_k \times d_v}$ |
| 更新规则 | $h_t = A h_{t-1} + B x_t$ | $M_t = \lambda M_{t-1} + \eta k_t v_t^\top$ |
| 可解释性 | 黑盒动态系统 | 显式键值记忆，类似快速 Hebbian 学习 |

---

## 六、总结

Gated DeltaNet 是 Qwen3-Next 中一种创新的**高效长程建模模块**，其核心思想是：
- 利用**外积形式**构建联想记忆；
- 通过**双门控机制**（遗忘门 + 写入门）实现可控的增量更新；
- 以**线性复杂度**替代注意力，同时保留对关键信息的长期记忆能力。

它代表了大模型架构从“纯注意力”向“混合记忆-注意力”演进的重要一步。

> 注：截至 2026 年初，Gated DeltaNet 的细节尚未完全公开，以上解析基于 Qwen 团队技术报告、ICLR/NeurIPS 相关工作（如 DeltaNet、Hebbian Transformer、Mamba）以及合理推断。实际 Qwen3-Next 实现可能包含更多工程优化（如分组、低秩、并行化等）。
