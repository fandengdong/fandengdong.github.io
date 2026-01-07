---
title: "SwiGLU激活函数解析"
date: 2026-01-07
math: true
---


SwiGLU（Swish-Gated Linear Unit）是一种在现代大语言模型（如PaLM、LLaMA等）中广泛使用的激活函数变体，用于替代传统的前馈网络（Feed-Forward Network, FFN）中的 ReLU 或 GELU。它结合了门控机制（gating）与 Swish 激活函数，具有更强的表达能力和训练稳定性。

---

## 一、SwiGLU 的由来

SwiGLU 最早由 **Shazeer (2020)** 在论文《**GLU Variants Improve Transformer**》中系统性地提出并评估。该工作探索了多种基于 **Gated Linear Unit (GLU)** 的激活函数变体，并发现 **SwiGLU** 在语言建模任务中表现最优。

### 背景：GLU（Gated Linear Unit）
GLU 最初由 Dauphin 等人（2017）在《Language Modeling with Gated Convolutional Networks》中提出，其基本形式为：

\[
\text{GLU}(x) = (W_1 x + b_1) \otimes \sigma(W_2 x + b_2)
\]

其中：
- \( \otimes \) 表示逐元素乘法（Hadamard product），
- \( \sigma \) 是 Sigmoid 激活函数。

GLU 引入了“门控”思想：一部分线性变换被另一部分通过激活函数“门控”，从而实现更灵活的信息控制。

### SwiGLU 的改进
SwiGLU 将 GLU 中的 Sigmoid 替换为 **Swish**（也称 SiLU）激活函数：

\[
\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
\]

因此，SwiGLU 定义为：

\[
\text{SwiGLU}(x) = \text{Swish}(W_1 x + b_1) \otimes (W_2 x + b_2)
\]

注意：有些实现中会将偏置项省略（尤其在大规模 Transformer 中），且权重维度通常为输入维度的倍数（如 2/3 倍或 4 倍扩展）。

---

## 二、基本原理

SwiGLU 的核心思想是：
- 将输入 \( x \in \mathbb{R}^d \) 投影到一个更高维空间（如 \( \mathbb{R}^{2r} \)），
- 将该高维向量拆分为两部分：\( a \) 和 \( b \)（各 \( r \) 维），
- 对 \( a \) 应用 Swish 激活，再与 \( b \) 逐元素相乘，
- 最后投影回原始维度。

这种结构比标准 FFN（两个线性层 + ReLU/GELU）更具表达能力，因为门控机制允许模型动态地“关闭”某些通道。

---

## 三、详细数学细节

设输入 \( x \in \mathbb{R}^d \)，SwiGLU 层包含两个可学习权重矩阵 \( W_1, W_2 \in \mathbb{R}^{r \times d} \) 和可选偏置 \( b_1, b_2 \in \mathbb{R}^r \)。

1. **线性投影**：
   \[
   a = W_1 x + b_1 \quad (\in \mathbb{R}^r) \\
   b = W_2 x + b_2 \quad (\in \mathbb{R}^r)
   \]

2. **Swish 激活**：
   \[
   \text{Swish}(a) = a \cdot \sigma(a) = \frac{a}{1 + e^{-a}}
   \]

3. **门控乘积**：
   \[
   h = \text{Swish}(a) \odot b \quad (\in \mathbb{R}^r)
   \]

4. **输出投影（可选，取决于上下文）**：
   若 SwiGLU 作为 FFN 的中间层，则通常还有一个输出投影 \( W_3 \in \mathbb{R}^{d \times r} \)：
   \[
   y = W_3 h
   \]

> 注意：在 LLaMA 等模型中，FFN 结构为：
> - 先用两个并行线性层将 \( x \) 映射到 \( (r, r) \)，
> - 应用 SwiGLU 得到 \( h \in \mathbb{R}^r \)，
> - 再用一个线性层映射回 \( d \)。

通常 \( r = \frac{4d}{3} \) 并向上取整为 256 的倍数（LLaMA 的做法）。

---

## 四、PyTorch 实现

下面是一个完整的、可运行的 SwiGLU 模块实现，包括 FFN 形式（带输入/输出投影）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        """
        SwiGLU 前馈网络模块。
        Args:
            input_dim: 输入维度 d
            hidden_dim: 中间维度 r（通常 > input_dim）
            bias: 是否使用偏置
        """
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, input_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., input_dim]
        a = self.w1(x)          # [..., hidden_dim]
        b = self.w2(x)          # [..., hidden_dim]
        swish_a = a * torch.sigmoid(a)  # Swish(a)
        gated = swish_a * b     # SwiGLU gate
        output = self.w3(gated) # [..., input_dim]
        return output

# 示例使用
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    hidden_dim = int(4 * d_model / 3)  # LLaMA 风格
    hidden_dim = ((hidden_dim + 255) // 256) * 256  # 向上对齐到 256 倍数

    x = torch.randn(batch_size, seq_len, d_model)
    swiglu = SwiGLU(input_dim=d_model, hidden_dim=hidden_dim)

    y = swiglu(x)
    print("Input shape:", x.shape)   # [2, 10, 512]
    print("Output shape:", y.shape)  # [2, 10, 512]
```

### 关键点说明：
- `w1` 和 `w2` 并行处理输入，分别生成门控信号和内容信号。
- 使用 `torch.sigmoid(a)` 实现 Swish（即 `a * sigmoid(a)`）。
- 最终通过 `w3` 投影回原维度。
- 这种结构替换了传统 FFN 中的 `Linear → GELU → Linear`。

---

## 五、为什么 SwiGLU 更好？

1. **门控机制**：允许模型选择性地传递信息，比 ReLU/GELU 更灵活。
2. **平滑性**：Swish 是光滑、非单调的激活函数，梯度更稳定。
3. **实证效果**：在多个语言建模基准上，SwiGLU 一致优于 ReLU、GELU、甚至其他 GLU 变体（如 ReGLU、GeGLU）。

---

## 参考文献

- Shazeer, N. (2020). **GLU Variants Improve Transformer**. arXiv:2002.05202.  
  https://arxiv.org/abs/2002.05202
- Dauphin, Y. N., et al. (2017). **Language Modeling with Gated Convolutional Networks**. ICML.
- Touvron, H., et al. (2023). **LLaMA: Open and Efficient Foundation Language Models**.

---
