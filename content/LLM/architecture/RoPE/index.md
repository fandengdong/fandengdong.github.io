---
title: "RoPE - 旋转位置编码解读"
date: 2025-12-19
math: true
---

旋转位置编码（RoPE, Rotary Position Embedding）是一种在Transformer模型中引入相对位置信息的方法，由苏剑林等人在2021年提出。相比传统的绝对位置编码（如BERT中的可学习位置嵌入或正弦位置编码），RoPE通过将位置信息以旋转变换的方式融入注意力机制，使得模型天然具备对相对位置的感知能力。

## 回顾原始位置编码

在Transformer模型中，位置编码（Position Encoding）用于将输入序列中的每个位置映射为向量，以表示其相对位置信息。原始位置编码的实现方式有很多种，如可学习位置嵌入（Learnable Position Embedding）或正弦位置编码（Sinusoidal Position Encoding）。

假设我们的词汇表大小为`[vocab_size, embedding_dim]`，输入句子的token长度为`seq_len`，每个token会被转换成一个vector向量，维度为`[embedding_dim]`，则整个输入句子的shape为`[ seq_len, embedding_dim]`。

但是，为了让模型理解序列的顺序，还需要加入位置编码(Positional Encoding)。这就是著名的公式：

$$
PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- `pos`是位置索引，即token在序列中的位置，从0开始；
- `i`是维度索引，从0开始到$(d_{model}-1)/2$结束，表示词嵌入的维度；
- $d_{model}$是词嵌入（word embedding）的维度`embedding_dim`

观察输入句子的shape: `[seq_len, embedding_dim]`，位置编码的时候，`pos`对应的是token在`seq_len`维度的位置，而维度索引`i`对应词嵌入`embedding_dim`的维度。

pytorch代码实现:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## RoPE 的核心思想

### 相对位置 vs 绝对位置

- 在传统Transformer中，位置编码是加到词嵌入上的（如 x + pos_emb），这种方式只保留了绝对位置信息。
- 而 RoPE 则通过将查询（Q）和键（K）向量进行旋转变换，使得注意力分数自然包含两个 token 之间的相对距离。

### 旋转操作

对于维度为 d 的向量，RoPE 将其分成若干二维子空间（通常是相邻两个维度一组），每组应用一个二维旋转矩阵：

$$
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)  \\\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$

其中：
- $m$ 是token的位置，取整数，从0开始；
- $\theta_i = 10000 ^ {-2 i / d}$, 与原始 Transformer 的正弦位置编码频率一致;
- i 是维度索引，从0开始到d-1结束。

这样，位置m的token向量经过旋转变换后，与其他位置n的token计算点积时，会自动包含 (m−n) 的信息。

## 数学直观理解

设$q_m$和$k_n$, 分别是位置 m 和 n 的 Q/K 向量，经过 RoPE 编码后：

$$
\text{Attention}(m,n)=f_{RoPE}(q,m)^⊤ f_{RoPE}(k,n) = g(q,k,m−n)
$$

## Pytorch代码实现

下面是一个简洁、清晰的 RoPE 实现，适用于任意偶数维度（奇数维度可补零或忽略最后一维）：

```python
import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        """
        Args:
            dim: embedding dimension (must be even)
            max_seq_len: maximum sequence length
            base: base for frequency computation (default 10000, as in original Transformer)
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even embedding dimension"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the rotation matrix for all positions [0, max_seq_len)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # shape: (dim//2,)
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)  # shape: (max_seq_len,)
        freqs = torch.outer(t, inv_freq)  # shape: (max_seq_len, dim//2)

        # Build complex numbers: cos + i*sin
        emb = torch.cat((freqs, freqs), dim=-1)  # shape: (max_seq_len, dim)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_dim: int = -2):
        """
        Apply rotary position embedding to input tensor.

        Args:
            x: input tensor of shape [bs, seq_len, dim]
            seq_dim: the dimension corresponding to sequence length (default: -2)

        Returns:
            rotated tensor of same shape as x
        """
        seq_len = x.size(seq_dim)
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"

        # Extract cos and sin for the actual sequence length
        cos = self.cos[:seq_len, :].unsqueeze(0)  # (1, seq_len, dim)
        sin = self.sin[:seq_len, :].unsqueeze(0)

        # Split x into even and odd parts (for rotation)
        x1, x2 = x.chunk(2, dim=-1)  # each: [bs seq_len, dim//2]

        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos[..., :self.dim//2] - x2 * sin[..., :self.dim//2] # [bs, seq_len, dim//2]
        rotated_x2 = x1 * sin[..., :self.dim//2] + x2 * cos[..., :self.dim//2]

        return torch.cat((rotated_x1, rotated_x2), dim=-1)
```

## Attention中使用

通常，RoPE 只应用于 Q 和 K，不用于 V：

```python
# 假设 q, k, v 已经通过线性层得到，shape: (B, L, D)
q = rope(q)
k = rope(k)

attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
attn_weights = torch.softmax(attn_scores, dim=-1)
output = torch.matmul(attn_weights, v)
```
这样，注意力机制就自动包含了相对位置信息。

## 实际应用

- LLaMA / LLaMA2 / Mistral 等大模型均采用 RoPE
- 支持 动态长度（只要不超过预设 max_seq_len）
- 可扩展为 线性缩放（NTK-aware） 或 YaRN 等变体以支持超长上下文

## RoPE原理演示

![RoPE原理演示](RoPE.jpg)

在Transformer模型中，传入的张量通常具有形状[batch_size, seq_len, hidden_dim]。为了应用RoPE，我们主要关注seq_len（序列长度）和hidden_dim（隐藏层维度）两个维度。假设这个张量为X，其形状为[token_index, hidden_index]，如上图中的二维网格所示。

在旋转过程中，我们将hidden_dim维度按每两个相邻维度为一组进行分组，并使用旋转矩阵对每组进行旋转变换，如图中所示。关键在于理解旋转矩阵的索引是如何映射到二维网格中的，这里的索引涉及：

- token索引（对应公式中的m）：表示序列中每个token的位置
- hidden维度索引（对应公式中的i）：表示嵌入向量中的维度位置
- 
通过这种分组旋转操作，RoPE能够将位置信息有效地编码到向量的内部表示中，使得模型能够感知token之间的相对位置关系。

## Reference

1. [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
