---
title: "ALiBi - 解决传统位置编码外推问题"
date: 2026-01-04
math: true
---

## 一、ALiBi 的由来：为什么需要它？

### 1. 传统位置编码的外推问题

在标准 Transformer 中，位置信息通过 **绝对位置编码（如正弦、可学习）或相对位置编码（如 RoPE）** 注入。但这些方法在 **训练长度 < 推理长度** 时表现不佳：

- **绝对位置编码**：无法处理训练时未见过的位置索引
- **RoPE**：虽可通过插值（如 YaRN）扩展，但外推能力仍有限，且需额外调参

> 💡 问题核心：**位置编码与序列长度强耦合**

### 2. ALiBi 的提出

- **论文**：《[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)》（ICLR 2022）
- **作者**：Ofir Press et al.（来自 AI21 Labs）
- **核心思想**：**完全移除位置编码**，改用 **与距离成线性关系的偏置（bias）** 直接加到 attention score 上

> ✅ 优势：
> - 模型可在短序列上训练，在超长序列上直接推理（无需微调）
> - 架构更简洁（无位置嵌入层）
> - 在长上下文任务上表现优异
> - 被 **BLOOM（176B）** 等大模型采用

---

## 二、基本原理

ALiBi 的关键洞察是：

> **人类语言中，近期 token 通常比远期 token 更相关**。  
> 因此，注意力应天然倾向于 **局部性（locality）**，且衰减速度可 head-specific 控制。

为此，ALiBi 在计算 attention score 时，对每个 head 引入一个 **线性偏置项**：

\[
\text{score}_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} - m_h \cdot |i - j|
\]

其中：
- \(i, j\) 是 token 位置（\(i\) 为 query 位置，\(j\) 为 key 位置）
- \(m_h > 0\) 是第 \(h\) 个 head 的 **衰减斜率（slope）**
- **注意**：即使 \(i < j\)（未来 token），偏置仍为负（但在 decoder-only 中会被 causal mask 屏蔽）

> 🔑 关键点：**不依赖任何位置嵌入**，仅靠距离 \(|i-j|\) 和可学习（或预设）的斜率控制注意力范围。

---

## 三、数学细节

### 1. 注意力计算（以 decoder-only 为例）

标准 causal attention：
\[
A_{ij} = 
\begin{cases}
\text{softmax}\left( \frac{Q_i K_j^\top}{\sqrt{d}} \right), & j \leq i \\
0, & j > i
\end{cases}
\]

ALiBi 修改为：
\[
A_{ij} = 
\begin{cases}
\text{softmax}\left( \frac{Q_i K_j^\top}{\sqrt{d}} - m_h \cdot (i - j) \right), & j \leq i \\
0, & j > i
\end{cases}
\]

> 📌 因为 \(j \leq i\)，所以 \(|i - j| = i - j\)，偏置为 \(-m_h (i - j)\)

### 2. 斜率 \(m_h\) 的设置

论文发现：**不同 head 应关注不同尺度的上下文**（有的看近，有的看远）。

因此，将 heads 分组，按指数衰减分配斜率：

\[
m_h = 2^{-\frac{8h}{H}}, \quad h = 1, 2, ..., H
\]

例如，当 \(H = 8\)：
- head 0: \(m = 2^{-1} = 0.5\)
- head 1: \(m = 2^{-2} = 0.25\)
- ...
- head 7: \(m = 2^{-8} \approx 0.0039\)

> ✅ 这样，部分 head 关注局部（大斜率），部分 head 关注全局（小斜率）

---

## 四、PyTorch 实现（可运行演示）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_alibi_bias(num_heads: int, seq_len: int, dtype=torch.float32):
    """
    构建 ALiBi 偏置矩阵。
    
    Args:
        num_heads: 注意力头数
        seq_len: 序列长度
        dtype: 数据类型
    
    Returns:
        bias: (num_heads, seq_len, seq_len)
    """
    # 预定义斜率 m_h = 2^(-8h/H)
    slopes = torch.pow(2.0, -torch.arange(1, num_heads + 1, dtype=torch.float32) * 8.0 / num_heads)
    slopes = slopes.view(num_heads, 1, 1)  # (H, 1, 1)

    # 构建距离矩阵: d[i, j] = i - j (仅对 j <= i 有效)
    position_ids = torch.arange(seq_len, dtype=torch.float32)
    relative_position = position_ids[None, :] - position_ids[:, None]  # (L, L)
    relative_position = relative_position.abs().unsqueeze(0)  # (1, L, L)

    # ALiBi 偏置 = -m_h * |i - j|
    alibi_bias = -slopes * relative_position  # (H, L, L)

    # 对于 decoder-only，应用 causal mask（上三角设为 -inf）
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    alibi_bias.masked_fill_(causal_mask, float('-inf'))

    return alibi_bias.to(dtype)

class ALiBiAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # 投影 Q, K, V
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)  # (B, H, L, L)

        # 添加 ALiBi 偏置
        alibi_bias = build_alibi_bias(H, L, dtype=scores.dtype).to(scores.device)
        scores = scores + alibi_bias.unsqueeze(0)  # 广播 batch 维度

        # Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # 加权求和
        output = torch.matmul(attn_weights, v)  # (B, H, L, Dh)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)
        return output


# ------------------ 演示 ------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D = 2, 8, 128
    H = 8

    x = torch.randn(B, L, D)
    model = ALiBiAttention(embed_dim=D, num_heads=H)

    out = model(x)
    print("Input shape:", x.shape)       # [2, 8, 128]
    print("Output shape:", out.shape)     # [2, 8, 128]

    # 可视化 ALiBi 偏置（第一个 head）
    bias = build_alibi_bias(H, L)
    print("\nALiBi bias for head 0 (first 4x4):")
    print(bias[0, :4, :4])
```

### 输出示例（偏置部分）：
```
ALiBi bias for head 0 (first 4x4):
tensor([[ 0.0000,    -inf,    -inf,    -inf],
        [-0.5000,  0.0000,    -inf,    -inf],
        [-1.0000, -0.5000,  0.0000,    -inf],
        [-1.5000, -1.0000, -0.5000,  0.0000]])
```

> 🔍 可见：
> - 对角线为 0（自己对齐自己）
> - 左下方为负值，且随距离线性减小
> - 上三角为 `-inf`（causal mask）

---

## 五、ALiBi vs RoPE：关键区别

| 特性 | RoPE | ALiBi |
|------|------|-------|
| 是否需要位置编码 | ✅ 是（旋转矩阵） | ❌ 否 |
| 外推能力 | 依赖插值（如 YaRN） | **天然支持任意长度** |
| 计算开销 | 需要复数乘法 | 仅加偏置（极低） |
| 适用场景 | 主流 LLM（LLaMA, Qwen） | BLOOM、长文本专用模型 |
| 对称性 | 支持双向（encoder） | 通常用于单向（decoder） |

---

## 六、总结

- **ALiBi = 无位置编码 + 距离线性偏置**
- **公式**：\(\text{score}_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} - m_h \cdot |i - j|\)
- **优势**：训练短、测试长；架构简洁；推理高效
- **应用**：BLOOM（176B 参数）、AI21 的 Jurassic 模型

> 💡 如果你正在设计一个需要 **超长上下文（>32K tokens）** 的模型，ALiBi 是一个值得考虑的轻量级替代方案。

---
