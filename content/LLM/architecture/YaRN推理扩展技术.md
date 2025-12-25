---
title: "YaRN -  旋转位置编码拓展"
date: 2025-12-25
math: true
---

旋转位置编码（RoPE, Rotary Position Embedding）是一种在Transformer模型中引入相对位置信息 的方法，由苏剑林等Credo在2021年提出。相比传统的绝对位置编码（如BERT中的可学习位置嵌入或正弦位置编码），RoPE通过将位置信息以旋变换的方式融入注意力机制，使得模型可以自然地感知相对位置信息。

但是在实践中，我们往往发现有些应用需要处理更长的序列，而模型在训练时候的输入序列长度往往有限，这时RoPE就无法满足需求。我们需要一个能够将位置编码进行外推的技术。YARN（Yet Another RoPE-based Attention Network）是一种用于扩展大语言模型上下文长度的技术，它是在RoPE（Rotary Position Embedding）基础上进行改进的，旨在让模型在训练时使用较短上下文，但在推理时能有效处理远超训练长度的输入序列。

## 背景：为什么需要 YARN？

### 1. 位置编码的外推问题

大多数大语言模型（如 LLaMA、ChatGLM 等）使用 RoPE（旋转位置编码） 来注入位置信息。RoPE 的优点是具有良好的相对位置建模能力，但其缺点是：

- **训练时最大上下文长度有限**（比如 2048 或 4096）。
- **直接推理更长序列时性能急剧下降**（因为 RoPE 中的频率参数未覆盖更长距离）。

### 2. 已有方法的局限

- **NTK-aware scaling（Neural Tangent Kernel 缩放）**：通过缩放 RoPE 的 base 频率来“拉伸”位置编码，但会牺牲短距离精度。
- **YaRN 提出了一种更精细的插值 + 缩放策略**，在保持短程精度的同时提升长程泛化能力。

## YARN 的核心思想

YARN 在 NTK-aware scaling 基础上引入了以下关键改进：

### 1. 局部窗口内保持原始 RoPE（不缩放）

    对于位置 ≤ 原始训练长度（如 32K），使用原始 RoPE，保留高精度。

### 2. 对超出部分使用缩放后的 RoPE

对于位置 > 训练长度，使用经过缩放的 base 频率（类似 NTK 方法）。

#### NTK-aware Scaling方法

原始的RoPE方法的核心公式为：

$$
\text{RoPE}_m(x) = 
\begin{bmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) \\
\sin(m\theta_0) & \cos(m\theta_0)
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1
\end{bmatrix},
$$

对每对$(x_{2i}, x_{2i+1})$做相同的操作，其中：

$$
\theta_i = b ^ {-2 i / d}, b = 10000 (通常)
$$

所以整个位置m对应的旋转角由 $m \times \theta_i$决定。

当上下文长度超出m的时候，还按照RoPE方法会造成模型性能显著下降。在YaRN之前，NTK-aware Scaling方法是这样处理的，定义缩放因子:

$$
s = L_{target} / L_{train}
$$

然后把原来的base b=10000替换为：

$$
b' = b \times s ^{d / (d-2)}
$$

关于s的指数$d/(d-2)$的由来，可以参考这篇[知乎文章](https://zhuanlan.zhihu.com/p/20328774059)

### 3. 引入平滑过渡（smooth transition）

使用一个 衰减函数（如高斯或线性） 在边界处混合原始和缩放 RoPE，避免突变。

### 4. 动态缩放因子计算

缩放因子基于目标长度 $L_{target}$和原始长度$L_{train}$:

$$
s = max\left(1, \frac{L_{target}}{L_{train}} - δ\right)
$$​

其中 δ 是一个小的偏移量（如 0.1），用于保留一定冗余。

### 5. 代码实现

```python
def get_yarn_rpe(q_len, dim, original_max_pos, target_max_pos, device, dtype):
    # 1. 计算缩放因子（带偏移 alpha=0.1）
    scale = target_max_pos / original_max_pos
    alpha = 0.1
    if scale <= 1.0:
        s = 1.0
        base = 10000.0
    else:
        s = max(1.0, scale - alpha)          # ← YaRN 的关键：减去 alpha
        base = 10000.0 * (s ** (dim / (dim - 2)))  # ← 修改 base

    # 2. 构造频率
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(q_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)  # shape: [q_len, dim//2]

    # 3. 构造 cos/sin
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1,1,L,d/2]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin
```

上面的代码只提供了频率计算部分，将上面的计算结果代入RoPE公式中，就可以按照RoPE公式计算得到旋转矩阵，像RoPE一样的使用了。

### 6. 实用的做法：局部不缩放，远处缩放 + 平滑过渡


YaRN 实际常用策略（简化版）：

- 如果当前位置 m≤$L_train$：用原始 RoPE（θi）
- 如果$m>L_{train}$：用缩放 RoPE（θi′）
- 在边界附近（比如 $m∈[L_{train}−Δ,L_{train}+Δ]$）用线性或高斯加权混合两者

不过，在开源实现中，最常用的是“全局使用缩放后的 base，但保留原始高频结构”，并通过实验调整 α 来平衡长短程性能。

## YARN 的优势总结

| 特性 | 说明 |
|------|------|
| ✅ 无需重新训练 | 可直接用于已有 RoPE 模型 |
| ✅ 保持短程精度 | 局部窗口内使用原始 RoPE |
| ✅ 支持超长上下文 | 如 32K、64K tokens |
| ✅ 即插即用 | 替换 RoPE 计算即可 |

## 缩放方案比较

| 方法 | 旋转角度 $\phi(m,i)$ | 特点 |
|------|------------------------|------|
| 原始 RoPE | $m \cdot \frac{1}{b^{2i/d}}$ | 训练长度内好，外推差 |
| NTK-scaling | $m \cdot \frac{1}{(bs^{d/(d-2)})^{2i/d}} = \frac{m}{s^{2i/(d-2)}} \cdot \frac{1}{b^{2i/d}}$ | 全局缩放，短距精度损失 |
| YaRN | $m \cdot \frac{1}{(bs^{d/(d-2)})^{2i/d}}$，但 $s = \frac{L_{\text{target}}}{L_{\text{train}}} - \alpha$，且可加平滑 | 保留更多高频信息，效果更好 |
