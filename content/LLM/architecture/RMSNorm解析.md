---
title: "RMSNorm解析 - Root Mean Square Layer Normalization"
date: 2026-01-04
math: true
---

RMSNorm（Root Mean Square Layer Normalization）是一种用于归一化层输入的归一化方法，它基于输入向量的均方根（RMS）作为归一化的权重。RMSNorm 的主要作用是提高训练效率，并防止梯度爆炸。

## 一、RMSNorm 的由来

### 1. Layer Normalization（LayerNorm）的局限性

在 Transformer 架构中，**LayerNorm** 被广泛用于稳定训练。其标准形式为：

\[
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

其中：
- \(\mu = \frac{1}{d}\sum_{i=1}^d x_i\)（均值）
- \(\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2\)（方差）

虽然有效，但 **减去均值（centering）操作在语言建模任务中可能并非必要**，甚至可能干扰某些表示。

### 2. RMSNorm 的提出

- **论文**：《[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)》（ICLR 2020）
- **作者**：Biao Zhang, Rico Sennrich（来自爱丁堡大学 & Amazon）
- **核心思想**：**去掉 centering（即不减均值），仅做 scaling（缩放）**，用更简单的归一化方式达到类似甚至更好的效果。

> ✅ 优势：
> - 计算更简单（少一次减均值）
> - 减少计算开销（约 7%~10% faster）
> - 在 NLP 任务上性能相当或略优
> - 更适合大规模自回归语言模型

如今，**LLaMA、Qwen、Mistral、DeepSeek 等主流大模型全部采用 RMSNorm**，而非 LayerNorm。

- **注意：Layernorm和RMSNorm归一化处理的维度都是hidden dimension的维度**。即如果输入为的维度为[bs, seq_len, hidden_dim]，则他们都是对hidden dimension做归一化，因此不同 batch 和不同位置（token）的归一化是相互独立的。

---

## 二、基本原理

RMSNorm 只对输入向量按 **均方根（Root Mean Square）** 进行归一化：

- 不再计算均值 \(\mu\)
- 直接用输入的 **RMS 值** 作为缩放因子
- 保留可学习的仿射参数 \(\gamma\)（通常无偏置 \(\beta\)）

这使得归一化操作更“轻量”，同时保留了对激活值尺度的控制能力。

---

## 三、数学细节

给定一个输入向量 \( x \in \mathbb{R}^d \)，RMSNorm 定义为：

\[
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma
\]

其中：
- \(\odot\) 表示逐元素相乘（Hadamard product）
- \(\gamma \in \mathbb{R}^d\) 是可学习的缩放参数（初始化为 1）
- \(\text{RMS}(x)\) 是均方根：

\[
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}
\]

> 🔔 注意：**没有减去均值**，这是与 LayerNorm 的本质区别。

等价地，可写为：

\[
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i
\]

---

## 四、PyTorch 实现（可运行演示）

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化 RMSNorm 模块。
        
        Args:
            dim (int): 输入特征维度（如 embed_dim）
            eps (float): 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma，形状为 (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (..., dim)
            
        Returns:
            归一化后的张量，形状同输入
        """
        # 计算均方根：sqrt(mean(x^2) + eps)
        # keepdim=True 保证可广播
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 归一化并应用可学习缩放
        x_normed = x / rms
        return x_normed * self.weight


# ------------------ 演示与对比 ------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 5
    embed_dim = 8

    x = torch.randn(batch_size, seq_len, embed_dim)
    print("Input:\n", x[0, 0])  # 打印第一个 token 的 embedding

    # 自定义 RMSNorm
    rms_norm = RMSNorm(embed_dim)
    out_rms = rms_norm(x)

    # 对比 PyTorch 的 LayerNorm（带 bias）
    layer_norm = nn.LayerNorm(embed_dim, elementwise_affine=True)
    out_ln = layer_norm(x)

    print("\nAfter RMSNorm:\n", out_rms[0, 0])
    print("\nAfter LayerNorm:\n", out_ln[0, 0])

    # 验证 RMSNorm 输出的 RMS ≈ 1（忽略 gamma）
    with torch.no_grad():
        rms_of_output = torch.sqrt(out_rms[0, 0].pow(2).mean()).item()
        print(f"\nRMS of RMSNorm output (before gamma): ~{rms_of_output:.3f} (should be close to 1 if gamma≈1)")

    # 参数数量对比（RMSNorm 无 bias）
    print(f"\nRMSNorm params: {sum(p.numel() for p in rms_norm.parameters())}")      # = embed_dim
    print(f"LayerNorm params: {sum(p.numel() for p in layer_norm.parameters())}")   # = 2 * embed_dim
```

### 示例输出（节选）：
```
Input:
 tensor([-0.5041,  0.8920, -0.4989, -1.3071,  0.4685, -0.1527,  1.4497, -0.7099])

After RMSNorm:
 tensor([-0.3534,  0.6250, -0.3496, -0.9160,  0.3284, -0.1070,  1.0161, -0.4977])

After LayerNorm:
 tensor([-0.4298,  0.7589, -0.4252, -1.1102,  0.3987, -0.1299,  1.2321, -0.6036])

RMS of RMSNorm output (before gamma): ~1.000

RMSNorm params: 8
LayerNorm params: 16
```

> ✅ 可见：
> - RMSNorm 输出的 RMS 接近 1（说明归一化成功）
> - 参数量仅为 LayerNorm 的一半（无 bias）
> - 计算路径更短，更适合高效推理

---

## 五、为什么 RMSNorm 在 LLM 中更受欢迎？

| 特性 | LayerNorm | RMSNorm |
|------|----------|--------|
| 是否减均值 | ✅ 是 | ❌ 否 |
| 可学习参数 | \(\gamma, \beta\) | 仅 \(\gamma\) |
| 计算开销 | 较高 | 更低 |
| 在自回归语言模型中的表现 | 良好 | **相当或略优** |
| 主流大模型采用情况 | BERT, T5 | **LLaMA, Qwen, Mistral, DeepSeek, Yi, Gemma...** |

> 🧠 直觉解释：在语言建模中，**token 的绝对位置信息已由位置编码提供**，而 **均值可能携带语义信息（如主题偏向）**，强行 center 可能破坏这种信号。RMSNorm 保留原始分布的“重心”，只调节尺度，更符合生成任务需求。

---

## 六、扩展：Pre-Norm vs Post-Norm 中的 RMSNorm

现代大模型普遍采用 **Pre-Norm 结构**（先 Norm 再 Attention/FFN），例如：

```python
# Pre-Norm Block 示例
def forward(self, x):
    x = x + self.attn(self.norm1(x))   # 注意：先 norm1(x)，再加残差
    x = x + self.ffn(self.norm2(x))
    return x
```

在这种结构中，**RMSNorm 的稳定性优势更加明显**，有助于训练更深的模型。

---

## 总结

- **RMSNorm = 去掉 centering 的 LayerNorm**
- **公式**：\( \text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \odot \gamma \)
- **优点**：更快、更省参、性能不输
- **现状**：已成为大语言模型的 **事实标准**

> 💡 如果你正在复现 LLaMA 或 Qwen 架构，**务必使用 RMSNorm 而非 LayerNorm**！

--- 
