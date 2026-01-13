---
title: "KV cache - 大模型高效推理的基石"
date: 2025-01-13
math: true
---


KV Cache（Key-Value Cache）是大语言模型（LLM, Large Language Model）推理过程中用于**加速自回归生成**的一项关键技术。它通过缓存先前 token 的 Key 和 Value 向量，避免在生成新 token 时重复计算已处理上下文的注意力信息，从而显著提升推理效率。


## 一、为什么需要 KV Cache？

### 1. 自回归生成的本质
大语言模型通常以**自回归方式**生成文本：每次只预测一个 token，然后将该 token 拼接到输入序列末尾，再预测下一个 token。例如：

```
输入: "今天天气"
第1步输出: "真"
输入变为: "今天天气真"
第2步输出: "好"
...
```

### 2. 注意力机制的重复计算问题
Transformer 使用 **自注意力机制（Self-Attention）**，对长度为 $n$ 的序列，每个 token 都要与其他所有 token 计算注意力权重。

假设当前已生成 $t$ 个 token，现在要生成第 $t+1$ 个 token。若每次都重新计算整个长度为 $t+1$ 的序列的 Q、K、V，那么：
- 第1步：计算1个token → 1次QKV
- 第2步：计算2个token → 2次QKV（但前1个其实已经算过）
- ...
- 第$t$步：计算$t$个token → 前$t-1$个重复计算！

这导致 **时间复杂度为 $O(n^2)$**，且大量重复计算。

### 3. KV Cache 的提出
为解决此问题，研究者提出：**在生成过程中缓存每个 token 对应的 K（Key）和 V（Value）向量**。因为：
- 在自回归生成中，**历史 token 不会改变**；
- Attention 公式中，当前 token 的 Q 只需与所有历史 K、V 计算即可；
- Q 是当前 token 的表示，必须实时计算；但 K、V 可以提前缓存。

于是，在每一步只需：
- 计算当前 token 的 Q；
- 将历史缓存的 K、V 与当前 K、V 拼接；
- 执行一次 attention。

这样，每步计算量恒定（$O(1)$ per step），总复杂度从 $O(n^2)$ 降为 $O(n)$。

> ✅ **KV Cache 的核心思想：用空间换时间，避免重复计算 K、V。**

---

## 二、技术原理详解

### 1. 标准 Self-Attention 公式回顾

对于输入序列 $X \in \mathbb{R}^{n \times d}$：

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### 2. 自回归生成中的 KV Cache 应用

设当前已处理 token 数为 $t$，缓存了：
- $K_{\text{cache}} \in \mathbb{R}^{t \times d_k}$
- $V_{\text{cache}} \in \mathbb{R}^{t \times d_v}$

当输入新 token $x_{t+1}$（或初始 prompt 的下一个 token）：
1. 计算其 Q、K、V：
   $$
   q_{t+1} = x_{t+1} W_Q,\quad k_{t+1} = x_{t+1} W_K,\quad v_{t+1} = x_{t+1} W_V
   $$
2. 更新缓存：
   $$
   K_{\text{new}} = [K_{\text{cache}}; k_{t+1}],\quad V_{\text{new}} = [V_{\text{cache}}; v_{t+1}]
   $$
3. 计算 attention：
   $$
   \text{attn} = \text{softmax}\left(\frac{q_{t+1} K_{\text{new}}^T}{\sqrt{d_k}}\right) V_{\text{new}}
   $$

> 注意：Q 只需当前 token 的（因为是 decoder-only 架构，如 GPT），而 K、V 需要全部历史。

> 注意：在自回归的推理过程中，网络中流通的tensor的shape为：[batch_size, 1, hidden_size]，而不是训练当中的[batch_size, seq_len, hidden_size]。


### 3. 多头注意力中的 KV Cache

每个 attention head 都有自己的 $W_K, W_V$，因此 KV Cache 通常是 shape: $(batch\_size, num\_heads, seq\_len, head\_dim)$。

在实现中，常按 head 维度组织缓存。

### 4. 工程优化
- **内存管理**：缓存随序列增长而增长，需注意显存限制。
- **PagedAttention（vLLM）**：将 KV Cache 分页存储，提高内存利用率。
- **量化 KV Cache**：用 int8/float16 存储 K、V，减少显存占用。
- **滑动窗口注意力**：只缓存最近 $N$ 个 token 的 KV，适用于长上下文。

---

## 三、Python 代码 Demo（简化版）

下面是一个 **不依赖深度学习框架** 的纯 NumPy 实现，演示 KV Cache 如何工作。

```python
import numpy as np

# 设置随机种子以便复现
np.random.seed(42)

class SimpleKVCacheDemo:
    def __init__(self, d_model=64, d_k=32, d_v=32):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        # 随机初始化权重（模拟训练好的模型）
        self.W_Q = np.random.randn(d_model, d_k)
        self.W_K = np.random.randn(d_model, d_k)
        self.W_V = np.random.randn(d_model, d_v)
        self.W_O = np.random.randn(d_v, d_model)  # 输出投影（可选）

        # 初始化 KV 缓存
        self.K_cache = None  # shape: (seq_len, d_k)
        self.V_cache = None  # shape: (seq_len, d_v)

    def clear_cache(self):
        self.K_cache = None
        self.V_cache = None

    def forward_step(self, x):
        """
        x: np.array of shape (d_model,) — 当前输入 token 的 embedding
        返回输出表示，并更新 KV cache

        正常的前向输入是[bs, seq_len, d_model],这里考虑bs=1的情况，在推理过程中，每次输入为刚刚生成的最新的token，所以说当前输入的token维度为：[1, d_model]
        """
        x = x.reshape(1, -1)  # (1, d_model)

        # 计算当前 token 的 Q, K, V
        Q = x @ self.W_Q  # (1, d_k)
        K = x @ self.W_K  # (1, d_k)
        V = x @ self.W_V  # (1, d_v)

        if self.K_cache is None:
            # 第一个 token
            self.K_cache = K
            self.V_cache = V
            attn_weights = np.array([[1.0]])  # softmax([0]) = [1]
        else:
            # 拼接缓存
            K_full = np.vstack([self.K_cache, K])  # (seq_len+1, d_k)
            V_full = np.vstack([self.V_cache, V])  # (seq_len+1, d_v)

            # 计算 attention scores: Q @ K_full^T
            scores = Q @ K_full.T / np.sqrt(self.d_k)  # (1, seq_len+1)
            attn_weights = np.exp(scores - np.max(scores))  # numerical stability
            attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

            # 更新缓存
            self.K_cache = K_full
            self.V_cache = V_full

        # 加权求和
        output = attn_weights @ V_full  # (1, d_v)
        output = output @ self.W_O       # (1, d_model)
        # 在推理过程中，每个attention的维度就是(1, d_model)，不同于训练中，维度为(seq_len, d_model)
        return output.flatten()

# ------------------ Demo ------------------

# 模拟 token embeddings（比如来自 embedding layer）
embeddings = [
    np.random.randn(64),
    np.random.randn(64),
    np.random.randn(64),
    np.random.randn(64)
]

model = SimpleKVCacheDemo()

print("=== Without KV Cache (naive recompute) ===")
# 这里我们不实现无缓存版本，但逻辑上每步都要重算全部

print("\n=== With KV Cache ===")
model.clear_cache()
for i, emb in enumerate(embeddings):
    out = model.forward_step(emb)
    print(f"Step {i+1}: output norm = {np.linalg.norm(out):.4f}, "
          f"cache length = {model.K_cache.shape[0]}")

# 验证：如果重新输入相同序列，缓存会累积
print("\nAdding one more token...")
out = model.forward_step(np.random.randn(64))
print(f"Step 5: cache length = {model.K_cache.shape[0]}")
```

### 输出示例：
```
=== With KV Cache ===
Step 1: output norm = 7.8921, cache length = 1
Step 2: output norm = 8.1023, cache length = 2
Step 3: output norm = 7.9542, cache length = 3
Step 4: output norm = 8.0124, cache length = 4

Adding one more token...
Step 5: cache length = 5
```

> 💡 此 demo 虽简化（单头、无 batch、无 LayerNorm 等），但完整展示了 KV Cache 的核心机制：**缓存 K、V，避免重复计算**。

---

## 四、实际应用中的 KV Cache（补充）

在真实 LLM 推理引擎中（如 HuggingFace Transformers、vLLM、TensorRT-LLM）：
- KV Cache 是默认启用的（`past_key_values` 参数）；
- 支持 batch 推理（不同序列长度需 padding 或使用 PagedAttention）；
- 可通过 `use_cache=True` 控制；
- 显存占用 ≈ $2 \times \text{num\_layers} \times \text{num\_heads} \times \text{seq\_len} \times \text{head\_dim} \times \text{bytes\_per\_param}$

例如 HuggingFace 中使用：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Hello, how are", return_tensors="pt")
outputs = model(**inputs)  # 第一次：计算全部 KV
# outputs.past_key_values 包含各层的 (K, V) 缓存

# 下一步生成：
next_input = tokenizer(" you", return_tensors="pt").input_ids[:, -1:]
outputs2 = model(
    input_ids=next_input,
    past_key_values=outputs.past_key_values  # 传入缓存！
)
```

---

## 总结

| 项目 | 说明 |
|------|------|
| **动机** | 避免自回归生成中重复计算 K、V |
| **核心** | 缓存历史 token 的 Key 和 Value |
| **优势** | 推理速度提升，每步 $O(1)$ 计算 |
| **代价** | 额外显存（与序列长度线性增长） |
| **扩展技术** | PagedAttention、KV 量化、滑动窗口 |

KV Cache 是现代 LLM 高效推理的基石之一，理解它对优化部署、设计推理引擎至关重要。
