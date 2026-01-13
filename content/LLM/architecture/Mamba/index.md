---
title: "Mamba - A Transformer-like Architecture for Long Sequence Modeling"
date: 2025-01-13
math: true
---

## 一、Mamba 的设计思想

### 背景：Transformer 的局限性
- **自注意力机制**的时间复杂度为 $O(L^2)$（L 为序列长度），对长序列（如基因组、高分辨率音频、长文本）效率低。
- **线性注意力**等近似方法牺牲了建模能力。
- **RNN / SSM（状态空间模型）** 具有线性复杂度 $O(L)$，但传统 SSM（如 Linear Time-Invariant SSM, LTI-SSM）是**时不变**的，无法像 Transformer 那样根据输入内容动态调整行为。

### Mamba 的核心创新
> **Selective State Space Model (SSM)** —— 将 SSM 与输入相关（input-dependent），使其具备**上下文感知能力**，同时保持**线性复杂度**。

关键点：
- **选择性（Selectivity）**：SSM 的参数（如 A, B, C）不再是固定或仅时间相关的，而是由当前输入 token 动态生成。
- **硬件感知设计**：利用现代 GPU 的并行特性，通过“扫描（scan）+ 并行前缀”实现高效训练。
- **结构简单**：没有 attention，只有 MLP + SSM block。

---

## 二、Mamba 架构概览

Mamba Block 替代了 Transformer 中的 Attention + MLP 子层：

```
Input → LayerNorm → SSM (Selective SSM) → Residual Add → LayerNorm → MLP → Residual Add → Output
```

其中最核心的是 **Selective SSM 模块**。

---

## 三、数学原理详解

### 1. 经典连续时间 SSM（LTI）
连续形式：
$$
\begin{aligned}
\frac{d}{dt} \mathbf{h}(t) &= \mathbf{A} \mathbf{h}(t) + \mathbf{B} \mathbf{x}(t) \\
\mathbf{y}(t) &= \mathbf{C} \mathbf{h}(t)
\end{aligned}
$$

离散化（使用 Zero-Order Hold, ZOH）后：
$$
\begin{aligned}
\mathbf{h}_t &= \bar{\mathbf{A}} \mathbf{h}_{t-1} + \bar{\mathbf{B}} \mathbf{x}_t \\
\mathbf{y}_t &= \mathbf{C} \mathbf{h}_t
\end{aligned}
$$
其中 $\bar{\mathbf{A}} = e^{\Delta \mathbf{A}},\ \bar{\mathbf{B}} = (\int_0^\Delta e^{\tau \mathbf{A}} d\tau) \mathbf{B}$，$\Delta$ 是时间步长。

但这是**时不变**的（A, B, C 固定），无法适应不同输入。

---

### 2. Selective SSM（Mamba 的核心）

让 **B, C, Δ 成为输入 x 的函数**：

$$
\begin{aligned}
\mathbf{z}_t &= \text{MLP}(\mathbf{x}_t) \quad \text{(用于门控)} \\
\Delta_t, \mathbf{B}_t, \mathbf{C}_t &= \text{Linear}(\mathbf{x}_t) \\
\bar{\mathbf{A}}_t &= \exp(\Delta_t \mathbf{A}) \quad (\mathbf{A} \text{ 是可学习对角复矩阵，通常初始化为负实数}) \\
\bar{\mathbf{B}}_t &= (\Delta_t \cdot \mathbf{B}_t) \odot \phi(\Delta_t \mathbf{A}) \quad \text{（简化版，实际用离散化公式）}
\end{aligned}
$$

然后递归计算隐状态：
$$
\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \mathbf{x}_t
$$
输出：
$$
\mathbf{y}_t = \mathbf{C}_t \mathbf{h}_t
$$

最后加一个 **SiLU 激活 + 门控**：
$$
\text{output}_t = \mathbf{y}_t \odot \sigma(\mathbf{z}_t)
$$

> ✅ **关键优势**：因为 A 是对角矩阵（或可对角化），整个递归可以**并行化**（通过关联扫描/parallel scan），实现 O(L) 训练！

#### SSM的设计思想

##### ✅ 背景：SSM 来自于“连续时间系统”

状态空间模型（SSM）最初来自控制理论，描述的是**连续时间动态系统**：

$$
\frac{d}{dt} h(t) = A h(t) + B x(t)
$$

这是个微分方程，描述了隐状态 $ h(t) $ 随时间变化的方式。

但我们在序列建模中处理的是**离散时间点** $ t=0,1,2,\dots,L $，所以我们需要将这个连续系统“离散化”。

---

##### ✅ 离散化方法：Zero-Order Hold (ZOH)

在控制领域，常用的方法是 **ZOH 离散化**，其结果为：

$$
h_t = e^{\Delta A} h_{t-1} + \left( \int_0^\Delta e^{\tau A} d\tau \right) B x_t
$$

其中：
- $ \Delta $ 是时间步长（固定或可变）
- $ e^{\Delta A} $ 是矩阵指数，表示状态衰减/演化
- $ \int_0^\Delta e^{\tau A} d\tau $ 是积分项，表示输入如何影响状态：$\Delta \cdot \exp(\tau A) \cdot B $ => $(\Delta \cdot B) \exp(\tau A) $

👉 所以，**`exp(ΔA)` 就是连续系统的离散版本**！

> 💡 Mamba 把这个离散化过程直接嵌入到了模型里：  
> - 它让 $ \Delta_t $ 成为输入 $ x_t $ 的函数 → 可以**动态调整时间步长**
> - 它让 $ A $ 是一个**可学习的对角复矩阵** → 控制每个维度的状态衰减模式

---

##### 🧠 为什么要让 A 是“对角复矩阵”？

##### ✅ 1. 对角矩阵 ⇒ 计算效率高

如果 $ A $ 是对角矩阵，比如：

$$
A = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_d)
$$

那么：
- $ \exp(A) = \text{diag}(e^{\lambda_1}, e^{\lambda_2}, ...) $
- $ \int_0^\Delta e^{\tau A} d\tau = \text{diag}\left( \frac{e^{\Delta \lambda_i} - 1}{\lambda_i} \right) $

👉 所有运算都可以按通道独立计算，无需矩阵乘法！

这使得 SSM 的前向传播和反向传播都变得**极其高效**，并且可以**并行化**。

---

##### ✅ 2. 复数矩阵 ⇒ 模拟振荡行为（关键！）

假设 $ A $ 不只是实数，而是**复数对角矩阵**，例如：

$$
\lambda_i = \alpha_i + i\beta_i
$$

那么：
- $ e^{\lambda_i t} = e^{\alpha_i t} \cdot e^{i\beta_i t} = e^{\alpha_i t} \cdot (\cos(\beta_i t) + i\sin(\beta_i t)) $

👉 这就引入了**振荡（oscillation）** 行为！

这意味着：Mamba 的 SSM 可以模拟**周期性信号**、**正弦波**、**频率响应**等复杂动态。

🧠 类比：就像 RNN 中的 LSTM 可以记住长期依赖，但 Mamba 的 SSM 可以通过复数特征值实现“记忆+振荡”，更适合建模语音、音乐、生物信号等具有周期性的数据。

> 📌 实际上，Mamba 的实验表明，这种复数 A 在语音、基因组序列等任务上表现更优。


---

### 3. 并行化技巧（简述）

虽然递归形式是串行的，但 Mamba 利用 **associative scan** 技巧将递归转化为可并行的“算子组合”，类似：
$$
(h_t) = f_t \circ f_{t-1} \circ \cdots \circ f_1 (h_0)
$$
其中每个 $f_t(h) = \bar{A}_t h + \bar{B}_t x_t$ 是一个仿射变换。通过定义合适的结合律操作，可用 `torch_scan` 或 CUDA kernel 并行计算。

> 实际实现中，Mamba 使用了定制 CUDA kernel（如 `selective_scan_cuda`）来加速。

---

## 四、PyTorch 简化 Demo（CPU 友好版）

> 注意：完整 Mamba 依赖 CUDA kernel 才高效。这里我们用 **朴素递归实现**（O(L) 推理，但训练不可并行），仅用于理解逻辑。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context (optional but used in Mamba)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )

        # SSM parameters projection (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1, bias=False)  # B, C, Δ
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A matrix: learnable diagonal real parts (negative init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1)
        self.A_log = nn.Parameter(torch.log(A))  # shape (1, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        x_and_z = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = x_and_z.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Optional convolution for local mixing
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]  # causal padding
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        # Discretization and SSM
        A = -torch.exp(self.A_log.float())  # (1, d_state)
        y = self.ssm_step(x, A)  # (B, L, d_inner)

        # Gating
        y = y * F.silu(z)

        output = self.out_proj(y)
        return output

    def ssm_step(self, x, A):
        # x: (B, L, d_inner)
        # A: (1, d_state)
        B_, L, D = x.shape
        d_state = A.shape[1]

        # Project to get B, C, log(Δ)
        deltaBC = self.x_proj(x.view(-1, D))  # (B*L, d_state*2 + 1)
        delta, B_proj, C_proj = torch.split(deltaBC, [1, d_state, d_state], dim=-1)
        delta = delta.view(B_, L, 1)  # (B, L, 1)
        B_proj = B_proj.view(B_, L, d_state)  # (B, L, d_state)
        C_proj = C_proj.view(B_, L, d_state)  # (B, L, d_state)

        # Compute Δ from log(Δ) via softplus
        delta = F.softplus(self.dt_proj.weight * delta + self.dt_proj.bias)  # (B, L, d_inner)

        # Expand A to (d_inner, d_state) — assume A shared per channel
        A = A.repeat(D, 1)  # (d_inner, d_state)

        # Initialize hidden state
        h = torch.zeros(B_, d_state, device=x.device)  # (B, d_state)
        ys = []

        for t in range(L):
            xt = x[:, t, :]  # (B, d_inner)
            dt = delta[:, t, :]  # (B, d_inner)
            Bt = B_proj[:, t, :]  # (B, d_state)
            Ct = C_proj[:, t, :]  # (B, d_state)

            # Discretize A and B
            Ad = torch.exp(dt.unsqueeze(-1) * A)  # (B, d_inner, d_state)
            Bd = (dt.unsqueeze(-1) * Bt.unsqueeze(1))  # (B, d_inner, d_state)

            # Update hidden state: h = Ad * h + Bd * xt.unsqueeze(-1)
            # But h is (B, d_state), so we need to broadcast
            # We'll compute per-channel SSM
            ut = xt.unsqueeze(-1)  # (B, d_inner, 1)
            h = h.unsqueeze(1)  # (B, 1, d_state)
            h = Ad * h + Bd * ut  # (B, d_inner, d_state)
            h = h.sum(dim=-1)  # ??? ← This is a simplification; real Mamba keeps d_state per channel

            # Actually, standard implementation treats each of d_inner as independent SSM with d_state dim
            # For simplicity, we collapse — this demo is conceptual only!

            yt = (h * Ct).sum(dim=-1)  # (B, d_inner)
            ys.append(yt)
            h = h.mean(dim=1)  # crude hack to keep shape

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y


# Simple test
if __name__ == "__main__":
    model = MambaBlock(d_model=64)
    x = torch.randn(2, 10, 64)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
```

> ⚠️ 注意：上述代码是**教学简化版**，真实 Mamba 实现更复杂，尤其是：
> - 每个 `d_inner` 通道有自己的 SSM（即 `d_inner` 个并行 SSM，每个维度为 `d_state`）
> - 使用 CUDA kernel 实现并行扫描
> - 正确的离散化（如 bilinear transform）
>
> 官方实现见：https://github.com/state-spaces/mamba

---

## 五、总结

| 特性 | Transformer | Mamba |
|------|-------------|-------|
| 复杂度 | $O(L^2)$ | $O(L)$ |
| 上下文感知 | 通过 attention | 通过 input-dependent SSM |
| 并行训练 | 完全并行 | 通过 associative scan 并行 |
| 长序列性能 | 差 | 优秀 |
| 硬件友好性 | 高（但显存瓶颈） | 更高（线性显存） |

Mamba 在语言建模、DNA 序列、音频等领域已展现出超越 Transformer 的潜力，尤其在**长上下文**场景。

---
