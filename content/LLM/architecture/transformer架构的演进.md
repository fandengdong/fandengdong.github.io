---
title: "综述 - Transformer架构的演进"
date: 2025-12-19
math: true
---

## Transformer架构演进：从自注意力到长序列与多模态的革命性突破

Transformer自2017年提出以来，已成为深度学习领域的基石架构，其演进历程反映了AI研究者对计算效率、长序列处理能力和跨模态应用的不懈追求。**从最初的O(n²)计算复杂度到如今的O(n)线性复杂度，从单模态文本处理到原生多模态架构，Transformer的演进不仅解决了自身局限，更推动了大语言模型、视觉模型和多模态系统的革命性发展**。这些演进主要体现在注意力机制创新、位置编码改进、模型架构变体与扩展、训练推理效率优化四大方向，共同构建了现代AI系统的计算基础。

### 一、注意力机制的演进：从全局到稀疏再到状态空间模型

Transformer原始架构的核心是自注意力机制，它通过计算查询(Q)、键(K)和值(V)之间的相似度来捕捉序列内部的长距离依赖关系  。标准自注意力的计算公式为：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V
$$

其中，$Q∈ℝ^{n×d_k}$，$K∈ℝ^{n×d_k}$，$V∈ℝ^{n×d_v}$，d_k是键向量的维度。这一机制虽然强大，但计算复杂度为O(n²)，随着序列长度n的增加呈平方增长，严重限制了处理长文本的能力。

为解决这一瓶颈，研究者从多个角度对注意力机制进行了创新：

**1. 稀疏注意力机制**

稀疏注意力通过限制每个token只能关注序列中的局部区域，将计算复杂度从O(n²)降至O(n)。主要代表包括：

- **Longformer**：采用滑动窗口注意力机制，每个token仅关注其周围固定窗口内的token，同时保留少数全局token以维持上下文连贯性  。其窗口注意力公式为：

$$
\text{WindowAttention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V_{\text{window}}
$$

  其中，V_window是窗口内值向量的子集，窗口大小通常为512或1024个token  。

- **BigBird**：结合三种注意力模式——随机注意力（通过Erdős-Rényi图采样）、滑动窗口注意力和全局注意力  。随机注意力使模型能够捕捉长距离依赖，同时保持O(n)的线性复杂度。其随机注意力矩阵可以表示为：

$$
\text{RandomAttention}(Q,K,V) = \text{softmax}\left( \frac{QW_{\text{random}}K^\top}{\sqrt{d_k}} \right)V_{\text{random}}
$$

  其中，$W_{random}$是随机掩码矩阵，$V_{random}$是随机采样值向量的子集  。

- **局部增强Mamba（LEVM）**：在Mamba架构中，通过动态选择性处理，对重要信息保留完整状态，对不重要信息进行压缩，实现O(n)计算复杂度和O(1)显存占用  。其核心状态方程为：

$$
h_t = \overline{A}h_{t-1} + \overline{B}x_t
$$

  其中，$\overline{A}$和$\overline{B}$是动态生成的参数矩阵，允许模型根据输入选择性地保留信息  。

**2. 线性注意力机制**

线性注意力通过数学近似技术，将注意力计算的复杂度降至O(n)，同时保持模型的表达能力。主要代表包括：

- **Performer**：采用核函数近似方法，将标准注意力公式中的softmax(QKᵀ)替换为核函数K(Q,K)，使得计算可以分解为O(n)的线性操作  。其核心公式为：

$$
\text{LinearAttention}(Q,K,V) = \text{核函数}(Q,K)V
$$

  其中，核函数可以是高斯核、多项式核或其他类型的核函数  。

- **Mamba**：将注意力机制替换为结构化状态空间模型（SSM），通过递归计算实现长序列处理  。Mamba在S4的基础上改进，引入选择性处理机制，允许模型根据输入动态生成参数$\overline{A}$, $\overline{B}$, Δ，从而实现对信息的选择性保留和压缩  。其状态方程为：

$$
  h_t = \overline{A}h_{t-1} + \overline{B}x_t,\quad y_t = Ch_t
$$

  其中，$h_t$是隐藏状态，$y_t$是输出，A、B、C是可学习的参数矩阵  。

- **Hyena**：通过长卷积和门控机制组合，递归定义算子，复杂度O(n)，训练速度比Transformer快100倍（序列长度128k时）  。Hyena算子的核心公式为：

$$
  y_t = \text{GatedConv}(x_t, h_{t-1})
$$

  其中，GatedConv表示门控卷积操作，允许模型在长序列上高效捕捉依赖关系  。

**3. 状态空间模型（SSM）替代**

状态空间模型通过将序列建模视为状态演化过程，彻底改变了Transformer的计算范式。主要代表包括：

- **S4**：基于连续时间SSM，离散化后公式为：

$$
  h_t = \overline{A}h_{t-1} + \overline{B}x_t,\quad y_t = Ch_t
$$

  其中，$\overline{A} = \exp(ΔA)$，$\overline{B} = △B·(\exp(ΔA)-I)$，Δ是步长参数  。S4将注意力替换为卷积形式，支持无限序列长度  。

- **Mamba**：在S4基础上改进，通过输入动态生成参数\overline{A}, \overline{B}, Δ，实现选择性信息处理（丢弃不重要信息，保留重要信息）  。Mamba架构是SSM与Transformer的MLP块的结合，两者相互融合，而非重叠，形成了更简单的结构  。

- **Titans**：结合RNN的响应速度与Transformer的性能优势，通过动态记忆更新机制实现运行时参数迭代更新  。Titans架构的核心公式为：

$$
\text{Memorize}(x_t) = \sigma_s(x_t) \cdot x_t,\quad M_t = \lambda M_{t-1} + \text{Memorize}(x_t)
$$

  其中，σ_s(x_t)是"惊讶度"门控，决定哪些信息值得被记住；λ是衰减因子，控制长期记忆的更新频率  。

下表总结了不同注意力机制的复杂度、优缺点及典型应用场景：

| 注意力机制 | 计算复杂度 | 显存复杂度 | 优点 | 缺点 | 典型应用场景 |
|------------|------------|------------|------|------|--------------|
| 标准自注意力 | O(n²) | O(n²) | 全局依赖建模能力强 | 长序列处理效率低 | 短文本理解与生成 |
| Longformer | O(n) | O(n) | 支持长序列处理 | 全局信息建模能力有限 | 长文档理解 |
| BigBird | O(n) | O(n) | 保留全局信息的同时支持长序列 | 实现复杂度较高 | 跨模态长序列处理 |
| Performer | O(n) | O(n) | 精确近似标准注意力 | 需要核函数近似 | 高精度长序列建模 |
| Mamba | O(n) | O(1) | 超长序列处理能力，显存恒定 | 需要重新训练模型 | 超长文本、语音、视频处理 |
| Titans | O(n) | O(n) | 200万token上下文窗口表现优异 | 需要复杂记忆门控 | 超长上下文推理任务 |

### 二、位置编码与归一化方法的改进：从绝对到相对再到旋转编码

位置编码是Transformer处理序列数据的关键组件，其演进主要围绕解决长序列外推能力和计算效率问题：

**1. 位置编码的演进**

- **可学习位置嵌入**：BERT、GPT等模型使用可训练的position embedding，将位置信息直接添加到输入向量中  。其公式为：

$$
  X_{\text{pos}} = X + E_{\text{pos}}
$$

  其中，X是输入向量，E_pos是位置嵌入向量  。

- **相对位置编码（Transformer-XL）**：通过相对位置偏置$B_{i-j}$调整注意力分数，使模型能够处理超出训练时最大序列长度的数据  。其核心公式为：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top + \sum_{l=1}^L (QW_{Q,l} \cdot p_{i-j,l}^K)}{\sqrt{d_k}} \right) \cdot (VW_{V} + \sum_{l=1}^L p_{i-j,l}^V)
$$

  其中，$p_{i-j,l}^K$和$p_{i-j,l}^V$是相对位置编码向量，$W_{Q,l}$和$W_V$是权重矩阵  。

- **旋转位置编码（RoPE）**：通过旋转矩阵编码相对位置，支持外推到长序列  。RoPE的旋转矩阵构造公式为：

$$
  R_i = \text{diag}(\cos\theta_i, \sin\theta_i, \dots), \quad \theta_i = 10000^{-2(i-1)/d_k}
$$

  其中，d_k是注意力头维度，位置i的查询/键向量$q_i$, $k_j$被拆分为两半（维度d_k/2），分别与$R_i$和$R_j$相乘后合并：

$$
  q_i' = [R_i q_i^{\text{first}}, R_i q_i^{\text{second}}], \quad k_j' = [R_j k_j^{\text{first}}, R_j k_j^{\text{second}}]
$$

  最终注意力分数为$q_i'·k_j'$，通过复数空间的旋转编码相对位置  。

- **ALiBi（线性偏置）**：用线性偏置替代位置嵌入，提升长上下文泛化能力  。ALiBi的注意力计算公式为：

$$
\text{Attention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top + M}{\sqrt{d_k}} \right)V
$$

  其中偏置矩阵M_{ij} = -m·(i-j)，m是按头设置的几何序列斜率（如m_h = 2^{-(h-1)}），直接通过线性偏置抑制远距离注意力  。

- **T5偏置方法**：与ALiBi类似但使用可学习偏置，使模型能够自适应地调整不同位置的注意力权重  。其公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^\top + \text{PosBias}}{\sqrt{d_k}} \right) V
$$

  其中，pos_bias是可学习的相对位置偏置矩阵，维度为(2L+1)×d_k，L是最大偏置距离  。

**2. 归一化方法的改进**

归一化方法在Transformer训练中扮演着关键角色，其演进主要围绕提高训练稳定性和计算效率：

- **LayerNorm（层归一化）**：原始Transformer使用层归一化稳定训练过程，公式为：

$$
\text{LayerNorm}(x) = \frac{x - \mu(x)}{\sqrt{\sigma^2(x) + \epsilon}} \cdot \gamma + \beta
$$

  其中，μ(x)是x的均值，σ²(x)是x的方差，γ和β是可学习的缩放和平移参数  。

- **RMSNorm（均方根归一化）**：LLaMA等模型采用RMSNorm去除偏置和均值中心化，提升训练稳定性  。其公式为：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{\|x\|_2^2}{d} + \epsilon}} \cdot \gamma
$$

  其中，d是输入维度，γ是可学习的缩放参数  。RMSNorm的参数量仅为LayerNorm的1/3，计算效率更高，适合超大模型（如LLaMA）  。

- **LayerScale**：在残差连接前乘以可学习缩放因子α，缓解梯度消失  。其公式为：

$$
\text{Output} = \text{FFN}(\text{Attention}(x)) + \alpha \cdot x
$$

  其中，α是一个可学习的标量，通常初始化为0.1，逐步学习调整梯度流  。

- **Pre-LN与Post-LN**：T5等模型将LayerNorm放在残差连接的外面，而非里面，提高了训练效率  。Pre-LN的公式为：

$$
\text{Output} = \text{LayerNorm}(x) + \text{FFN}(\text{Attention}(\text{LayerNorm}(x)))
$$

  而Post-LN的公式为：

$$
\text{Output} = \text{LayerNorm}(\text{FFN}(\text{Attention}(\text{LayerNorm}(x))) + x)
$$

  Pre-LN在训练过程中表现出更好的稳定性，尤其是在长序列任务中  。

这些位置编码和归一化方法的演进，使Transformer能够更好地处理长序列数据，同时提高了训练效率和模型性能。**RoPE和ALiBi的出现，使得Transformer在处理长序列时不再依赖固定的上下文窗口，而是能够通过相对位置编码自适应地调整注意力分布，大大扩展了模型的应用范围**  。

### 三、模型架构的变体与扩展：从单模态到多模态的全面突破

Transformer架构的演进不仅体现在注意力机制和位置编码上，更体现在整体架构的变体与扩展上：

**1. 基础架构变体**

- **Encoder-only**：BERT、RoBERTa等模型采用纯编码器结构，适合理解类任务（如分类、问答）  。其架构为多层自注意力+前馈网络的堆叠，没有解码器部分  。

- **Decoder-only**：GPT系列采用纯解码器结构，通过掩码自注意力实现生成任务，成为大语言模型（LLM）的主流架构  。其掩码自注意力公式为：

$$
\text{MaskedAttention}(Q,K,V) = \text{softmax}\left( \frac{QK^\top + \text{mask}}{\sqrt{d_k}} \right)V
$$

  其中，mask是一个下三角矩阵，确保每个位置只能关注之前的token  。

- **Encoder-Decoder**：T5、BART等模型采用编码器-解码器结构，适合序列到序列的任务（如翻译、摘要）  。其核心思想是将源序列编码为中间表示，然后解码器基于该表示生成目标序列  。

**2. 多模态架构扩展**

- **Vision Transformer（ViT）**：将图像分块为token，直接应用Transformer处理视觉数据，性能超越CNN  。ViT的输入处理公式为：

$$
  X_{\text{patch}} = \text{Flatten}(\text{Patchify}(I)) + E_{\text{pos}}
$$

  其中，I是输入图像，Patchify将图像分割为固定大小的块，Flatten将块展平为向量，E_pos是位置嵌入  。

- **NEO架构**：商汤科技与南洋理工大学联合团队发布的原生多模态架构，通过统一视觉语言处理层实现原生多模态建模  。NEO的三大核心创新包括：

  - **原生图块嵌入**：直接从像素构建连续token映射，无需离散tokenizer，提升图像细节捕捉能力  。
  - **原生三维旋转位置编码（Native-RoPE）**：视觉维度高频、文本维度低频，支持跨模态位置建模  。
  - **统一多头注意力**：文本用掩码自注意力，视觉用双向注意力，参数共享  。

  **NEO仅用3.9亿图像文本对训练即达到模块化旗舰模型精度，标志着多模态模型从"拼凑"时代迈入"原生"时代**  。

- **Mamba的多模态应用**：Mamba架构在计算机视觉领域也有广泛应用，如LEVM（局部增强Mamba）模块用于图像分割和分类  。在语音分离中，采用双路径Mamba（DPMamba）结合长短时双向状态空间模型，提升时序建模能力  。

**3. 混合专家（MoE）架构**

- **Switch Transformer**：使用MoE范式，将前馈网络（FFN）层替换为稀疏Switch FFN层  。其门控机制公式为：

$$
  G = \text{softmax}(W_g \cdot x)
$$

  其中，G∈ℝ^E是专家选择向量，E是专家数量  。每个token仅激活Top-k个专家（如k=2），输出为：

$$
\text{Output} = \sum_{e=1}^E G_e \cdot \text{Expert}_e(x)
$$

  Switch Transformer拥有1.6万亿参数，是迄今为止规模最大的NLP模型，在相同计算资源下将预训练速度提高了7倍  。

- **GQA（分组查询注意力）**：将查询头分为g组，每组共享K和V，公式为：

$$
  K_g = \frac{1}{n_h/g} \sum_{i \in \text{group}_g} W_{K,i}X,\quad V_g = \frac{1}{n_h/g} \sum_{i \in \text{group}_g} W_{V,i}X
$$

  其中，n_h是总查询头数，g是分组数  。当g=1时，GQA等价于MQA；当g=n_h时，GQA等价于MHA  。

- **Mamba-MoE**：Pióro et al.和Anthony et al.将Mamba架构与MoE结合，实现了对长序列和复杂任务的高效建模  。Mamba-MoE在临床笔记生成等任务中表现出色，证明了状态空间模型与MoE的协同潜力  。

**4. 长序列处理架构**

- **Transformer-XL**：引入循环机制和相对位置编码，允许模型跨段处理长序列，突破了固定上下文窗口的限制  。其循环机制公式为：

$$
  h_t = f(h_{t-1}, x_t),\quad y_t = g(h_t, x_t)
$$

  其中，h_t是隐藏状态，f是更新函数，g是检索函数  。

- **Titans架构**：谷歌在NeurIPS 2025大会上发布的架构，结合RNN的响应速度与Transformer的性能优势  。Titans架构包含三个模块：核心模块（短期记忆）、长期记忆模块和持久记忆模块（固定参数存储任务先验知识）  。在BABILong基准测试中，Titans在200万token上下文窗口下表现最优  。

这些架构变体与扩展，使Transformer能够适应不同任务和数据模态的需求，从纯文本处理扩展到视觉、语音、多模态等复杂场景，大大扩展了模型的应用范围。**多模态架构的演进，特别是NEO等原生多模态架构的出现，标志着AI模型从"文本为中心"向"多模态融合"的转变，为构建更接近人类认知能力的通用AI奠定了基础**  。

### 四、训练与推理效率的优化方向：从KV缓存到并行生成

随着Transformer模型规模的扩大，训练和推理效率成为制约模型应用的关键因素，研究者从多个角度进行了优化：

**1.KV缓存优化**

KV缓存是Transformer推理时的主要瓶颈，尤其是处理长序列时。优化方向包括：

- **MQA（多查询注意力）**：所有头共享单个K和V，公式为：

$$
\text{MQA}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

  其中，K = V = W_K X，参数量减少(h-1)/h  。MQA被PaLM、StarCoder、Gemini等模型采用，显存占用降低  。

- **GQA（分组查询注意力）**：将查询头分为g组，每组共享K和V，公式为：

$$
\text{GQA}(Q,K,V) = \text{Concat}(\text{head}_1^g, \dots, \text{head}_g^g) W^O
$$

  其中，K_g是组内参数均值，V_g同理  。GQA-4（4组）在Mixtral中KV缓存减少75%，推理速度提升，且性能损失有限  。

- **TiDAR技术**：英伟达2025年发布的单步生成多个Token的技术，通过结构化注意力掩码将输入分为三个区域：前缀区、验证区和起草区  。TiDAR在15亿参数版本中实现了4.71倍的吞吐量增长，80亿参数版本更达到5.91倍的提升  。

**2. 计算优化**

- **FlashAttention**：通过I/O感知算法减少GPU显存访问，将注意力计算优化为O(n)显存  。FlashAttention将Q和Kᵀ分块为B_r×B_c的小块，逐块计算并缓存中间结果（如行最大值m_i和行和l_i），避免显存中存储完整的N×N矩阵  。

- **内核融合（fused kernels）**：如FlashAttention-2通过合并计算步骤减少内存访问，提高计算效率  。内核融合将多个计算操作（如矩阵乘法、激活函数、归一化等）合并为单一内核，减少数据搬运次数，提高GPU利用率  。

- **HOPE框架**：引入M3优化器，具备"快慢动量"特性，能够同时关注当前的梯度和未来的全局趋势  。M3优化器的公式为：

$$
  v_{\text{fast}} = \beta v_{\text{fast}} + (1-\beta)\nabla_{\text{current}},\quad v_{\text{slow}} = \beta v_{\text{slow}} + (1-\beta)\nabla_{\text{global}}
$$

  其中，v_fast和v_slow分别是快慢动量，β是动量衰减因子  。

**3. 参数优化**

- **HOPE的混合表征机制**：将参数分为高频（短期）、中频（中期）、低频（长期）三层，动态更新：

$$
  \theta_{\text{high}} = \theta_{\text{high}} + \eta \cdot \nabla_{\text{high}},\quad \theta_{\text{low}} = \theta_{\text{low}} + \eta \cdot \nabla_{\text{low}} \cdot \lambda
$$

  其中，λ是衰减因子，控制长期记忆更新频率  。这一机制使模型能够根据信息的重要性动态调整更新频率，提高训练效率  。

- **参数共享**：如GQA分组共享Key/Value参数，减少模型参数量  。参数共享不仅降低了显存占用，还提高了计算效率，使模型能够在有限资源下处理更复杂任务  。

**4. 记忆优化**

- **Titans的记忆门控**：通过"惊讶度"（surprise）门控σ_s控制信息保留：

$$
\text{Memorize}(x_t) = \sigma_s(x_t) \cdot x_t,\quad M_t = \lambda M_{t-1} + \text{Memorize}(x_t)
$$

  其中，σ_s(x_t)决定哪些信息值得被记住；λ是衰减因子，控制长期记忆的更新频率  。

- **HOPE的持续学习能力**：使模型能够在与用户的交互中不断进化，提升自身的智能水平  。HOPE框架支持"原地改造"，现有的AI模型如Llama或Qwen只需重新分配各层的更新频率，无需重训即可获得持续学习的能力  。

这些训练与推理效率的优化方向，使Transformer能够在有限资源下处理更复杂任务，支持更长的上下文窗口，同时保持模型的表达能力和性能  。**TiDAR和HOPE等技术的出现，标志着AI模型从"一次性消耗品"向"持续进化的智能体"的转变，为构建真正意义上的通用人工智能（AGI）奠定了基础**  。

### 五、未来发展趋势：更低复杂度、更低成本、更高效率

Transformer架构的演进仍在继续，未来可能朝着以下几个方向发展：

**1. 计算复杂度进一步降低**

当前的注意力机制优化已经将复杂度从O(n²)降至O(n)，但仍有改进空间。未来可能通过更高效的数学近似、更智能的稀疏策略或完全替代注意力的新机制，进一步降低计算复杂度，使模型能够处理百万甚至千万级token的超长序列。

**2. 记忆机制更加完善**

Titans和HOPE架构提出的长期记忆模块，是解决Transformer短期记忆局限的重要尝试。未来可能发展更复杂的记忆系统，包括多时间尺度记忆、记忆检索机制和记忆压缩算法，使模型能够像人类一样区分短期记忆和长期记忆，并根据任务需求动态调整记忆策略。

**3. 多模态融合更加深入**

NEO等原生多模态架构的出现，标志着多模态模型从"拼凑"向"原生"的转变  。未来可能发展更统一的多模态处理框架，使文本、图像、视频、音频等不同模态的数据能够共享同一套表征和处理机制，实现真正的跨模态理解和生成。

**4. 训练推理一体化**

HOPE框架提出的持续学习能力，使模型能够在与用户的交互中不断进化  。未来可能发展更完善的训练推理一体化机制，使模型在推理过程中也能学习新知识，并将这些知识整合到长期记忆中，实现真正的"边活边学"  。

**5. 硬件感知架构设计**

随着AI硬件的发展，Transformer架构可能更加注重与硬件的协同优化。例如，Mamba架构的硬件感知并行算法，通过扫描（Scan）+ Kernel Fusion，在GPU上实现训练期并行、推理期恒定显存  。未来可能发展更多针对特定硬件（如TPU、专用AI芯片）优化的Transformer变体，提高计算效率和资源利用率。

### 六、总结：Transformer架构演进的四大支柱

Transformer架构的演进可以总结为四大支柱：

**1. 注意力机制创新**：从标准自注意力到稀疏注意力、线性注意力和状态空间模型，解决了长序列处理效率问题  。

**2. 位置编码与归一化改进**：从绝对位置编码到相对位置编码、旋转位置编码和ALiBi，提高了模型对长序列的外推能力；从LayerNorm到RMSNorm和LayerScale，提升了训练稳定性和效率  。

**3. 架构变体与扩展**：从Encoder-only、Decoder-only到Encoder-Decoder、多模态架构，从纯文本处理到视觉、语音、多模态等复杂场景，大大扩展了模型的应用范围  。

**4. 训练与推理效率优化**：从FlashAttention到MQA/GQA、TiDAR和HOPE框架，通过KV缓存优化、计算优化、参数优化和记忆优化，提高了模型的训练和推理效率  。

**这些演进不仅解决了Transformer自身的局限，更推动了AI技术的革命性发展，使大语言模型、视觉模型和多模态系统能够在有限资源下处理复杂任务，为构建更接近人类认知能力的通用AI奠定了基础**。随着技术的不断进步，Transformer架构可能进一步演进，甚至可能被更先进的架构取代，但其核心思想——注意力机制和并行计算——将继续影响AI领域的未来发展。

Transformer的演进历程表明，**真正的AI突破往往来自于对基础架构的重新思考和创新，而非单纯增加参数量或数据量**。未来，随着对人类记忆机制和认知过程的深入理解，Transformer架构可能会更加接近人类大脑的工作方式，为构建真正的通用人工智能铺平道路。

说明：报告内容由千问AI生成，仅供参考。
