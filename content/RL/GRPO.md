---
title: "GRPO - Group Relative Policy Optimization"
date: 2025-01-13
math: true
---

### 1.背景

在传统 PPO 中，目标函数为：
$$
L^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$
其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是重要性采样比；
- $\hat{A}_t$ 是通过 critic 估计的优势函数（如 GAE）。

**GRPO 的关键突破**：  
> **不依赖 critic，而是通过同一 prompt 下多个 responses 构造一个无偏/低方差的“组内相对优势估计”**，并保留重要性采样结构。

---

### 2. 形式化定义

#### （1）数据生成
- 对每个 prompt $x$，使用**旧策略 $\pi_{\text{old}}$**（即当前策略的快照）生成 $K$ 个 responses：$\{y^{(1)}, \dots, y^{(K)}\}$
- 每个 response $y^{(i)}$ 由 token 序列组成，记其轨迹为 $\tau_i = (x, y^{(i)})$

#### （2）奖励与优势构造
- 使用奖励模型获得标量奖励：$r_i = \text{ORM}(x, y^{(i)})$
- 定义**组内相对优势（Group-Relative Advantage）**：
  $$
  \hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}
  $$
  > 即：每个 response 的优势 = 其奖励 - 组内平均奖励。这相当于一种**去中心化的优势估计**，无需 value function。

#### （3）策略梯度目标（带重要性采样）
GRPO 的目标函数为：
$$
\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ 
\frac{1}{G} \sum_{i=1}^G 
\underbrace{
\frac{\pi_\theta(y^{(i)} | x)}{\pi_{\text{old}}(y^{(i)} | x)}
}_{\text{importance weight } \rho_i}
\cdot \hat{A}_i
\right]
$$

> 这与 REINFORCE 的重要性采样形式一致，但优势项 $\hat{A}_i$ 来自组内比较而非 critic。

#### （4）加入裁剪（Clipping）以稳定训练（可选但推荐）
类似于 PPO，GRPO 也可引入裁剪机制防止策略更新过大：
$$
\mathcal{L}_{\text{GRPO}}^{\text{clip}}(\theta) = \mathbb{E}_{x} \left[ 
\frac{1}{G} \sum_{i=1}^G 
\min\left(
\rho_i \hat{A}_i,\ 
\text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i
\right)
\right]
$$

#### （5）KL 正则化（防止策略崩溃）
通常还会加上对 reference policy（如 SFT 模型 $\pi_{\text{ref}}$）的 KL penalty：
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GRPO}}^{\text{clip}} - \lambda \cdot \mathbb{E}_{x,y \sim \pi_\theta} \left[ D_{\text{KL}} \left( \pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x) \right) \right]
$$

---

### 3. 为什么这样有效？

- **组内平均作为 baseline**：$r-\text{mean}(r)$ 起到了类似 value function 的作用（降低方差），但无需训练 critic。
- **保留重要性采样**：确保 off-policy 更新的无偏性（因为 responses 是从 $\pi_{\text{old}}$ 采样的，但我们要更新 $\pi_\theta$）。
- **天然支持 batch 内多响应**：非常适合 LLM 并行生成多个候选回复的场景。

---

### 4. 更准确的 Python 伪代码（含重要性采样）

```python
import torch
import torch.nn.functional as F

def compute_log_prob(model, input_ids, output_ids):
    """计算 output_ids 在给定 input_ids 下的 log probability"""
    # 拼接完整序列
    full_seq = torch.cat([input_ids, output_ids], dim=1)
    logits = model(full_seq).logits[:, -output_ids.shape[1]-1:-1, :]  # 取生成部分的 logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(2, output_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum(dim=1)  # [batch_size]

def grpo_step_correct(
    prompts: list[str],
    policy_model,
    old_policy_model,      # π_old (frozen during sampling)
    ref_model,             # π_ref for KL
    reward_model,
    tokenizer,
    K=4,
    eps=0.2,
    beta_kl=0.01,
    max_new_tokens=128
):
    device = policy_model.device
    total_loss = 0.0
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-7)

    for prompt in prompts:
        # Tokenize prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]

        # 用 π_old 生成 K 个 responses
        with torch.no_grad():
            outputs = old_policy_model.generate(
                input_ids.repeat(K, 1),
                do_sample=True,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 提取生成的 token
        gen_ids = [seq[input_len:] for seq in outputs.sequences]

        # 过滤空响应
        valid_indices = [i for i, ids in enumerate(gen_ids) if len(ids) > 0]
        if len(valid_indices) < 2:
            continue
        gen_ids = [gen_ids[i] for i in valid_indices]
        K_eff = len(gen_ids)

        # 计算 rewards
        rewards = []
        for ids in gen_ids:
            resp = tokenizer.decode(ids, skip_special_tokens=True)
            r = reward_model(prompt, resp)  # scalar
            rewards.append(r)
        rewards = torch.tensor(rewards, device=device)  # [K_eff]

        # 计算组内相对优势: A_i = r_i - mean(r)
        advantages = rewards - rewards.mean()

        # 计算 log π_old(y|x) 和 log π_θ(y|x)
        old_logprobs = []
        new_logprobs = []
        ref_logprobs = []

        for ids in gen_ids:
            ids = ids.unsqueeze(0)
            # 注意：ids 可能长度不同，这里简化假设等长或已 padding
            with torch.no_grad():
                old_lp = compute_log_prob(old_policy_model, input_ids, ids)
                ref_lp = compute_log_prob(ref_model, input_ids, ids)
            new_lp = compute_log_prob(policy_model, input_ids, ids)

            old_logprobs.append(old_lp)
            new_logprobs.append(new_lp)
            ref_logprobs.append(ref_lp)

        old_logprobs = torch.cat(old_logprobs)  # [K_eff]
        new_logprobs = torch.cat(new_logprobs)  # [K_eff]
        ref_logprobs = torch.cat(ref_logprobs)  # [K_eff]

        # 重要性采样比 ρ = π_θ / π_old
        ratios = torch.exp(new_logprobs - old_logprobs)  # [K_eff]

        # GRPO clipped objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps, 1 + eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty w.r.t reference model
        kl = (new_logprobs - ref_logprobs).mean()
        loss = policy_loss + beta_kl * kl

        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()
    return total_loss / len(prompts)
```

---

### 5. 总结：GRPO 的关键要素

| 组件 | 是否包含 | 说明 |
|------|--------|------|
| 重要性采样 $\frac{\pi_\theta}{\pi_{\text{old}}}$ | ✅ | 保证 off-policy 更新正确性 |
| 相对优势 $\hat{A}_i = r_i - \bar{r}$ | ✅ | 组内去中心化，替代 critic |
| Clipping | ✅（可选） | 提高稳定性，类似 PPO |
| KL 正则化 | ✅（推荐） | 防止语言模型退化 |
| Value Network | ❌ | 完全不需要 |

---
