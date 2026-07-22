[← 返回 README](../README.md)

# 0. Abstract（摘要）

## 📌 预览

这段摘要把全文的核心矛盾一句话点破：**test-time training（TTT）要往 fast weight 里写什么？**现有的 in-place TTT 方法解决了「怎么写、写在哪」（rank-one 更新、放在 MLP down-projection），但「写什么值（value target）」仍用一个 learned local proxy，和模型自己的 next-token prediction（NTP）信号脱节。本文提出 TTT-NTP：把 fast-weight 的 value target 换成**模型自己在下一位置的同层 contextual hidden state $h_{\ell,t+1}$**，让每次局部写入沿着模型做 NTP 的因果计算轨迹走。

---

Next-token prediction is the self-supervised signal that trains language models, and every observed prompt token provides the same signal at test time. We study whether this signal can define the inner-loop objective for test-time training (TTT) in pretrained long-context language models. Many TTT architectures require models to be trained with test-time adaptation in mind, limiting their direct applicability to released LLM checkpoints. While recent in-place TTT methods make fastweight adaptation possible for pretrained LLMs without redesigning the backbone, they leave a central question unresolved: what should each test-time write store? Existing recipes train the fast weight to match a learned local value proxy but they are not directly tied to the self-supervised next-token prediction signal. We introduce Test-Time Training with Next-Token Prediction (TTT-NTP), a drop-in fastweight adaptation method for pretrained LLMs that instead supervises updates using the model’s own next contextual hidden state. This makes each local write follow the same causal computation that supports next-token prediction: the value target is a pointwise linear projection of a single next-position contextual state. On RULER Full-13 (averaged over 4k, 8k, 16k, and 32k context lengths), TTT-NTP is the only method that consistently improves the released backbone across four models spanning three families and a 0.6–8B size range: Llama-3.1-8B (+3.9), Mistral-7B-v0.3 (+3.0), and the Qwen3 series (Qwen3-4B +4.1, Qwen3-0.6B +2.9). On the real-world LongBench-v2 long-document QA benchmark, TTT-NTP improves over the base model on both Llama-3.1-8B (+5.6) and Mistral-7B-v0.3 (+3.7), while preserving commonsense and knowledge performance.

> 💡 **问题动机**（Hao 批注）: 摘要的逻辑链要拆成三层才看得清楚。
> 1. **NTP 本来就是免费的 test-time 信号**：prompt 里每个 token $x_{t+1}$ 和它的前缀 $x_{1:t}$ 天然构成一对 (prefix, next-token)，可以拿来在推理时把模型 adapt 到当前文档/主题/检索问题上。这是「为什么用 NTP 做 TTT」。
> 2. **痛点是「写什么」而不是「怎么写」**：作者明确说 in-place TTT（Feng et al. 2026）已经把「write mechanism（rank-one）+ placement（MLP down-proj）」标准化了，唯一没解决的开放轴是 supervisory target。现有做法写的是一个 learned local value proxy（用邻域激活的小卷积网络拼出来的），它对训练有用但**不是模型做 NTP 时真正使用的表示**。
> 3. **本文的替换**：value target 直接用「观测到下一个 token 后、同一层产生的 contextual hidden state $h_{\ell,t+1}$」，再经过一个轻量线性投影 $W_{\ell}^{\text{proj}}$。因为这个 state 就在模型形成后续 next-token 分布的因果轨迹上，所以每次写入是在「把 MLP 输出往模型自己的预测路径上推」。

> 💡 **关键数字锚点**（Hao 批注）: 摘要抛出的核心 claim 是「唯一一个在四个 backbone 上都提升 RULER 的方法」。RULER Full-13（4k/8k/16k/32k 平均）：Llama-3.1-8B +3.9、Mistral-7B-v0.3 +3.0、Qwen3-4B +4.1、Qwen3-0.6B +2.9。LongBench-v2：Llama +5.6、Mistral +3.7。注意「唯一」这个词很重，意味着 CPT、In-Place TTT、qTTT 在某些 backbone 上会掉分——这正是 Table 1 要证明的证据链。

---

## 🔖 Section 总结

### 关键数字速查

| 指标 | 数值 |
|------|------|
| RULER Full-13 提升 (Llama-3.1-8B) | +3.9 |
| RULER Full-13 提升 (Mistral-7B-v0.3) | +3.0 |
| RULER Full-13 提升 (Qwen3-4B) | +4.1 |
| RULER Full-13 提升 (Qwen3-0.6B) | +2.9 |
| LongBench-v2 提升 (Llama-3.1-8B) | +5.6 |
| LongBench-v2 提升 (Mistral-7B-v0.3) | +3.7 |
| backbone 规模跨度 | 0.6B–8B，3 个 family |

### 核心洞察

1. **fast-weight 的「写什么」是被忽视的开放设计轴**：write mechanism（rank-one）和 placement（MLP down-proj）已标准化，target 才是决定下游表现的地方。
2. **model-native target 优于 learned proxy**：用模型自己在下一位置的 hidden state 做 target，比用邻域卷积拼出来的 local proxy 更贴合 NTP。
3. **提升长上下文而不牺牲通用能力**：commonsense/knowledge 分数基本不变，说明改的只是长上下文通路。

### 可追问点

- 为什么「下一位置的 hidden state」比 raw vocabulary target 更适合写进 MLP down-projection？→ 见 §2.3 与 §3.2。
- 训练时用 inner-product（Hebbian）write，推理时用 ridge 闭式解，两者为什么必须不一样？→ 见 §3.4 与消融 §4.5.2。
