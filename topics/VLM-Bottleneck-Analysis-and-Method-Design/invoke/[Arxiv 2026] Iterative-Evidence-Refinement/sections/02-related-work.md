[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

相关工作分两条主线：(1) 工具增强的多模态推理——显式调用 crop/zoom/detection 等工具回访视觉信息；(2) 潜空间多模态推理——在表示空间而非像素空间中操作，避免显式图像操作。SIEVE 位于两者的交叉点：它不调用外部工具，但也不构建新的潜空间，而是直接在**文本推理空间**中复用原始视觉 embedding。

---

## 二、原始文本

### 2.1 Tool-augmented Multi-modal Reasoning

Recent work extends VLM inference beyond a single visual pass by introducing explicit visual re-query mechanisms during reasoning. A prominent line equips models with visual tools (e.g., cropping, zooming, object detection) that produce targeted observations and feed them back as additional inputs Huang et al. [2025b], Su et al. [2025a], Fan et al. [2025], Chen et al. [2025], Hu et al.. Later approaches learn tool-use policies with reinforcement learning, rewarding effective re-query trajectories and verification behaviors, often adopting coarse-to-fine strategies that start from a global view and refine under higher-resolution observations Zheng et al. [2025], Zhong et al. [2025]. Related methods further encourage pixel-space exploration, either via instruction tuning to broaden search coverage Su et al. [2025b] or RL to strengthen perceptual competence and steer attention to task-relevant regions Yu et al. [2025a]. Despite strong performance, these systems typically incur inference-time overhead due to repeated view generation and re-encoding.

> 💡 **工具增强推理谱系**: 这条线经历了三个阶段的发展：
> 1. **Heuristic 工具调用** (Huang, Su, Fan, Chen, Hu): 预定义工具集 (crop/zoom/detection)，模型通过 prompt 或轻量训练学会调用。问题是工具选择缺乏灵活性。
> 2. **RL-based 策略学习** (DeepEyes, Omni-R1): 用 RL 训练模型学习"何时调用哪个工具"，引入 coarse-to-fine 策略。问题是 RL 训练本身成本高，且 action space 包括工具选择和参数确定。
> 3. **像素空间探索** (Pixel Reasoner, Perception-R1): 不再局限于预定义工具，鼓励模型在像素空间自由探索。问题是计算开销大，且探索效率低。
> - **SIEVE 在这一谱系中的位置**: SIEVE 本质上是把"工具调用"替换为"embedding 检索"——action space 缩小到二元决策（插入 evidence or not），这大幅降低了训练复杂度。但代价是灵活性下降（无法像工具调用那样动态生成任意区域的新 view）。

> 💡 **共同痛点 — 推理时开销**: 所有工具增强方法的共同瓶颈是 "repeated view generation and re-encoding"。每次调用 crop/zoom 都需要 (1) 根据坐标裁剪原图；(2) 重新过 vision encoder 生成新的 visual token。这两个操作在推理时是纯 overhead，与生成过程串行。SIEVE 不需要这些步骤，因为它直接复用第一次编码的结果。

### 2.2 Multi-modal Reasoning in Latent Space

In parallel, several lines reduce reliance on explicit image operations by operating in representation space or by selectively using visual tokens. Generation-based methods synthesize auxiliary visual representations to support inference Xu et al. [2025], Chern et al. [2025], Li et al. [2025b]. More recently, latent thinking-with-images paradigms introduce learnable latent visual tokens and embedding-level manipulation to internalize certain visual operations and enable mode switching during inference Li et al. [2025c], Zhang et al. [2025a], Yang et al. [2025b]. A complementary efficiency literature observes that dense patch sequences are redundant and studies how to select, prune, or merge visual tokens while preserving representational fidelity Chen et al. [2024], Cao et al. [2024], Wang et al. [2025], Bolya et al. [2022], Zeng et al. [2025], Zhang et al. [2024a], Huang et al. [2024], Li et al. [2025d], Hu et al. [2025], Yu et al. [2025b], Song et al. [2024]. These approaches share a common premise: they construct reasoning processes within a latent visual space and train models to perform reasoning in that space. However, this paradigm requires substantial effort to enable the model to learn reasoning over newly introduced visual latents that differ from the native textual representation space. Motivated by this limitation, we propose a framework that directly leverages the embeddings of images within the textual reasoning space, rather than constructing and training reasoning in a separate latent space.

> 💡 **潜空间推理的三条支线 — SIEVE 属于哪一条？**: 
>
> | 支线 | 代表工作 | 核心操作 | 代价 |
> |------|---------|---------|------|
> | 图像生成辅助推理 | Visual Planning, Thinking with Generated Images, MVoT | 在推理过程中**生成**辅助图像/草图 | 生成模型开销大，生成质量不稳定 |
> | 可学习潜视觉令牌 | Latent Visual Reasoning, DeepSketcher, Machine Mental Imagery | 引入**新的可学习 token**，在潜空间操作视觉表示 | 需要大量训练对齐，潜令牌与文本空间不天然兼容 |
> | 视觉 token 选择性使用 (效率线) | Token Merging, IVTP, LESS, ReGaTE | 在已有 token 中**选择/剪枝/合并** | 目标偏效率而非推理能力增强 |
>
> - **SIEVE 的独特定位**: SIEVE 不属于上述任何一条。它既不生成新图像，也不引入新潜令牌，也不以效率为首要目标。它的操作对象是**原始视觉编码的 embedding**——从已经存在的信息中检索和复用。这是一个被之前工作忽视的方向：大部分潜空间推理工作倾向于"造新的"，而不是"翻已有的"。
>
> - **与 Token Pruning 的关键区别**: Token pruning 的目标是**减少 token 数量**以提高效率——删除"冗余" patch。SIEVE 的目标是**增加关键 token** 以提高推理质量——在推理链中插入"关键" region。方向相反，但底层技术（patch 重要性评估）可以互通。

---

## 三、Summary

- **工具增强线**: crop/zoom/detection + RL 策略学习 → 强但推理开销大，训练数据需求大
- **潜空间推理线**: 图像生成辅助 / 可学习潜令牌 / token 效率 → 各有代价（生成不稳定 / 训练成本高 / 仅关注效率）
- **SIEVE 的空白填补**: 复用已有 embedding 而非创造新的——从 "generation" 转向 "retrieval"，从 "外部工具" 转向 "内部信号"
