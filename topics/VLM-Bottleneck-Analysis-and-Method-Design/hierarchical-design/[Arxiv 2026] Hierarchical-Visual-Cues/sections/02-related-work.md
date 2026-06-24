[← 返回 README](../README.md)

# 2. Related Work

## 一、Preview

Related Work 分为两条主线：(1) 潜空间思考（Latent Space Reasoning），从 Coconut 的"行为诱导"到 Huginn 的"结构递归"，建立两条技术路线的对比；(2) MLLM 演进，从 CLIP/BLIP 的对齐到 Qwen3-VL 的 DeepStack 层级注入，点出"现有 MLLM 仍基于 feed-forward + static visual encoding"的根本限制——这也是 HIVE 的 motivation。

---

## 二、原始文本

### 2.1. Thinking in Latent Space

Recently, many models that perform reasoning process in the latent space have emerged. (Zhu et al., 2025). Early work, such as Coconut (Hao et al., 2024), each language reasoning step in CoT is gradually substituted by hidden states from model. Subsequent work like SoftCoT (Xu et al., 2025c) generates "soft prompts" using a lightweight auxiliary model to serve as an initial CoT before formal reasoning, with frozen backbone LLM parameter, it prevents the catastrophic forgetting observed in Coconut and gain better results. While these approaches primarily focus on the language models, extending latent reasoning to the multimodal realm introduces unique challenges, such as the direct alignment of visual features with abstract logical steps. Consequently, recent research has begun to bridge this gap by integrating cross-modal inputs into the latent thinking process.

> 💡 **机制拆解 — "行为诱导型"潜空间推理**:
> - **Coconut**: 训练时将 CoT 的每一步逐步替换为模型自身的 hidden state（continuous thought），相当于"教会模型在脑子里想"。副作用：catastrophic forgetting。
> - **SoftCoT**: 用辅助模型生成 soft prompt 作为 CoT 的 warm start，冻结骨干以缓解遗忘。
> - **共同特点**: 不改架构，通过训练诱导模型将显式 CoT 内化为隐式表示。本质上是一种"行为模仿"，不是结构性推理。

Within the multimodal latent-space reasoning framework (Shen et al., 2025; Pham & Ngo, 2025), Heima adopts a training-driven approach akin to Coconut, where textual CoT is progressively compressed into specialized "thinking tokens." While Heima achieves latent reasoning through this behavioral adaptation, it does not alter the fundamental model architecture. In contrast, loop transformer recurrence (Dehghani et al., 2019; Mohtashami et al., 2025; Bae et al., 2025; Gao et al., 2025; Geiping et al., 2025b) introduces an explicit structural recurrence. This paradigm enables the iterative refinement of hidden states within a single forward pass by cycling through shared layers, effectively decoupling the depth of "thinking" from the physical parameter count. A representative implementation of this architectural philosophy is Huginn (Geiping et al., 2025b), which serves as the foundational backbone for our proposed framework.

> 💡 **核心对比 — 两条技术路线**:
>
> | 维度 | 行为诱导型 (Coconut/Heima) | 结构递归型 (Huginn) |
> |------|--------------------------|---------------------|
> | 核心思想 | 教会模型在隐空间"模仿"推理行为 | 让架构天然支持迭代推理 |
> | 实现方式 | 训练时将 CoT token 逐步替换为 continuous latent token | Loop transformer：同一组参数循环使用 |
> | 架构改动 | 无 | 需要特殊的 recurrent block 设计 |
> | CoT 依赖 | 高（需要大量 CoT 数据） | 低（推理能力来自架构本身） |
> | 推理控制 | 潜 token 数量固定 | 迭代数可动态调整 |
> | 关键优势 | 不改架构，即插即用 | 天然解耦"思考深度"与"参数量" |

> 💡 **Loop Transformer 的核心思想**: 将"思考"的深度与模型的参数数量解耦。一个 1.5B 的 recurrent block，通过 32 次迭代可以实现深度推理，而不需要 32×1.5B 的参数。这等价于用时间（compute）换空间（parameter）。

Specifically, Huginn is characterized by a tripartite loop transformer architecture consisting of three core components: the Embedding Blocks E, which project the input into the latent space; the Recurrent block R, which performs the iterative computations; and the Language Head H, which handles decoding and outputs probabilities. All three modules are built from fundamental decoder blocks.

Huginn utilizes a vocabulary of 65536 tokens via BPE (Sennrich et al., 2016). This 3.5B-parameter model was pretrained on 0.8T tokens without subsequent finetuning, The model comprises approximately 1.5B parameters in the nonrecurrent embedding blocks and language head, 1.5B parameters in the core recurrent block, and 0.5B parameters in the tied input embedding.

To optimize the training of such recursive structures, Huginn employs truncated backpropagation through depth. Unlike standard transformers, gradients are only propagated through the final k iterations of the recurrent unit, significantly reducing memory overhead while maintaining the stability of deep latent refinement (Mikolov et al., 2011).

> 💡 **Huginn 架构精读**:
> - **三元结构**: E（Embedding, 1.5B 参数）→ R（Recurrent Block, 1.5B 参数，循环 N 次）→ H（Language Head, 1.5B 参数）+ Tied Embedding（0.5B）
> - **关键数据**: 3.5B 参数，0.8T token 预训练，BPE 65536 词表
> - **训练 trick**: Truncated Backpropagation Through Depth——梯度只回传最后 k 次迭代。这是训练深层递归结构的关键稳定性技巧，避免梯度爆炸/消失
> - **重要提醒**: Huginn 的 0.8T pretraining 远小于同类 LLM（LLaMA2-7B 用 2T，Gemma3-4B 用 4T，Phi-3-mini 用 3.3T），这可能是 HIVE 性能上限的瓶颈

### 2.2. Multimodal Large Language Models

Early MLLMs focus on aligning visual and textual representations in a shared semantic space. CLIP (Radford et al., 2021) demonstrates the effectiveness of large-scale contrastive pretraining for zero-shot transfer, while BLIP (Li et al., 2022) and its variants (Li et al., 2023b; Liu et al., 2025) extend this paradigm toward generative multimodal modeling by connecting pretrained vision encoders with large language models. These works lay the foundation for subsequent MLLMs.

Building upon large language models, LLaVA (Liu et al., 2023c;a) introduces a projector-based connection scheme and instruction tuning to enable multimodal dialogue and reasoning. By directly mapping visual features into the language embedding space, LLaVA and its follow-ups achieve effective vision-language alignment with relatively low training complexity. However, due to the limited resolution of pretrained vision encoders, such approaches face challenges in tasks requiring fine-grained visual understanding.

> 💡 **MLLM 演进第一代 — Projector-based**: CLIP（对比学习对齐）→ BLIP（生成式扩展）→ LLaVA（Projector + Instruction Tuning）。共同特点：仅使用 ViT 最后一层特征，投影到 LLM embedding 空间后拼接。这种方案在细粒度视觉理解上天然受限——单层特征无法同时保留空间细节和语义信息。

Recent large-scale multimodal models emphasize unified training and improved visual feature utilization. LLaVA-OneVision-1.5 (An et al., 2025; Xie et al., 2025; Li et al., 2024) extends the LLaVA framework with a unified training pipeline to improve robustness across diverse visual tasks, while Qwen3-VL (Bai et al., 2023; Wang et al., 2024a; Bai et al., 2025c;a), introduces deepStack (Meng et al., 2024) to hierarchically inject fine-grained visual features into early layers of large language models, strengthening vision-language interaction without increasing input tokens. Nevertheless, these models remain based on feed-forward Transformer architectures with statically encoded visual representations, limiting explicit iterative or latent-space reasoning.

> 💡 **MLLM 演进第二代 — Hierarchical Feature Utilization**:
> - **LLaVA-OneVision**: 统一训练 pipeline，跨任务鲁棒性
> - **Qwen3-VL + DeepStack**: 将细粒度视觉特征分层注入 LLM 的早期层——这跟 HIVE 的层级注入有相似之处！但关键区别在于：
>   - DeepStack 是**静态注入**（前向时一次性注入不同层的 visual features 到 LLM 的不同层）
>   - HIVE 是**动态注入**（在 recurrent block 的不同迭代步中依次注入，且注入顺序受 recurrency depth 影响）
>   - 本质区别：注入发生在参数空间的不同位置 vs 注入发生在时间维度的不同步

> 💡 **关键论断 — 现有 MLLM 的根本限制**: "remain based on feed-forward Transformer architectures with statically encoded visual representations"。两层意思：(1) 前向架构→没有迭代 refine 的能力；(2) 静态视觉编码→视觉表示在推理过程中不变，无法动态适应推理需求。

Beyond architectural scaling, recent models highlight a shift in multimodal reasoning paradigms. The emergence of GPT-4o (OpenAI, 2024a;b) demonstrates the feasibility of fully unified multimodal models, supporting vision, language, audio, and video within a single framework. Moreover, recent studies (Yao et al., 2024; Xu et al., 2025a) suggest that enhanced reasoning performance increasingly relies on transitioning from fast, single-pass inference toward slower, deliberative reasoning processes, enabling multi-step refinement and improved decision making. This paradigm shift motivates the exploration of multimodal models that explicitly support iterative and structured reasoning mechanisms.

> 💡 **趋势总结**: 从"单次前向快推理"到"多步 deliberative 慢推理"的范式转换已是共识。GPT-4o 展示了全模态统一的可行性，但推理机制仍是前向的。HIVE 的定位：在全模态统一之外，开辟"结构化迭代推理"的维度。

---

## 三、Summary

- **Latent Space Reasoning 两条路线**:
  - 行为诱导型（Coconut/Heima）：训练时压缩 CoT → 不改架构，CoT 数据依赖高
  - 结构递归型（Huginn）：loop transformer 天然支持迭代 → 不依赖 CoT，推理深度可调
- **Huginn 关键参数**: 3.5B（E:1.5B + R:1.5B + H:1.5B，Embedding:0.5B），0.8T pretraining，truncated BPTT 训练
- **MLLM 演进**: Projector-based（LLaVA） → Hierarchical Feature（Qwen3-VL+DeepStack） → 但仍是 feed-forward + static visual encoding
- **根本 Gap**: 缺少"迭代推理 + 动态视觉利用"的结合——这正是 HIVE 的目标
