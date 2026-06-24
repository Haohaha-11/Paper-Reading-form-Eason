[← 返回 README](../README.md)

# 3. Method

## 一、Preview

Method 分为四个子模块：(3.1) 整体架构概述——双分支设计；(3.2) 指令引导的视觉特征调制——CLIP text encoder 提取 [CLS] 嵌入后通过 AdaLN 注入 ViT 各层；(3.3) 双分支特征融合——Zero-FFN 实现渐进式融合；(3.4) MM4 benchmark 的设计原则。

---

## 二、原始文本

In this section, we introduce iGVLM, a decoupled instruction-guided vision encoder designed to condition the utilization of visual features on linguistic instructions while preserving pretrained visual representations. We first present an overview of the overall architecture, followed by detailed descriptions of (i) instruction-guided visual feature modulation and (ii) dual-branch feature fusion. Finally, we introduce MM4, a controlled diagnostic benchmark for evaluating question-aware visual perception in Vision–Language Models (VLMs).

### 3.1. Overall Architecture

An overview of the proposed framework is illustrated in Figure 2(a). Given an image–text pair, iGVLM conditions visual feature generation on the textual instruction through a dedicated conditioning pathway, while preserving the original perceptual capacity of the pretrained vision backbone. Specifically, the textual instruction is first encoded into a compact semantic representation, which serves as a global guidance signal for visual modulation. This instruction embedding conditions a pretrained vision encoder, enabling visual features to be selectively modulated according to task-specific linguistic cues.

To explicitly separate representation preservation from instruction-conditioned adaptation, iGVLM adopts a dualbranch architecture. A static branch retains a frozen vision encoder to preserve task-agnostic visual priors, while a dynamic branch generates instruction-adapted visual features through lightweight modulation modules. The outputs of these two branches are fused to obtain a balanced visual representation that combines general-purpose perceptual semantics with task-specific adaptation. The fused visual features are subsequently projected into the language embedding space and provided, together with the instruction tokens, to a large language model (LLM) for multimodal reasoning and response generation.

> 💡 **架构核心 — Figure 2(a) 数据流解析**:
> ```
> Image ──┬── Frozen ViT (Static Branch) ──→ y₀ ──┐
>          │                                         ├──→ $y_{I}$ = Z(Norm($y_{ct}$)) + y₀ ──→ LLM
>          └── AdaLN-ViT (Dynamic Branch) ──→ $y_{ct}$ ─┘         ↑
>                                    ↑                      Zero-FFN
>                          [CLS] embedding $c_{t}$
>                                    ↑
>              Text Instruction → CLIP Text Encoder → Linear Proj → ĉ_t
> ```
> 关键设计：文本指令通过 CLIP Text Encoder → [CLS] token → 线性投影 → 注入 AdaLN 调制 ViT 各层的 attention 和 MLP 模块。

### 3.2. Instruction-Guided Visual Feature Modulation

To enable instruction-conditioned visual perception, we derive a global textual guidance signal from the instruction. We adopt the text encoder from a pretrained CLIP model and truncate the input text to a maximum length of 77 tokens. The resulting [CLS] token embedding summarizes the semantic intent of the instruction and is mapped into the vision latent space through a lightweight linear projection:

$$

c _ { t } = \mathscr { F } _ { T } ( T _ { \leq 7 7 } ) , \quad \hat { c } _ { t } = \mathscr { H } _ { t } ( \mathrm { N o r m } ( c _ { t } ) ) ,

$$

where $\mathcal { F } _ { T } ( \cdot )$ denotes the CLIP text encoder, and $\mathcal { H } _ { t } ( \cdot )$ aligns the text embedding with the vision feature space.

We incorporate Adaptive Layer Normalization (AdaLN) (Perez et al., 2018) into each transformer block of the CLIP vision encoder to inject textual conditioning in a stable and localized manner. The projected instruction embedding $\hat { c } _ { t }$ is transformed into layer-wise modulation parameters that control feature scaling and shifting within both the self-attention and feedforward submodules. By integrating AdaLN across all transformer layers, iGVLM enables hierarchical instruction-conditioned modulation while preserving the pretrained weights of the vision backbone.

> 💡 **为什么选择 AdaLN 而不是 Cross-Attention？**:
> - **AdaLN**: 通过 scale & shift 参数对特征做仿射变换，不改变 token 序列长度，不引入额外的注意力计算。是一种**轻量级的、逐层的**条件化方式。
> - **Cross-Attention**: 需要让视觉 token 去 attend 文本 token，引入 O(N²) 的额外计算。
> - **Table 5 验证**: iGVLM-Cross (cross-attention 版本) 性能低于原始 iGVLM，且推理开销更大。
> - AdaLN 的关键优势：**hierarchical modulation**——每一层都有独立的 scale/shift 参数，使得浅层调制低级纹理，深层调制语义概念。

Formally, given an input image I and instruction embedding $\hat { c } _ { t } ,$ the instruction-guided vision encoder produces:

$$

y _ { c t } = \mathscr { F } _ { c t } ( I , \hat { c } _ { t } ; \Theta _ { \mathrm { C L I P } } ) ,

$$

where $y _ { c t } \in \mathbb { R } ^ { N _ { I } \times D _ { I } }$ denotes the instruction-conditioned visual features and $\Theta _ { \mathrm { C L I P } }$ represents the frozen pretrained parameters.

> 💡 **冻结参数 + AdaLN 的可训参数**: CLIP ViT 的原始参数完全冻结，只有 (1) CLIP text encoder 到 vision space 的线性投影 (2) 每层的 AdaLN scale/shift 参数是可训练的。参数量增加极少（Table 4：13.35B → 13.78B），只有约 430M 新增参数，且主要是 AdaLN 的仿射参数。

### 3.3. Dual-Branch Feature Fusion

While instruction-conditioned modulation enables taskspecific adaptation, preserving the original perceptual semantics is essential for stable and generalizable visual understanding. To this end, iGVLM employs a dual-branch fusion mechanism that explicitly combines instruction-guided features with the original frozen visual representations.

Let $y _ { c t } \in \mathbb { R } ^ { N _ { I } \times D _ { I } }$ denote the instruction-guided features from $\mathcal { F } _ { c t }$ , and let $y _ { 0 } = \mathcal { F } _ { I } ( I ; \Theta _ { \mathrm { C L I P } } )$ denote the corresponding frozen features from the original vision encoder. The fused visual representation is computed as:

$$

y _ { I } = \mathcal { Z } ( \mathrm { N o r m } ( y _ { c t } ) ) + y _ { 0 } ,

$$

where Z is a learnable linear projection initialized to zero. This initialization ensures that the fused representation initially matches the pretrained visual features, allowing instruction-conditioned adaptation to be introduced gradually and safely during training.

Following the LLaVA-1.5 framework, the fused visual features $y _ { I }$ are projected into the input embedding space of the LLM via a learnable linear transformation. Training proceeds in two stages: first, the instruction-guided vision encoder and projection layers are optimized while keeping the pretrained vision backbone and LLM frozen; second, all components are jointly optimized to enable coherent multimodal reasoning.

> 💡 **Zero-FFN 的关键设计意图 — 渐进式安全初始化**:
> - 公式 (3) 中的 $\mathcal{Z}$ 被零初始化。这意味着在训练开始时，$y_I = 0 + y_0 = y_0$，模型的输出与原始 LLaVA-1.5 **完全一致**。
> - 训练过程中，$\mathcal{Z}$ 的权值逐渐偏离零，指令调制的特征逐步"混入"静态特征。
> - 这种初始化策略保证了：(1) 不破坏预训练先验 (2) 训练稳定 (3) 如果指令调制无效，模型自然退化为 baseline。这是一种**优雅的防御性设计**。

> 💡 **两阶段训练策略**:
> - **Stage 1 (Alignment Pretraining)**: 只训练 AdaLN 参数 + 投影层，视觉 backbone 和 LLM 均冻结。学习率 6e-4（比 LLaVA-1.5 的 1e-3 更低）。
> - **Stage 2 (Instruction Tuning)**: 所有组件联合优化。学习率 2e-5。
> - 这种渐进式训练确保了 AdaLN 调制能力先收敛，再与 LLM 协同精调。

### 3.4. MM4: A Diagnostic Benchmark for Question-Aware Visual Perception

To complement existing multimodal benchmarks such as MMStar (Chen et al., 2024b), which primarily assess general-purpose multimodal reasoning, we introduce MM4, a controlled diagnostic benchmark designed to evaluate question-aware and multi-query visual perception. MM4 consists of 180 images and 720 manually verified question–answer pairs, with annotations curated by domain experts to ensure quality and consistency.

Each image in MM4 is associated with four semantically distinct questions, constructed according to three design principles: (i) robustness through answer reversal, (ii) multiperspective semantic diversity, and (iii) balanced answer distribution. This design enables MM4 to jointly assess intra-image consistency and inter-question diversity. To evaluate multi-query reasoning, MM4 adopts a hierarchical scoring protocol that credits a model only if it correctly answers at least n of the four questions per image, encouraging consistent instruction-aware reasoning rather than isolated accuracy.

> 💡 **MM4 三大设计原则**:
>
> | 原则 | 含义 | 为什么重要 |
> |------|------|-----------|
> | Answer Reversal | 用同一问题的选项重排版本来测试鲁棒性（如 Q1: A:Red B:Blue / Q2: A:Blue B:Red） | 排除模型对选项位置的 bias |
> | Multi-Perspective Semantic Diversity | 四问必须从不同语义角度提问（CP/FP/IR/LR/ST/MA） | 强制模型在不同维度上切换视觉关注 |
> | Balanced Answer Distribution | 正确答案在 A/B/C/D 间均匀分布 | 排除猜测策略 |

> 💡 **Hierarchical Scoring 的设计智慧**:
> - n=1: 模型答对至少 1/4 即得分 → 测"基本能力"
> - n=2: 答对至少 2/4 → 测"中等一致性"
> - n=3: 答对至少 3/4 → 测"高一致性"
> - n=4: 全部答对 → 测"完美一致性"
> - 随机猜测：n=1 期望约 68%，n=4 期望约 0.4%。所以 n=4 是最严格的指标，区分度最大。
> - 从 Table 2 可以看出，随着 n 增大，所有模型得分都在下降，但 **iGVLM 的下降速度更慢**（尤其是 n=3 和 n=4），这正是 instruction-conditioned perception 在起作用。

---

## 三、Summary

| 模块 | 核心机制 | 关键设计要点 |
|------|---------|------------|
| **3.1 整体架构** | 双分支：Static (frozen ViT) + Dynamic (AdaLN-ViT) | 解耦 = 分离表征保留与指令调制 |
| **3.2 调制** | CLIP text enc [CLS] → Linear Proj → AdaLN → ViT 每层 | AdaLN 替代 Cross-Attn，轻量+逐层+hierarchical |
| **3.3 融合** | $y_{I}$ = Zero-FFN(Norm($y_{ct}$)) + y_0 | Zero init → 训练初 = baseline，渐近混入调制特征 |
| **3.4 MM4** | 180图 x 4问 = 720QA, n-out-of-4 scoring | 三大原则：Answer Reversal + 多视角 + 选项均衡 |
