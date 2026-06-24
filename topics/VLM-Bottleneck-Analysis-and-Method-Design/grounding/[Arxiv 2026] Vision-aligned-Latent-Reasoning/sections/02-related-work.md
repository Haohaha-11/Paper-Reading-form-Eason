[← 返回 README](../README.md)

# 2. Related Works

## 一、Preview

本文从三条相互关联的研究线索展开 Related Work：(1) MLLM 架构演进，从 resampler/Q-Former 到轻量投影器；(2) CoT 推理的发展，从显式 prompt 到潜空间推理，再到"思考时看图"；(3) MLLM 中利用外部视觉编码器的最新趋势。三条线索的交汇点正是 VaLR 的定位：**利用外部视觉编码器来监督潜空间的视觉推理**。

---

## 二、原始文本

### Multi-modal Large Language Models (MLLMs)

Recent advancements in MLLMs harness the inherent reasoing proficiency of LLMs to establish unified architectures designed to handle multiple modalities within a single framework. Pioneering studies integrate visual information into LLMs, predominantly utilizing either resamplers (Alayrac et al., 2022; Awadalla et al., 2023; Li et al., 2025d; Cha et al., 2024) or Q-Former (Li et al., 2023; Dai et al., 2023; Zhu et al., 2023; Lin et al., 2024). Despite the effectiveness of these specialized architectures, LLaVA (Liu et al., 2023; 2024a) and its successors (Chen et al., 2024a; Liu et al., 2024b; Chu et al., 2023; 2024; Bai et al., 2023a;b; Yang et al., 2024; 2025a) demonstrate that aligning each modality through a trainable lightweight projector is sufficient when paired with visual instruction tuning. Nevertheless, these models suffer from solving problems that require comprehensive reasoing, falling short of the reasoing capabilities exhibited by Chain-of-Thought (CoT).

> 💡 **MLLM 架构演化路线**: Resampler/Q-Former → Lightweight Projector (LLaVA-line)。作者指出即使是最先进的 MLLM 架构，在需要综合推理的问题上仍然不足。这个陈述为后文将 CoT 和潜推理引入 MLLM 做了铺垫。

### Chain-of-Thought (CoT) and Latent Reasoing

The emergence of Chain-of-Thought (CoT) prompting has significantly enhanced the reasoing capabilities of large language models (LLMs) by decomposing complex problems into intermediate linguistic steps. While early works (Wei et al., 2022; Khot et al., 2022; Zhou et al., 2022) primarily relied on explicit prompting to derive these chains, subsequent research has focused on intrinsic enhancement through supervised fine-tuning (Yue et al., 2023; Yu et al., 2023) or reinforcement learning (Wang et al., 2024b; Havrilla et al., 2024; Shao et al., 2024b; Yu et al., 2024). To expand the search space of reasoing chains during inference, extensive studies have introduced tree-based (Xie et al., 2023; Yao et al., 2023; Hao et al., 2024a) and trajectory-based (Lehnert et al., 2024; Gandhi et al., 2024; Su et al., 2024) exploring algorithms. Motivated by the insight that natural language cannot encapsulate all forms of reasoing, recent paradigms have shifted toward operating directly within latent space (Hao et al., 2024b; Wang et al., 2025c; Li et al., 2025b) or learning to generate visual information from latent reasoing features (He et al., 2024; Li et al., 2025e). Orthogonally, several approaches (Zheng et al., 2025b; Yang et al., 2025d) propose to interleave visual tokens in reasoing trajectories to empower multi-modal reasoing. In this work, we align the latent reasoing tokens with features from vision encoders to facilitate visual reasoing within the latent space.

> 💡 **CoT 推理的发展谱系**:
>
> | 阶段 | 方法类型 | 代表工作 | 核心思想 |
> |------|---------|---------|---------|
> | 1. Prompt-based CoT | Few-shot CoT prompting | Wei et al., 2022 | 用提示词激发 LLM 的逐步推理 |
> | 2. Training-based CoT | SFT / RL | Math-Shepherd, DeepSeekMath | 通过训练内化推理能力 |
> | 3. Search-based CoT | Tree/Trajectory search | ToT, Stream of Search | 扩大推理链搜索空间 |
> | 4. Latent Reasoing | 潜空间推理 | COCONUT, Monet, LVR | 在连续潜空间中推理，不依赖自然语言 |
> | 5. Visual CoT | 交错视觉 token | DeepEyes, MVoT, MMI | 推理过程中插入视觉信息 |

> 💡 **关键引用 — "natural language cannot encapsulate all forms of reasoing"**: 这个论断是潜推理研究的哲学基础。作者认为自然语言作为一种离散符号系统，无法完全捕捉所有推理形式（尤其是视觉推理），因此需要在连续潜空间中进行推理。VaLR 在此基础上更进一步：**不仅要在潜空间中推理，还要让潜空间中的推理与视觉表征对齐**。

### Leveraging External Vision Encoders in MLLMs

Recent MLLMs incorporate rich visual features, e.g., CLIP (Radford et al., 2021), DINO (Oquab et al., 2023; Simeoni et al., 2025), SigLIP (Zhai et al., 2023), and VGGT (Wang et al., 2025b), to enhance their visual and spatial reasoing capabilities. For instance, PrismaticVLM (Karamcheti et al., 2024) integrates CLIP and DINO features through trainable projection layers to leverage rich visual representations. Similarly, PaliGemma (Beyer et al., 2024) exploits the dense features of SigLIP to enable comprehensive visual understanding with fewer parameters. To enhance the spatial awareness of MLLMs, several studies (Zheng et al., 2025a; Wu et al., 2025; Huang et al., 2025) leverage VGGT to inject token-wise spatial information. Concurrent with our work, CoVT (Qin et al., 2025a) and Monet (Wang et al., 2025c) enhance the visual understanding of MLLMs by leveraging rich visual features to perform reasoing directly within the visual space. However, as the reasoing chain lengthens, they suffer from diminishing visual signals since the visual information is only utilized as a fixed initial context. To alleviate this attenuation, our VaLR aligns latent tokens at each reasoing step with vision encoders, thereby enabling long-context visual reasoing.

> 💡 **现有方法的关键局限**: 这段话是 Related Work 部分最重要的批评。PrismaticVLM、PaliGemma、CoVT、Monet 等方法虽然利用了强大的外部视觉编码器，但都只将视觉特征作为**固定的初始上下文**。随着推理链增长，这些视觉信号的相对影响力自然衰减。VaLR 的创新在于**在每个推理步都重新对齐视觉特征**，实现动态的视觉信息保持。

> 💡 **VaLR vs. CoVT/Monet 的定位差异**:
> - CoVT/Monet: 在视觉空间中进行推理（"thinking in visual space"），但视觉特征仅作为固定初始化
> - VaLR: 在潜空间中进行推理，但**每步**都用视觉特征进行对齐监督（"thinking with visual alignment"）
> - 关键差异：前者是"一旦注入，永远使用"；后者是"每步检查，持续对齐"

---

## 三、Summary

- **MLLM 架构**: 从复杂的 resampler/Q-Former 收敛到简单的 light projector，但推理能力仍不足。
- **CoT 发展**: 从 Prompt → Training → Search → Latent → Visual CoT，趋势是向连续潜空间和多模态融合演进。
- **外部视觉编码器**: PrismaticVLM、PaliGemma 等证明了利用外部编码器的价值，但都是静态一次性注入。
- **VaLR 的定位**: 结合潜空间推理和外部视觉编码器，但实现**每步动态对齐**——这是与所有 prior work 的关键区别。
