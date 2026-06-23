# 1 Introduction

[← 返回 README](../README.md)

---

The integration of vision and language into Multimodal Foundation Models (MFMs) promises a future where AI agents can perceive and reason about the physical world. However, this promise is threatened by hallucination, the tendency of models to generate confident but factually incorrect assertions. To deploy these models in safety-critical domains (e.g., robotics, medical imaging), we must be able to quantify their reliability.

Traditionally, interpretability research has looked to the "Attention Mechanism" as a window into the model's mind Jain & Wallace (2019). In Vision-Language Models (VLMs), this manifests as the Attention-Confidence Assumption: the belief that a model's reliability is correlated with the quality of its visual grounding. If a model is asked, "Is there a dog?" and it focuses sharply on the dog, we assume that it "knows" the answer. If its attention is diffuse or focuses on the background, we assume that it is hallucinating.

> 💡 **问题动机 — Attention-Confidence Assumption的起源与局限**: 这个假设的根源可以追溯到NLP可解释性研究中对注意力机制的"过度解读"。Jain & Wallace (2019) 在纯文本领域就提出了"Attention is not Explanation"，而本文将其扩展到多模态场景。核心问题在于：注意力机制本质上是"信息检索"而非"信息判断"——它告诉你模型从哪里提取了特征，但不告诉你模型是否正确地理解了这些特征。后续的"See but not believe" (Liu et al., 2025) 工作也证实了这一点。

In this work, we rigorously test this assumption across three representative VLM families (LLaVA-1.5, PaliGemma, and Qwen2-VL) Liu et al. (2023); Beyer et al. (2024); Wang et al. (2024). We perform a comprehensive analysis of reliability signals by comparing "structural" metrics derived from visual cross-attention against "linguistic" metrics derived from generation dynamics. We explicitly position novelty at the hidden-state reliability probe and cross-family layer-wise analysis; attention-failure and self-consistency are treated as important prior findings that we extend and calibrate in the VLM setting.

> 💡 **机制拆解 — 三种架构的选择逻辑**: 
> - LLaVA-1.5: 经典的prefix-based架构，视觉token作为前缀拼接在文本token之前，视觉编码器(CLIP ViT-L)冻结，仅通过2层MLP投影到LLM空间。这是最广泛使用的架构范式。
> - PaliGemma: 早期融合架构，使用SigLIP编码器+Gemma语言骨干，视觉信息在浅层即与语言token交互。
> - Qwen2-VL: 原生多模态架构，视觉token与文本token交错排列，支持动态分辨率。代表了最新一代的设计思路。
> 这三种架构覆盖了从"后期注入视觉"到"早期融合视觉"的完整谱系，使得跨架构结论更具说服力。

> 💡 **机制拆解 — 创新边界自我定位**: 作者非常诚实地指出：他们不声称首次发现"注意力不忠"或"self-consistency有用"——这些在NLP/VLM文献中已有先例。本文的创新在于：(1) 统一跨架构的可靠性研究；(2) 将"Early Locking/Symbolic Detachment"动力学与下游正确性联系起来；(3) 证明隐藏状态探针提供最强的单次推理可靠性信号。

Terminology note. We use VLM as the default term throughout; MFM and LVLM are used only when matching prior-work phrasing.

Reproducibility. Code and evaluation scripts are available at https://github.com/itsloganmann/VLM-Reliability-Probe (prompts, split definitions, and probe training pipeline).

---

**📌 Preview:** 引言部分建立了全文的核心研究问题——Attention-Confidence Assumption是否成立——并概述了三阶段研究策略。实验部分将通过结构指标、机制探针和行为指标三个层次进行验证。

**🔖 Summary:** 本文研究的核心张力在于：VLM社区普遍默认"好的注意力=好的答案"，但这一假设从未在VLM场景下被系统性地跨架构检验。作者选择三族代表性架构(prefix-based、early-fusion、native-multimodal)，设计从相关性到因果性再到机制的递进式分析，最终揭示出"Symbolic Detachment"这一理论发现。
