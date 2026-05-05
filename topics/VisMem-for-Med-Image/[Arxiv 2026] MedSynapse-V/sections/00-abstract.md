[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文件保留论文题名、作者、摘要等开场信息，先抓住作者定义的问题、方法名和主要 claim。

---

# MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution

Chunzheng Zhu $^ { 1 }$ , Jiaqi Zeng1, Junyu Jiang1, Jianxin Lin $\bot$ ⋆, and Yijun Wang1

Hunan University, Changsha, China {zhuchzh, zjqxxl, jiangjy, linjianxin, wyjun}@hnu.edu.cn

Abstract. High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose MedSynapse-V, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model’s hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR) which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privilegedautonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that MedSynapse-V, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-theart methods, particularly Chain-of-Thought (CoT) paradigms, in diagnostic accuracy and multi-dataset generalization without compromising the inference efficiency of standard VLMs.

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

Keywords: VLMs · Implicit Diagnostic Memory $\cdot$ Latent Space Memory · Causal Counterfactual · Memory Distillation

> 💡 **批注**: 这里的核心是 latent-space 计算：作者希望在连续表示中完成推理/记忆，而不是完全依赖显式文本链。

---

## 🔖 Section 总结

### 核心洞察
1. 先记住本文的问题定义、方法名和主要 claim。
2. 后续阅读时回看摘要里的 claim 是否被实验支持。
