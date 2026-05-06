[← 返回 README](../README.md)

# Abstract

## 📌 预览
这一节给出 DMLR 的核心定位：不是训练一个会显式写长 CoT 的多模态模型，而是在测试时优化少量 latent think tokens，并在不确定时动态注入视觉 patch。

---

# Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space

Chengzhi Liu\*1 Yuzhe Yang\* Yue Fan3 Qingyue Wei2 Sheng Liu†2 Xin Eric Wang†1 1 University of California, Santa Barbara 2 Stanford University 3 University of California, Santa Cruz \* Equal contribution † Equal advising {chengzhi,yuzheyang,ericxwang}@ucsb.edu, shengl@stanford.edu

> 💡 **论文入口批读**: 作者组合横跨 UCSB、Stanford、UCSC，且 Sheng Liu / Xin Eric Wang 共同 advising。题目里的 “Within the Mind” 对应方法上不显式生成中间图像或长文本，而是在模型内部 latent space 做推理更新。

Abstract: Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced cross-modal understanding and reasoning by incorporating Chain-of-Thought (CoT) reasoning in the semantic space. Building upon this, recent studies extend the CoT mechanism to the visual modality, enabling models to integrate visual information during reasoning through external tools or explicit image generation. However, these methods remain dependent on explicit step-by-step reasoning, unstable perception–reasoning interaction and notable computational overhead. Inspired by human cognition, we posit that thinking unfolds not linearly but through the dynamic interleaving of reasoning and perception within the mind. Motivated by this perspective, we propose DMLR, a test-time Dynamic Multimodal Latent Reasoning framework that employs confidence-guided latent policy gradient optimization to refine latent think tokens for in-depth reasoning. Furthermore, a Dynamic Visual Injection Strategy is introduced, which retrieves the most relevant visual features at each latent think token and updates the set of best visual patches. The updated patches are then injected into latent think token to achieve dynamic visual–textual interleaving. Experiments across seven multimodal reasoning benchmarks and various model architectures demonstrate that DMLR significantly improves reasoning and perception performance while maintaining high inference efficiency.

> 💡 **摘要批读**: 摘要把问题拆成三类痛点：显式逐步推理、感知-推理交互不稳定、计算开销高。DMLR 的回答是 test-time latent optimization：输入和模型参数不变，只把 latent think tokens 当作可优化变量；视觉信息不是一次性全塞进去，而是随每个 latent step 动态选 patch。

§ Project Page: https://mllm-dmlr.github.io/

> 💡 **项目入口**: Project page 说明这篇论文有外部展示入口。批读时仍以 arXiv v3 和 MinerU full.md 为准，项目页适合后续查代码、demo 或视觉案例是否开放。

![Figure 1](../images/b28aaa4fa9315c2fc06257bda44c48889b89cbda406c4160b27750da7544168c.jpg)
*Figure 1: Comparison between DMLR and two reasoning paradigms. (A) Text-only reasoning: relies solely on explicit CoT, often causing visual grounding errors and redundant steps. (B) Think-with-Image reasoning: depends on external perception tools, leading to unstable tool calls and extra overhead. (C) DMLR (ours): refines latent think tokens in the latent space through confidence-guided optimization and dynamically injects visual information, achieving self-improving reasoning without additional training while maintaining high efficiency.*

> 💡 **Figure 1 批读**: 图 1 是全篇路线图。A 的问题是只有语言 CoT，容易“多想少看”；B 的问题是借外部视觉工具，调用不稳且慢；C 的关键是把视觉检索和 reasoning update 都压到 latent space，用 confidence 作为内部反馈信号。

---

## 🔖 Section 总结

### 关键数字速查
| 指标 | 数值 |
|------|------|
| arXiv | 2512.12623 v3 |
| 最新版本日期 | 2026-04-09 |
| 方法类型 | test-time, training-free latent optimization |
| 评测范围 | 7 个 multimodal reasoning benchmark |

### 核心洞察
1. DMLR 的 novelty 不在新 backbone，而在测试时把 latent token 当作可优化状态。
2. 视觉 patch 的注入由 confidence/reward 驱动，避免固定位置或全量视觉注入。
3. 摘要的 claim 是“提升 reasoning 和 perception，同时保持效率”，后文实验需要分别验证这三点。
