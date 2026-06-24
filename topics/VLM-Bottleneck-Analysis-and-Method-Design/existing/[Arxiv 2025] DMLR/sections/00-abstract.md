[← 返回 README](../README.md)

# Abstract & Figure 1

## 一、论文信息速览

| 项目 | 内容 |
|------|------|
| **标题** | Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space |
| **作者** | Chengzhi Liu\*1, Yuzhe Yang\*, Yue Fan3, Qingyue Wei2, Sheng Liu†2, Xin Eric Wang†1 |
| **单位** | 1 UC Santa Barbara, 2 Stanford University, 3 UC Santa Cruz |
| **发表** | arXiv 2025 (2512.12623) |
| **项目页** | https://mllm-dmlr.github.io/ |

---

## 二、原始文本

Reasoning Within the Mind: Dynamic Multimodal Interleaving in Latent Space

Chengzhi Liu\*1 Yuzhe Yang\* Yue Fan3 Qingyue Wei2 Sheng Liu†2 Xin Eric Wang†1 1 University of California, Santa Barbara 2 Stanford University 3 University of California, Santa Cruz \* Equal contribution † Equal advising {chengzhi,yuzheyang,ericxwang}@ucsb.edu, shengl@stanford.edu

Abstract: Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced cross-modal understanding and reasoning by incorporating Chain-of-Thought (CoT) reasoning in the semantic space. Building upon this, recent studies extend the CoT mechanism to the visual modality, enabling models to integrate visual information during reasoning through external tools or explicit image generation. However, these methods remain dependent on explicit step-by-step reasoning, unstable perception--reasoning interaction and notable computational overhead. Inspired by human cognition, we posit that thinking unfolds not linearly but through the dynamic interleaving of reasoning and perception within the mind. Motivated by this perspective, we propose DMLR, a test-time Dynamic Multimodal Latent Reasoning framework that employs confidence-guided latent policy gradient optimization to refine latent think tokens for in-depth reasoning. Furthermore, a Dynamic Visual Injection Strategy is introduced, which retrieves the most relevant visual features at each latent think token and updates the set of best visual patches. The updated patches are then injected into latent think token to achieve dynamic visual--textual interleaving. Experiments across seven multimodal reasoning benchmarks and various model architectures demonstrate that DMLR significantly improves reasoning and perception performance while maintaining high inference efficiency.

> 💡 **一句话概括**: DMLR 提出了一种测试时动态多模态潜空间推理框架，通过置信度引导的策略梯度优化来迭代精炼"潜思维令牌 (latent think tokens)"，同时动态检索并注入最相关的视觉 patch，在不需要额外训练的前提下，实现了更高效、更准确的多模态推理。

---

![Figure 1](../images/b28aaa4fa9315c2fc06257bda44c48889b89cbda406c4160b27750da7544168c.jpg)

*Figure 1: Comparison between DMLR and two reasoning paradigms. (A) Text-only reasoning: relies solely on explicit CoT, often causing visual grounding errors and redundant steps. (B) Think-with-Image reasoning: depends on external perception tools, leading to unstable tool calls and extra overhead. (C) DMLR (ours): refines latent think tokens in the latent space through confidence-guided optimization and dynamically injects visual information, achieving self-improving reasoning without additional training while maintaining high efficiency.*

> 💡 **Figure 1 批读**: 这张图展示了三种多模态推理范式的对比。(A) 纯文本推理：依赖显式 CoT，容易产生 visual grounding 错误和冗余步骤。(B) "看图思考"推理：依赖外部感知工具调用，工具调用不稳定且推理开销大。(C) DMLR：在潜空间内通过置信度引导优化 refine 潜思维令牌，并动态注入视觉信息，无需额外训练、保持高效推理。核心洞察是：将推理从显式 token 空间迁移到连续潜空间，配合动态视觉注入，实现了 self-improving reasoning。

> 💡 **问题动机**: 当前 MLLM 的多模态推理面临三大挑战：(1) 纯文本推理缺乏视觉 grounding，导致幻觉；(2) Think-with-Image 方法依赖外部工具，调用不稳定且推理开销大；(3) 现有潜空间推理方法仍需要额外训练且固定在特定位置触发，缺乏灵活性。本文从人类认知中获得灵感：人在思考时会动态回顾视觉信息，尤其在遇到不确定性时。这种"动态交错的推理与感知"是本文的核心设计动机。

---

## 三、Summary

- **核心问题**: 多模态推理中，感知与推理的静态分离导致 visual grounding 不足和计算开销大。
- **核心假设**: 有效的多模态推理应该像人类认知一样，动态地在潜空间内交错推理与视觉感知。
- **核心方案**: DMLR = 置信度引导的潜空间策略梯度优化 + 动态视觉注入策略。
- **核心优势**: Training-free（测试时优化），高推理效率（潜空间操作，无额外序列生成），跨模型/跨任务泛化。
