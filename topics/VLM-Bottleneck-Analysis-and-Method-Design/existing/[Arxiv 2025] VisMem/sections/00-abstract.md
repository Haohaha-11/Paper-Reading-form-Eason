[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要给出整篇论文的压缩版：VLM 在长链生成中会逐步丢失视觉 grounding，VisMem 用短期/长期 latent vision memory 在解码过程中补回视觉证据和语义经验。

---

# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models

Xinlei $\mathrm { { Y u ^ { 1 } } }$ Chengming $\mathrm { { X u ^ { 2 } } }$ Guibin Zhang1 Zhangquan Chen3 Yudong Zhang5   
Yongbo $\mathrm { H e ^ { 4 } }$ Peng-Tao Jiang6 Jiangning Zhang4 Xiaobin $\mathrm { H u ^ { 1 * } }$ Shuicheng Yan1 1National University of Singapore 2Fudan University 3Tsinghua University 4Zhejiang University 5University of Science and Technology of China 6vivo

# Abstract

Despite the remarkable success of Vision-Language Models (VLMs), their performance on a range of complex visual tasks is often hindered by a “visual processing bottleneck”: a propensity to lose grounding in visual evidence and exhibit a deficit in contextualized visual experience during prolonged generation. Drawing inspiration from human cognitive memory theory, which distinguishes short-term visually-dominant memory and longterm semantically-dominant memory, we propose VisMem, a cognitively-aligned framework that equips VLMs with dynamic latent vision memories, a short-term module for finegrained perceptual retention and a long-term module for abstract semantic consolidation. These memories are seamlessly invoked during inference, allowing VLMs to maintain both perceptual fidelity and semantic consistency across thinking and generation. Extensive experiments across diverse visual benchmarks for understanding, reasoning, and generation reveal that VisMem delivers a significant average performance boost of $1 1 . 0 \%$ relative to the vanilla model and outperforms all counterparts, establishing a new paradigm for latent-space memory enhancement. The code will be available: https://github.com/YU-deep/VisMem.git.

> 💡 **摘要批读**: 摘要里有三个关键 claim。第一，瓶颈不是“不会推理”，而是长生成时视觉证据逐步让位给文本上下文。第二，VisMem 把记忆拆成短期视觉主导和长期语义主导：前者保当前图像细节，后者保可迁移的视觉语义。第三，性能证据是 12 个视觉理解/推理/生成 benchmark 上相对 vanilla 平均 +11.0%，所以后文要重点核对这个提升是否来自动态记忆，而不是单纯更多参数或更强训练数据。

> 💡 **批注**: 摘要里“seamlessly invoked during inference”也值得追问。VisMem 的价值不只是 memory content 本身，还在于它没有把 base VLM 改成完全不同的推理流程，而是把记忆 token 插回原 autoregressive generation 中。

---

## 🔖 Section 总结

### 核心洞察
1. VisMem 的核心对象是 latent memory tokens，不是外部检索库、像素级重绘或固定视觉 token 选择。
2. 短期/长期记忆的分工决定了后文实验应分别验证细粒度感知与语义推理收益。
3. 摘要的 +11.0% 是全局 claim，后面要看主结果、跨域、遗忘和效率是否同时成立。
