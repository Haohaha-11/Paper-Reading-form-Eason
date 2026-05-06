[← 返回 README](../README.md)

# Abstract

## 📌 预览
摘要快速给出本文问题、方法和结果。读完先记住一句话：VisMem 提出 latent vision memory，通过 memory invocation 和 formation 缓解 VLM 长生成中的视觉 grounding 丢失。

---

# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models

Xinlei $\mathrm { { Y u ^ { 1 } } }$ Chengming $\mathrm { { X u ^ { 2 } } }$ Guibin Zhang1 Zhangquan Chen3 Yudong Zhang5   
Yongbo $\mathrm { H e ^ { 4 } }$ Peng-Tao Jiang6 Jiangning Zhang4 Xiaobin $\mathrm { H u ^ { 1 * } }$ Shuicheng Yan1 1National University of Singapore 2Fudan University 3Tsinghua University 4Zhejiang University 5University of Science and Technology of China 6vivo

# Abstract

Despite the remarkable success of Vision-Language Models (VLMs), their performance on a range of complex visual tasks is often hindered by a “visual processing bottleneck”: a propensity to lose grounding in visual evidence and exhibit a deficit in contextualized visual experience during prolonged generation. Drawing inspiration from human cognitive memory theory, which distinguishes short-term visually-dominant memory and longterm semantically-dominant memory, we propose VisMem, a cognitively-aligned framework that equips VLMs with dynamic latent vision memories, a short-term module for finegrained perceptual retention and a long-term module for abstract semantic consolidation. These memories are seamlessly invoked during inference, allowing VLMs to maintain both perceptual fidelity and semantic consistency across thinking and generation. Extensive experiments across diverse visual benchmarks for understanding, reasoning, and generation reveal that VisMem delivers a significant average performance boost of $1 1 . 0 \%$ relative to the vanilla model and outperforms all counterparts, establishing a new paradigm for latent-space memory enhancement. The code will be available: https://github.com/YU-deep/VisMem.git.

> 💡 **批注**: 这段是 latent memory / medical VLM 主线：关注视觉证据如何进入 latent space、如何被记忆/更新/调用，以及是否能支撑可靠诊断。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
