[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文件保留论文题名、作者、摘要等开场信息，先抓住作者定义的问题、方法名和主要 claim。

---

# VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models

Xinlei $\mathrm { { Y u ^ { 1 } } }$ Chengming $\mathrm { { X u ^ { 2 } } }$ Guibin Zhang1 Zhangquan Chen3 Yudong Zhang5   
Yongbo $\mathrm { H e ^ { 4 } }$ Peng-Tao Jiang6 Jiangning Zhang4 Xiaobin $\mathrm { H u ^ { 1 * } }$ Shuicheng Yan1 1National University of Singapore 2Fudan University 3Tsinghua University 4Zhejiang University 5University of Science and Technology of China 6vivo

# Abstract

Despite the remarkable success of Vision-Language Models (VLMs), their performance on a range of complex visual tasks is often hindered by a “visual processing bottleneck”: a propensity to lose grounding in visual evidence and exhibit a deficit in contextualized visual experience during prolonged generation. Drawing inspiration from human cognitive memory theory, which distinguishes short-term visually-dominant memory and longterm semantically-dominant memory, we propose VisMem, a cognitively-aligned framework that equips VLMs with dynamic latent vision memories, a short-term module for finegrained perceptual retention and a long-term module for abstract semantic consolidation. These memories are seamlessly invoked during inference, allowing VLMs to maintain both perceptual fidelity and semantic consistency across thinking and generation. Extensive experiments across diverse visual benchmarks for understanding, reasoning, and generation reveal that VisMem delivers a significant average performance boost of $1 1 . 0 \%$ relative to the vanilla model and outperforms all counterparts, establishing a new paradigm for latent-space memory enhancement. The code will be available: https://github.com/YU-deep/VisMem.git.

> 💡 **批注**: 这是记忆机制段落：重点区分“调用/读出 memory”和“形成/写入 memory”，以及 memory 是否动态变化。

---

## 🔖 Section 总结

### 核心洞察
1. 先记住本文的问题定义、方法名和主要 claim。
2. 后续阅读时回看摘要里的 claim 是否被实验支持。
