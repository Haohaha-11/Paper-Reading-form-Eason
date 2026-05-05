[← 返回 README](../README.md)

# Abstract

## 📌 预览
本文件保留论文题名、作者、摘要等开场信息，先抓住方法解决的 one-step SR 核心瓶颈。

---

# ONE-STEP FLOW FOR IMAGE SUPER-RESOLUTION WITH TUNABLE FIDELITY-REALISM TRADE-OFFS

Yuanzhi ${ \bf Z } { \bf h } { \bf u } ^ { \ast \dagger { 1 } , 2 }$ Ruiqing Wang\*1 Shilin ${ \bf L } { \bf u } ^ { 3 }$ Junnan Li2 Hanshu Yan2 Kai Zhang‡1 1Nanjing University 2Rhymes.AI 3Nanyang Technological University

# ABSTRACT

Recent advances in diffusion and flow-based generative models have demonstrated remarkable success in image restoration tasks, achieving superior perceptual quality compared to traditional deep learning approaches. However, these methods either require numerous sampling steps to generate high-quality images, resulting in significant computational overhead, or rely on common model distillation, which usually imposes a fixed fidelity-realism trade-off and thus lacks flexibility. In this paper, we introduce OFTSR, a novel flow-based framework for one-step image super-resolution that can produce outputs with tunable levels of fidelity and realism. Our approach first trains a conditional flow-based superresolution model to serve as a teacher model. We then distill this teacher model by applying a specialized constraint. Specifically, we force the predictions from our one-step student model for same input to lie on the same sampling ODE trajectory of the teacher model. This alignment ensures that the student model’s singlestep predictions from initial states match the teacher’s predictions from a closer intermediate state. Through extensive experiments on datasets including FFHQ $( 2 5 6 \times 2 5 6 )$ , DIV2K, ImageNet $( 2 5 6 \times 2 5 6 )$ ) and real world SR datasets, we demonstrate that OFTSR achieves state-of-the-art performance for one-step image superresolution, while having the ability to flexibly tune the fidelity-realism tradeoff. Code and pre-trained models are available at https://github.com/yuanzhizhu/OFTSR and https://huggingface.co/Yuanzhi/OFTSR, respectively.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

---

## 🔖 Section 总结

### 核心洞察
1. 先记住本文的问题定义、方法名和主要 claim。
2. 后续阅读时回看摘要里的 claim 是否被实验支持。
