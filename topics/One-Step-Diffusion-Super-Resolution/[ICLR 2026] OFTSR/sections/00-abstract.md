[← 返回 README](../README.md)

# ABSTRACT

## 📌 预览
摘要快速给出本文问题、方法和结果。读完先记住一句话：OFTSR 用 conditional flow teacher 和 ODE trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

# ONE-STEP FLOW FOR IMAGE SUPER-RESOLUTION WITH TUNABLE FIDELITY-REALISM TRADE-OFFS

Yuanzhi ${ \bf Z } { \bf h } { \bf u } ^ { \ast \dagger { 1 } , 2 }$ Ruiqing Wang\*1 Shilin ${ \bf L } { \bf u } ^ { 3 }$ Junnan Li2 Hanshu Yan2 Kai Zhang‡1 1Nanjing University 2Rhymes.AI 3Nanyang Technological University

# ABSTRACT

Recent advances in diffusion and flow-based generative models have demonstrated remarkable success in image restoration tasks, achieving superior perceptual quality compared to traditional deep learning approaches. However, these methods either require numerous sampling steps to generate high-quality images, resulting in significant computational overhead, or rely on common model distillation, which usually imposes a fixed fidelity-realism trade-off and thus lacks flexibility. In this paper, we introduce OFTSR, a novel flow-based framework for one-step image super-resolution that can produce outputs with tunable levels of fidelity and realism. Our approach first trains a conditional flow-based superresolution model to serve as a teacher model. We then distill this teacher model by applying a specialized constraint. Specifically, we force the predictions from our one-step student model for same input to lie on the same sampling ODE trajectory of the teacher model. This alignment ensures that the student model’s singlestep predictions from initial states match the teacher’s predictions from a closer intermediate state. Through extensive experiments on datasets including FFHQ $( 2 5 6 \times 2 5 6 )$ , DIV2K, ImageNet $( 2 5 6 \times 2 5 6 )$ ) and real world SR datasets, we demonstrate that OFTSR achieves state-of-the-art performance for one-step image superresolution, while having the ability to flexibly tune the fidelity-realism tradeoff. Code and pre-trained models are available at https://github.com/yuanzhizhu/OFTSR and https://huggingface.co/Yuanzhi/OFTSR, respectively.

> 💡 **批注**: 这段是 one-step SR 主线：关注效率、保真-真实感权衡、扩散/flow 先验或单步生成路径。

---

## 🔖 Section 总结

### 核心洞察
1. 本节对应论文原始大分节，原文已完整保留。
2. 阅读重点是把本节的机制/证据映射到论文主 claim。
3. 后续如有疑问，可在本 section 继续补充更细批注。
