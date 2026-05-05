[← 返回 README](../README.md)

# ABSTRACT

## 📌 预览
摘要给出问题、核心方法和主张结果；读这段时要抓住作者认为的“旧方法瓶颈”和新方法的唯一卖点。

> 💡 **与 OFTSR 主线的关系**: OFTSR 用 conditional flow teacher 和 ODE-trajectory alignment distillation 构建 one-step SR，并保留可调 fidelity-realism trade-off。

---

Recent advances in diffusion and flow-based generative models have demonstrated remarkable success in image restoration tasks, achieving superior perceptual quality compared to traditional deep learning approaches. However, these methods either require numerous sampling steps to generate high-quality images, resulting in significant computational overhead, or rely on common model distillation, which usually imposes a fixed fidelity-realism trade-off and thus lacks flexibility. In this paper, we introduce OFTSR, a novel flow-based framework for one-step image super-resolution that can produce outputs with tunable levels of fidelity and realism. Our approach first trains a conditional flow-based superresolution model to serve as a teacher model. We then distill this teacher model by applying a specialized constraint. Specifically, we force the predictions from our one-step student model for same input to lie on the same sampling ODE trajectory of the teacher model. This alignment ensures that the student model’s singlestep predictions from initial states match the teacher’s predictions from a closer intermediate state. Through extensive experiments on datasets including FFHQ $( 2 5 6 \times 2 5 6 )$ , DIV2K, ImageNet $( 2 5 6 \times 2 5 6 )$ ) and real world SR datasets, we demonstrate that OFTSR achieves state-of-the-art performance for one-step image superresolution, while having the ability to flexibly tune the fidelity-realism tradeoff. Code and pre-trained models are available at https://github.com/yuanzhizhu/OFTSR and https://huggingface.co/Yuanzhi/OFTSR, respectively.

> 💡 **批注**: 这里的关键词是单步推理：作者试图把原本多次 denoising 的生成先验压缩到一次前向中。

---

## 🔖 Section 总结

### 核心洞察

1. 抓住作者定义的核心 gap。
2. 把方法名和它解决的瓶颈对应起来。
3. 记录 claimed result，但等 Experiments 再验证。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Teacher type | conditional flow-based SR model |
| Distillation target | same sampling ODE trajectory alignment |
| Datasets | FFHQ 256×256, DIV2K, ImageNet 256×256 |
