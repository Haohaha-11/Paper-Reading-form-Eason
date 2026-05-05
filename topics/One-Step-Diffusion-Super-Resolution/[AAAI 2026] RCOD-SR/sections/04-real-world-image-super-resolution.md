[← 返回 README](../README.md)

# Real-World Image Super-Resolution

## 📌 预览
Method 是核心：关注输入从 LQ 到 latent/feature 的路径、训练目标、控制变量以及与 teacher/先验的交互方式。

> 💡 **与 RCOD-SR 主线的关系**: RCOD-SR 在 one-step diffusion Real-ISR 中引入 latent domain grouping、退化感知采样蒸馏和视觉 prompt 注入，让单步模型在推理时可控地调节 realism。

---

To address the challenge of recovering real-world lowresolution (LR) observations with unknown degradations, the Real-ISR task has been introduced. Due to the complex degradations involved, Real-ISR has been a challenging problem for some time (Ignatov et al. 2017; Liu et al. 2022; Ji et al. 2020). Initial methods trained their models with simple downsampling techniques (Kim, Kwon Lee, and Mu Lee 2016; Zhang et al. 2018b), which led to poor performance on real-world datasets. BSRGAN (Zhang et al. 2021) and Real-ESRGAN (Wang et al. 2021) were among the first to introduce a more realistic synthesis pipeline for LR images, enabling deep learning methods to be applied to real-world scenarios. However, these GAN-based methods often suffer from artifacts and training instability. With the emergence of the powerful pre-trained text-to-image generation model Stable Diffusion (SD) (Rombach et al. 2022), many efforts have been made to leverage its strong generative capabilities for solving the Real-ISR problem, such as DiffBIR (Lin et al. 2023), StableSR (Wang et al. 2024a) and SeeSR (Wu et al. 2024b), and FaithDiff (Chen, Pan, and Dong 2025). These SD-based methods improve fidelity and perceptual quality, but their application is limited due to the significant computational resources and time required. This is primarily due to the dozens to hundreds of timesteps involved in the diffusion denoising process.

> 💡 **批注**: 这里在讨论 fidelity-realism/perception-distortion 张力：SR 既要贴近 GT/LQ 结构，又要生成自然高频细节。

---

## 🔖 Section 总结

### 核心洞察

1. 明确输入、输出、teacher/student 或控制变量。
2. 把每个 loss/模块对应到 fidelity、realism、speed 或 controllability。
3. 关注哪些组件是训练时使用，哪些是推理时仍有成本。

### 关键数字速查

| 指标 | 数值 |
|------|------|
| Inference steps | 1 |
| Control variable | latent-domain group / denoising degree |
| Accepted venue | AAAI 2026 Oral according to arXiv comment |
| Training data change | minimal paradigm modification and original training data claimed |
